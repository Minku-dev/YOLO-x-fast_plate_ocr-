import csv
import io
import os
import tempfile
import time
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

st.set_page_config(page_title="License plate detect and OCR", layout="wide")

from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer


MIN_HEIGHT = 10
MIN_WIDTH = 10


def check_frame_size(frame):
    height, width = frame.shape[:2]
    return width >= MIN_WIDTH and height >= MIN_HEIGHT


def correct_skew(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is not None:
        rho, theta = lines[0][0]
        angle = (theta * 180 / np.pi) - 90
        if abs(angle) > 5:
            h, w = frame.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            frame = cv2.warpAffine(
                frame, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
    return frame


def adjust_brightness_contrast(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 50 or mean_brightness > 200:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge((l, a, b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return frame


def preprocess_license_plate(frame):
    if not check_frame_size(frame):
        return None
    frame = adjust_brightness_contrast(frame)
    frame = correct_skew(frame)
    return frame


def normalize_plate_text(text):
    return "".join((text or "").split()).upper()


def load_ground_truth_from_bytes(file_bytes):
    truth = defaultdict(list)
    if not file_bytes:
        return truth
    text = file_bytes.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        frame_no = int(row["frame"])
        truth[frame_no].append(normalize_plate_text(row["plate_text"]))
    return truth


#model loader

@st.cache_resource(show_spinner="Đang tải model YOLO")
def load_yolo_model(weights_path):
    return YOLO(weights_path)


@st.cache_resource(show_spinner="Đang tải model OCR")
def load_ocr_engine(model_name):
    return LicensePlateRecognizer(model_name)


#pipeline

def process_video(
    input_path,
    output_path,
    yolo_weights,
    ocr_model_name,
    det_conf_thres,
    high_conf_thres,
    enable_preprocess,
    ground_truth,
    progress_callback=None,
):
    if YOLO is None:
        raise RuntimeError(f"Không import đc ultralytics.YOLO")
    if LicensePlateRecognizer is None:
        raise RuntimeError(f"Không import đc fast_plate_ocr")

    model = load_yolo_model(yolo_weights)
    engine = load_ocr_engine(ocr_model_name)

    has_ground_truth = len(ground_truth) > 0

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video: {input_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1:
        src_fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Không đọc được frame")

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, src_fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    yolo_times_ms, ocr_times_ms, frame_times_ms = [], [], []
    perf_rows = []
    all_det_confs, all_ocr_texts, ocr_confs = [], [], []

    yolo_scored_frames = 0
    yolo_correct_frames = 0
    ocr_total_expected = 0
    ocr_correct = 0

    frame_count = 0
    plate_count = 0
    run_start = time.perf_counter()

    while cap.isOpened():
        frame_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model.predict(frame, conf=det_conf_thres, verbose=False)

        for r in results:
            boxes = r.boxes
            yolo_infer_ms = r.speed.get("inference", 0.0)
            yolo_times_ms.append(yolo_infer_ms)

            frame_ocr_texts = []

            for box in boxes:
                det_conf = float(box.conf[0])
                all_det_confs.append(det_conf)
                x_tl, y_tl, x_br, y_br = box.xyxy[0].to(torch.int64).tolist()

                cv2.rectangle(frame, (x_tl, y_tl), (x_br, y_br), (0, 255, 0), 2)

                y_tl_safe = max(0, y_tl)
                y_br_safe = min(frame.shape[0], y_br)
                x_tl_safe = max(0, x_tl)
                x_br_safe = min(frame.shape[1], x_br)
                plate_region = frame[y_tl_safe:y_br_safe, x_tl_safe:x_br_safe]

                if plate_region.size == 0 or not check_frame_size(plate_region):
                    continue

                if enable_preprocess:
                    processed = preprocess_license_plate(plate_region)
                    if processed is None:
                        continue
                    plate_region = processed

                plate_array = cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB)

                ocr_conf = None
                ocr_start = time.perf_counter()
                try:
                    pred = engine.run_one(plate_array, return_confidence=True)
                    concat_number = pred.plate if pred.plate else "Not recognized"
                    if pred.char_probs is not None and len(pred.char_probs) > 0:
                        ocr_conf = float(np.mean(pred.char_probs))
                except Exception:
                    concat_number = "Error"
                ocr_infer_ms = (time.perf_counter() - ocr_start) * 1000
                ocr_times_ms.append(ocr_infer_ms)
                plate_count += 1

                all_ocr_texts.append(concat_number)
                if ocr_conf is not None:
                    ocr_confs.append(ocr_conf)
                frame_ocr_texts.append(concat_number)

                cv2.putText(
                    img=frame,
                    text=concat_number,
                    org=(x_tl, max(0, y_tl - 10)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 255, 0),
                    thickness=2,
                )

                perf_rows.append({
                    "frame": frame_count,
                    "plate_text": concat_number,
                    "det_conf": round(det_conf, 4),
                    "ocr_conf": round(ocr_conf, 4) if ocr_conf is not None else "",
                    "yolo_ms": round(yolo_infer_ms, 3),
                    "ocr_ms": round(ocr_infer_ms, 3),
                })

            if has_ground_truth and frame_count in ground_truth:
                expected = list(ground_truth[frame_count])

                yolo_scored_frames += 1
                if len(boxes) == len(expected):
                    yolo_correct_frames += 1

                remaining = expected.copy()
                for txt in frame_ocr_texts:
                    norm = normalize_plate_text(txt)
                    if norm in remaining:
                        remaining.remove(norm)
                        ocr_correct += 1
                ocr_total_expected += len(expected)

        frame_ms = (time.perf_counter() - frame_start) * 1000
        frame_times_ms.append(frame_ms)
        current_fps = 1000.0 / frame_ms if frame_ms > 0 else 0.0

        hud = f"FPS: {current_fps:.1f}"
        cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(frame)

        if progress_callback is not None:
            progress_callback(frame_count, total_frames)

    cap.release()
    out.release()

    total_time_s = time.perf_counter() - run_start

    summary = {
        "frame_count": frame_count,
        "total_time_s": total_time_s,
        "plate_count": plate_count,
        "yolo_times_ms": yolo_times_ms,
        "ocr_times_ms": ocr_times_ms,
        "frame_times_ms": frame_times_ms,
        "all_det_confs": all_det_confs,
        "all_ocr_texts": all_ocr_texts,
        "ocr_confs": ocr_confs,
        "has_ground_truth": has_ground_truth,
        "yolo_scored_frames": yolo_scored_frames,
        "yolo_correct_frames": yolo_correct_frames,
        "ocr_total_expected": ocr_total_expected,
        "ocr_correct": ocr_correct,
        "high_conf_thres": high_conf_thres,
    }

    return perf_rows, summary




st.title("Nhận diện biển số xe")

with st.sidebar:
    st.header("Cấu hình model")
    yolo_weights_file = st.file_uploader("File trọng số YOLO (.pt)", type="pt")
    ocr_model_name = st.text_input("Phiên bản OCR (fast-plate-ocr)", value="cct-s-v2-global-model")

    st.header("Tham số detect")
    det_conf_thres = st.slider("Ngưỡng tin cậy", 0.0, 1.0, 0.25, 0.01)
    high_conf_thres = st.slider("Ngưỡng tin cậy cao ", 0.0, 1.0, 0.5, 0.01)
    enable_preprocess = st.checkbox("Bật tiền xử lí ảnh", value=False)

    st.header("Tải Ground truth (.csv)")
    gt_file = st.file_uploader("File CSV ground truth (cột: frame, plate_text)", type=["csv"])

uploaded_video = st.file_uploader("Tải lên video để xử lý", type=["mp4", "avi", "mov", "mkv"])

run_button = st.button(
    "RUN", type="primary", disabled=uploaded_video is None or yolo_weights_file is None
)

if uploaded_video is not None and yolo_weights_file is None:
    st.warning("Tải lên file trọng số YOLO (.pt) ở sidebar trước khi chạy.")

if "anpr_results" not in st.session_state:
    st.session_state["anpr_results"] = None

if run_button and uploaded_video is not None and yolo_weights_file is not None:
    ground_truth = load_ground_truth_from_bytes(gt_file.read()) if gt_file is not None else defaultdict(list)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input_video")
        output_path = os.path.join(tmpdir, "output_video.mp4")
        yolo_weights_path = os.path.join(tmpdir, "weights.pt")

        with open(input_path, "wb") as f:
            f.write(uploaded_video.read())

        with open(yolo_weights_path, "wb") as f:
            f.write(yolo_weights_file.read())

        progress_bar = st.progress(0.0, text="Đang xử lý")
        status_text = st.empty()

        def _on_progress(frame_count, total_frames):
            if total_frames:
                pct = min(frame_count / total_frames, 1.0)
                progress_bar.progress(pct, text=f"Đang xử lý frame {frame_count}/{total_frames}")
            else:
                status_text.write(f"Đang xử lý frame {frame_count}...")

        try:
            with st.spinner("Đang chạy pipeline detect + OCR"):
                perf_rows, summary = process_video(
                    input_path=input_path,
                    output_path=output_path,
                    yolo_weights=yolo_weights_path,
                    ocr_model_name=ocr_model_name,
                    det_conf_thres=det_conf_thres,
                    high_conf_thres=high_conf_thres,
                    enable_preprocess=enable_preprocess,
                    ground_truth=ground_truth,
                    progress_callback=_on_progress,
                )
        except Exception as e:
            st.error(f"Lỗi khi xử lý file: {e}")
            st.stop()

        progress_bar.progress(1.0, text="Hoàn tất")
        with open(output_path, "rb") as f:
            video_bytes = f.read()

        st.session_state["anpr_results"] = {
            "perf_rows": perf_rows,
            "summary": summary,
            "video_bytes": video_bytes,
        }

results = st.session_state["anpr_results"]
if results is not None:
    perf_rows = results["perf_rows"]
    summary = results["summary"]
    video_bytes = results["video_bytes"]

    if True:
        #Thống kê hiệu năng
        st.subheader("Tổng quan hiệu năng")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Số frame đã xử lý", summary["frame_count"])
        col2.metric("Số biển số xe đã OCR", summary["plate_count"])
        col3.metric("Thời gian xử lý (s)", f"{summary['total_time_s']:.2f}")
        overall_fps = summary["frame_count"] / summary["total_time_s"] if summary["total_time_s"] > 0 else 0
        col4.metric("FPS trung bình", f"{overall_fps:.2f}")

        if summary["yolo_times_ms"]:
            st.write(f"YOLO inference: trung bình {np.mean(summary['yolo_times_ms']):.2f} ms/frame")
        if summary["ocr_times_ms"]:
            st.write(f"OCR inference: trung bình {np.mean(summary['ocr_times_ms']):.2f} ms/biển số")

        # Độ chính xác
        st.subheader("Độ chính xác")
        if summary["has_ground_truth"]:
            if summary["yolo_scored_frames"]:
                yolo_acc = summary["yolo_correct_frames"] / summary["yolo_scored_frames"] * 100
                st.write(
                    f"YOLO detection accuracy: **{yolo_acc:.2f}%** "
                    f"({summary['yolo_correct_frames']}/{summary['yolo_scored_frames']} frame đúng số lượng box)"
                )
            if summary["ocr_total_expected"]:
                ocr_acc = summary["ocr_correct"] / summary["ocr_total_expected"] * 100
                st.write(
                    f"OCR accuracy: **{ocr_acc:.2f}%** "
                    f"({summary['ocr_correct']}/{summary['ocr_total_expected']} biển số đọc đúng)"
                )
        else:
            st.info("Không có ground truth -> Chỉ hiển thị các chỉ số proxy dựa trên độ tin cậy.")
            if summary["all_det_confs"]:
                high_conf_rate = (
                    sum(c >= summary["high_conf_thres"] for c in summary["all_det_confs"])
                    / len(summary["all_det_confs"])
                    * 100
                )
                st.write(f"Tỷ lệ detect tin cậy cao (>= {summary['high_conf_thres']}): **{high_conf_rate:.2f}%**")
            if summary["all_ocr_texts"]:
                recognized = sum(1 for t in summary["all_ocr_texts"] if t not in ("Error", "Not recognized"))
                recog_rate = recognized / len(summary["all_ocr_texts"]) * 100
                st.write(f"Tỷ lệ OCR đọc được (không lỗi /không rỗng): **{recog_rate:.2f}%**")
            if summary["ocr_confs"]:
                st.write(f"Độ tin cậy OCR trung bình: **{np.mean(summary['ocr_confs']) * 100:.2f}%**")

        #Output video
        st.subheader("Video kết quả")
        st.download_button(
            "Tải video kết quả (.mp4)",
            data=video_bytes,
            file_name="output_video.mp4",
            mime="video/mp4",
            key="download_video_btn",
        )
        #Bảng thống kê
        st.subheader("Bảng các biển số đã nhận diện")
        if perf_rows:
            df = pd.DataFrame(perf_rows)
            st.dataframe(df, use_container_width=True)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Tải xuống bảng thống kê biển số (.csv)",
                data=csv_bytes,
                file_name="performance_log.csv",
                mime="text/csv",
                key="download_csv_btn",
            )
        else:
            st.write("Không phát hiện biển số nào trong video.")

        if st.button("Xoá kết quả và chạy video mới"):
            st.session_state["anpr_results"] = None
            st.rerun()

elif uploaded_video is None:
    st.info("Hãy tải video lên và nhấn 'RUN' để bắt đầu")
