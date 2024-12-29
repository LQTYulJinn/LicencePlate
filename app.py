import os
import csv
import uuid
import tempfile
from datetime import datetime

import re  # Dùng regex để loại bỏ ký tự đặc biệt
import cv2
import easyocr
import numpy as np
import streamlit as st
import pandas as pd
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# ==============================
# TẠO THƯ MỤC LƯU ẢNH BIỂN SỐ
# ==============================
PLATES_FOLDER = "plates"
if not os.path.exists(PLATES_FOLDER):
    os.makedirs(PLATES_FOLDER)

# ==============================
# BẢNG ÁNH XẠ 2 SỐ ĐẦU -> TỈNH
# ==============================
province_map = {
    "11": "Cao Bằng",
    "12": "Lạng Sơn",
    "14": "Quảng Ninh",
    "15": "Hải Phòng",
    "16": "Hải Phòng",
    "17": "Thái Bình",
    "18": "Nam Định",
    "19": "Phú Thọ",
    "20": "Thái Nguyên",
    "21": "Yên Bái",
    "22": "Tuyên Quang",
    "23": "Hà Giang",
    "24": "Lào Cai",
    "25": "Lai Châu",
    "26": "Sơn La",
    "27": "Điện Biên",
    "28": "Hòa Bình",
    "29": "Hà Nội",
    "30": "Hà Nội",
    "31": "Hà Nội",
    "32": "Hà Nội",
    "33": "Hà Nội",
    "34": "Hải Dương",
    "35": "Ninh Bình",
    "36": "Thanh Hóa",
    "37": "Nghệ An",
    "38": "Hà Tĩnh",
    "39": "Đồng Nai",
    "40": "Hà Nội",
    "41": "TP. HCM",
    "43": "Đà Nẵng",
    "47": "Đắk Lắk",
    "48": "Đắk Nông",
    "49": "Lâm Đồng",
    "50": "TP. HCM",
    "51": "TP. HCM",
    "52": "TP. HCM",
    "53": "TP. HCM",
    "54": "TP. HCM",
    "55": "TP. HCM",
    "56": "TP. HCM",
    "57": "TP. HCM",
    "58": "TP. HCM",
    "59": "TP. HCM",
    "60": "Đồng Nai",
    "61": "Bình Dương",
    "62": "Long An",
    "63": "Tiền Giang",
    "64": "Vĩnh Long",
    "65": "Cần Thơ",
    "66": "Đồng Tháp",
    "67": "An Giang",
    "68": "Kiên Giang",
    "69": "Cà Mau",
    "70": "Tây Ninh",
    "71": "Bến Tre",
    "72": "Bà Rịa - Vũng Tàu",
    "73": "Quảng Bình",
    "74": "Quảng Trị",
    "75": "Thừa Thiên Huế",
    "76": "Quảng Ngãi",
    "77": "Bình Định",
    "78": "Phú Yên",
    "79": "Khánh Hòa",
    "81": "Gia Lai",
    "82": "Kon Tum",
    "83": "Sóc Trăng",
    "84": "Trà Vinh",
    "85": "Ninh Thuận",
    "86": "Bình Thuận",
    "88": "Vĩnh Phúc",
    "89": "Hưng Yên",
    "90": "Hà Nam",
    "92": "Quảng Nam",
    "93": "Bình Phước",
    "94": "Bạc Liêu",
    "95": "Hậu Giang",
    "97": "Bắc Kạn",
    "98": "Bắc Giang",
    "99": "Bắc Ninh",
}

# -----------------------------
# HÀM XỬ LÝ CSV
# -----------------------------
def load_csv_to_dataframe(csv_path="plates_database.csv"):
    """Đọc CSV thành pandas DataFrame (nếu có)."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return df

def save_plate_info_to_csv(data_list, csv_path="plates_database.csv"):
    """
    Lưu danh sách biển số vào CSV, format:
      [id, class_name, confidence, raw_text, province, bbox, timestamp, image_path]
    """
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "id",
                "class_name",
                "confidence",
                "raw_text",
                "province",
                "bbox",
                "timestamp",
                "image_path",
            ])
        for item in data_list:
            writer.writerow([
                item["id"],
                item["class_name"],
                item["confidence"],
                item["raw_text"],
                item["province"],
                item["bbox"],
                item["timestamp"],
                item["image_path"],
            ])

def check_plate_exists_in_csv(plate_text, csv_path="plates_database.csv"):
    """
    Tìm trong CSV bằng cột 'raw_text'.
    Trả về dict nếu tìm thấy, None nếu không.
    """
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            if row[3].strip() == plate_text.strip():
                return {
                    "id": row[0],
                    "class_name": row[1],
                    "confidence": row[2],
                    "raw_text": row[3],
                    "province": row[4],
                    "bbox": row[5],
                    "timestamp": row[6],
                    "image_path": row[7],
                }
    return None

def check_plate_exists_by_id(plate_id, csv_path="plates_database.csv"):
    """
    Tìm trong CSV bằng cột 'id'.
    Trả về dict nếu tìm thấy, None nếu không.
    """
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            if row[0].strip() == plate_id.strip():
                return {
                    "id": row[0],
                    "class_name": row[1],
                    "confidence": row[2],
                    "raw_text": row[3],
                    "province": row[4],
                    "bbox": row[5],
                    "timestamp": row[6],
                    "image_path": row[7],
                }
    return None

def remove_plate_by_id(plate_id, csv_path="plates_database.csv"):
    """
    Xoá dòng trong CSV có cột 'id' = plate_id.
    Trả về dict vừa xoá hoặc None nếu không tìm thấy.
    """
    if not os.path.exists(csv_path):
        return None

    removed_item = None
    rows = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header:
            rows.append(header)

        for row in reader:
            if not row:
                continue
            if row[0].strip() == plate_id.strip():
                removed_item = {
                    "id": row[0],
                    "class_name": row[1],
                    "confidence": row[2],
                    "raw_text": row[3],
                    "province": row[4],
                    "bbox": row[5],
                    "timestamp": row[6],
                    "image_path": row[7],
                }
            else:
                rows.append(row)

    # Ghi lại CSV (đã bỏ dòng tương ứng)
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    return removed_item

def remove_plate_by_text(plate_text, csv_path="plates_database.csv"):
    """
    Xoá dòng trong CSV có cột 'raw_text' = plate_text.
    Trả về dict dòng xoá hoặc None nếu không tìm thấy.
    Chỉ xoá 1 dòng (đầu tiên) nếu trùng.
    """
    if not os.path.exists(csv_path):
        return None

    removed_item = None
    rows = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header:
            rows.append(header)

        for row in reader:
            if not row:
                continue
            if row[3].strip() == plate_text.strip():
                removed_item = {
                    "id": row[0],
                    "class_name": row[1],
                    "confidence": row[2],
                    "raw_text": row[3],
                    "province": row[4],
                    "bbox": row[5],
                    "timestamp": row[6],
                    "image_path": row[7],
                }
                break
            else:
                rows.append(row)

    # Ghi lại CSV
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    return removed_item

# -----------------------------
# EASYOCR + YOLO
# -----------------------------
def generate_plate_id():
    return str(uuid.uuid4())

def get_province_from_plate(plate_text: str) -> str:
    # Dùng plate_text đã được làm sạch (A-Z0-9)
    plate_text = plate_text.replace(" ", "").upper()
    if len(plate_text) < 2:
        return "Unknown"
    prefix = plate_text[:2]
    if not prefix.isdigit():
        return "Unknown"
    return province_map.get(prefix, "Unknown")

def save_plate_image(plate_img_rgb, plate_id):
    """Lưu ảnh cắt biển số thành file <id>.jpg trong thư mục 'plates/'."""
    image_path = os.path.join(PLATES_FOLDER, f"{plate_id}.jpg")
    plate_img_bgr = cv2.cvtColor(plate_img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, plate_img_bgr)
    return image_path

def detect_and_read_license_plate_easyocr(
        frame_rgb: np.ndarray,
        model: YOLO,
        reader: easyocr.Reader,
        conf_thres: float = 0.25
):
    """
    Dò object bằng YOLO, cắt vùng biển số, OCR bằng EasyOCR,
    lưu ảnh cắt + trả về info (id, raw_text, province, image_path, ...)
    """
    # Dự đoán
    results = model.predict(source=frame_rgb, conf=conf_thres)
    detection = results[0]
    class_names = model.names

    ocr_data = []
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    for box_data in detection.boxes.data.tolist():
        x1, y1, x2, y2, conf, class_id = box_data
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_name = class_names[int(class_id)]

        # Cắt biển số (RGB)
        plate_img_rgb = frame_rgb[y1:y2, x1:x2]

        # OCR với EasyOCR
        ocr_out = reader.readtext(plate_img_rgb, detail=1)
        recognized_texts = [txt for (_, txt, _) in ocr_out]
        raw_text = " ".join(recognized_texts).strip()

        # ==============
        # XÓA KÝ TỰ KHÔNG PHẢI A-Z0-9 (bỏ khoảng trắng, dấu ., -, ...)
        cleaned_text = re.sub(r'[^A-Za-z0-9]+', '', raw_text)
        # Chuyển sang chữ hoa
        cleaned_text = cleaned_text.upper()
        # ==============

        # Lấy tỉnh
        province = get_province_from_plate(cleaned_text)

        # Tạo ID, timestamp
        plate_id = generate_plate_id()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Lưu ảnh cắt (nếu bounding box hợp lệ)
        image_path = ""
        if plate_img_rgb.size > 0:
            image_path = save_plate_image(plate_img_rgb, plate_id)

        # Ghi vào list
        ocr_data.append({
            "id": plate_id,
            "class_name": cls_name,
            "confidence": float(conf),
            # Lưu chuỗi đã loại bỏ ký tự đặc biệt
            "raw_text": cleaned_text,
            "province": province,
            "bbox": [x1, y1, x2, y2],
            "timestamp": now_str,
            "image_path": image_path,
        })

        # Vẽ bounding box + text
        color = (0, 255, 0)
        thickness = 2
        display_text = f"{cls_name}: {cleaned_text} ({province})"
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            frame_bgr,
            display_text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    annotated_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, ocr_data

def process_video(
        video_path: str,
        model_path: str,
        conf_thres: float = 0.25,
        progress_callback=None
):
    """
    Mở video, đọc từng frame -> detect_and_read_license_plate_easyocr
    ghi ra 1 video tạm + trả về all_frames_data
    """
    model = YOLO(model_path)
    reader = easyocr.Reader(['en'], gpu=False)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Không mở được video!")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video_path = temp_video.name
    temp_video.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    all_frames_data = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        current_frame += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        annotated_rgb, ocr_data = detect_and_read_license_plate_easyocr(
            frame_rgb, model, reader, conf_thres
        )
        all_frames_data.append(ocr_data)

        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        out.write(annotated_bgr)

        if progress_callback and frame_count > 0:
            progress_callback(current_frame / frame_count)

    cap.release()
    out.release()

    return temp_video_path, all_frames_data

def detect_plate_in_image(image_rgb: np.ndarray, model: YOLO, reader: easyocr.Reader, conf_thres=0.25):
    """
    Detect + OCR nhanh, trả về list các biển số (raw_text) đã loại bỏ ký tự đặc biệt
    (Không vẽ, không lưu).
    """
    results = model.predict(source=image_rgb, conf=conf_thres)
    detection = results[0]

    raw_texts = []
    for box_data in detection.boxes.data.tolist():
        x1, y1, x2, y2, conf, class_id = box_data
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        plate_img_rgb = image_rgb[y1:y2, x1:x2]

        # OCR
        ocr_out = reader.readtext(plate_img_rgb, detail=1)
        recognized_texts = [txt for (_, txt, _) in ocr_out]
        raw_text = " ".join(recognized_texts).strip()

        # Loại bỏ ký tự đặc biệt, giữ A-Z0-9
        cleaned_text = re.sub(r'[^A-Za-z0-9]+', '', raw_text).upper()

        raw_texts.append(cleaned_text)

    return raw_texts

# -----------------------------
# VIDEO PROCESSOR FOR WEBCAM
# -----------------------------
class LicensePlateProcessor(VideoProcessorBase):
    def __init__(self, model_path, conf_thres):
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.conf_thres = conf_thres

    def process(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect and annotate license plates
        annotated_rgb, ocr_data = detect_and_read_license_plate_easyocr(
            img_rgb, self.model, self.reader, self.conf_thres
        )

        # Convert back to BGR for display
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        # Save new license plates to CSV
        plates_to_save = []
        for plate_info in ocr_data:
            raw_text = plate_info["raw_text"]
            if raw_text:
                existing = check_plate_exists_in_csv(raw_text)
                if not existing:
                    plates_to_save.append(plate_info)
        if plates_to_save:
            save_plate_info_to_csv(plates_to_save)

        # Convert to RGB for Streamlit display
        annotated_rgb_display = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return av.VideoFrame.from_ndarray(annotated_rgb_display, format="rgb24")

# ====================
# STREAMLIT APP
# ====================
def process_tab(model_path, conf_thres):
    st.subheader("1) Xử Lý Ảnh & Video + Lưu CSV")

    mode = st.radio("Chế độ", ["Ảnh", "Video"], horizontal=True)

    if mode == "Ảnh":
        st.markdown("### 📷 Xử Lý Ảnh")
        uploaded_file = st.file_uploader(
            "Chọn file ảnh (jpg, jpeg, png)...",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📸 Ảnh Gốc")
                st.image(image_rgb, channels="RGB", use_container_width=True)
            with col2:
                st.empty()

            if st.button("🔍 Chạy Phát Hiện + OCR"):
                with st.spinner("Đang xử lý..."):
                    try:
                        model = YOLO(model_path)
                        reader = easyocr.Reader(['en'], gpu=False)
                        annotated_rgb, ocr_data = detect_and_read_license_plate_easyocr(
                            image_rgb, model, reader, conf_thres
                        )

                        # Kiểm tra trùng & Lưu mới
                        plates_to_save = []
                        for plate_info in ocr_data:
                            raw_text = plate_info["raw_text"]
                            if raw_text:
                                existing = check_plate_exists_in_csv(raw_text)
                                if existing:
                                    st.warning(
                                        f"⚠️ Biển số '{existing['raw_text']}' đã tồn tại (ID: {existing['id']})."
                                    )
                                else:
                                    plates_to_save.append(plate_info)

                        if plates_to_save:
                            save_plate_info_to_csv(plates_to_save)
                            st.success(f"✅ Đã lưu {len(plates_to_save)} biển số mới vào CSV.")

                        with col2:
                            st.subheader("🖼️ Ảnh Đã Annotate")
                            st.image(annotated_rgb, channels="RGB", use_container_width=True)

                        st.markdown("### 📝 Kết Quả OCR")
                        if ocr_data:
                            for i, item in enumerate(ocr_data, start=1):
                                with st.expander(f"📄 Biển số #{i}"):
                                    st.write(f"- **ID:** {item['id']}")
                                    st.write(f"- **Class:** {item['class_name']}")
                                    st.write(f"- **Confidence:** {item['confidence']:.2f}")
                                    st.write(f"- **Raw Text:** {item['raw_text']}")
                                    st.write(f"- **Province:** {item['province']}")
                                    st.write(f"- **BBox:** {item['bbox']}")
                                    st.write(f"- **Timestamp:** {item['timestamp']}")
                                    st.write(f"- **Image Path:** {item['image_path']}")
                        else:
                            st.info("ℹ️ Không phát hiện biển số nào.")

                    except Exception as e:
                        st.error(f"❌ Lỗi: {e}")

    elif mode == "Video":
        video_mode = st.radio("Chọn nguồn video", ["Upload Video", "Sử dụng Webcam"], horizontal=True)

        if video_mode == "Upload Video":
            st.markdown("### 🎥 Xử Lý Video Từ File")
            uploaded_video = st.file_uploader(
                "Chọn file video (mp4, mov, avi, mkv...)",
                type=["mp4", "mov", "avi", "mkv"]
            )
            if uploaded_video is not None:
                temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_video_file.write(uploaded_video.read())
                temp_video_file_path = temp_video_file.name
                temp_video_file.close()

                st.video(uploaded_video)

                if st.button("🔄 Chạy Phát Hiện + OCR Cho Video"):
                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    def update_progress(progress):
                        progress_bar.progress(progress)
                        progress_text.text(f"📈 Tiến độ: {int(progress * 100)}%")

                    with st.spinner("Đang xử lý video..."):
                        try:
                            processed_video_path, all_frames_data = process_video(
                                temp_video_file_path,
                                model_path,
                                conf_thres,
                                progress_callback=update_progress
                            )
                            st.success("✅ Xử lý xong!")

                            # Dàn phẳng dữ liệu OCR
                            flattened_ocr = [plate for frame in all_frames_data for plate in frame]

                            # Kiểm tra trùng & Lưu mới
                            new_plates = []
                            for plate_info in flattened_ocr:
                                raw_text = plate_info["raw_text"]
                                if raw_text:
                                    exist = check_plate_exists_in_csv(raw_text)
                                    if exist:
                                        st.warning(
                                            f"⚠️ Biển số '{exist['raw_text']}' đã tồn tại (ID: {exist['id']})."
                                        )
                                    else:
                                        new_plates.append(plate_info)

                            if new_plates:
                                save_plate_info_to_csv(new_plates)
                                st.success(f"✅ Đã lưu {len(new_plates)} biển số mới vào CSV.")

                            st.markdown("### 📝 Video Đã Annotate")
                            st.video(processed_video_path)

                            with st.expander("📋 Chi Tiết OCR Theo Từng Frame"):
                                for frame_idx, frame_data in enumerate(all_frames_data):
                                    with st.expander(f"Frame {frame_idx + 1}"):
                                        if not frame_data:
                                            st.write("ℹ️ Không phát hiện biển số.")
                                        else:
                                            for i, item in enumerate(frame_data, start=1):
                                                st.write(f"**Biển số #{i}:**")
                                                st.write(f"- **ID:** {item['id']}")
                                                st.write(f"- **Class:** {item['class_name']}")
                                                st.write(f"- **Confidence:** {item['confidence']:.2f}")
                                                st.write(f"- **Raw Text:** {item['raw_text']}")
                                                st.write(f"- **Province:** {item['province']}")
                                                st.write(f"- **BBox:** {item['bbox']}")
                                                st.write(f"- **Timestamp:** {item['timestamp']}")
                                                st.write(f"- **Image Path:** {item['image_path']}")
                                                st.markdown("---")

                            # Xoá file tạm
                            os.remove(processed_video_path)

                        except Exception as e:
                            st.error(f"❌ Lỗi: {e}")

                    progress_bar.empty()
                    progress_text.empty()

            st.markdown("---")

        elif video_mode == "Sử dụng Webcam":
            st.markdown("### 📹 Xử Lý Video Từ Webcam")

            # Initialize webrtc_streamer
            ctx = webrtc_streamer(
                key="license-plate-webcam",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
                video_processor_factory=lambda: LicensePlateProcessor(model_path, conf_thres),
                async_processing=True,
            )

            if ctx.video_processor:
                st.info("🔍 Đang chờ phát hiện biển số...")

def check_tab():
    st.subheader("2) Kiểm Tra Biển Số")
    """
    - Tìm bằng **ID**
    - Tìm bằng **raw_text**
    - Hoặc upload **Ảnh** (chỉ kiểm tra, không lưu mới)
    """

    # Tìm bằng ID
    st.markdown("#### 🔍 Tìm Biển Số Bằng ID")
    with st.form("search_id_form"):
        search_id = st.text_input("Nhập ID:")
        submitted_id = st.form_submit_button("Kiểm tra bằng ID")
    if submitted_id:
        if search_id:
            info = check_plate_exists_by_id(search_id)
            if info:
                st.success("✅ Tìm thấy ID trong CSDL!")
                st.write(info)
                if info["image_path"] and os.path.exists(info["image_path"]):
                    st.image(info["image_path"], caption="🖼️ Ảnh Biển Số Đã Lưu", use_container_width=True)
            else:
                st.error("❌ Không tìm thấy ID này.")
        else:
            st.warning("⚠️ Vui lòng nhập ID.")

    st.markdown("---")

    # Tìm bằng raw_text
    st.markdown("#### 🔍 Tìm Biển Số Bằng raw_text")
    with st.form("search_text_form"):
        search_text = st.text_input("Nhập biển số (VD: 30E12345):")
        submitted_text = st.form_submit_button("Kiểm tra bằng Biển Số")
    if submitted_text:
        if search_text:
            info2 = check_plate_exists_in_csv(search_text)
            if info2:
                st.success("✅ Đã tìm thấy biển số trong CSV!")
                st.write(info2)
                if info2["image_path"] and os.path.exists(info2["image_path"]):
                    st.image(info2["image_path"], caption="🖼️ Ảnh Biển Số Đã Lưu", use_container_width=True)
            else:
                st.error("❌ Không tìm thấy biển số này trong CSV.")
        else:
            st.warning("⚠️ Vui lòng nhập biển số.")

    st.markdown("---")

    # Kiểm tra bằng Ảnh
    st.markdown("#### 📷 Kiểm Tra Bằng Ảnh (Không Lưu)")
    with st.form("check_image_form"):
        check_image = st.file_uploader("Chọn ảnh để kiểm tra biển số...", type=["jpg", "jpeg", "png"])
        submitted_check_image = st.form_submit_button("🔍 Phân Tích Ảnh Kiểm Tra")
    if submitted_check_image and check_image is not None:
        file_bytes = np.asarray(bytearray(check_image.read()), dtype=np.uint8)
        check_img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        check_img_rgb = cv2.cvtColor(check_img_bgr, cv2.COLOR_BGR2RGB)
        st.image(check_img_rgb, caption="🖼️ Ảnh Kiểm Tra", channels="RGB", use_container_width=True)

        with st.spinner("🔍 Đang OCR..."):
            try:
                model = YOLO("best.pt")
                reader = easyocr.Reader(['en'], gpu=False)

                found_plates = detect_plate_in_image(check_img_rgb, model, reader, conf_thres=0.25)
                if not found_plates:
                    st.warning("⚠️ Không phát hiện biển số nào trong ảnh.")
                else:
                    st.success("✅ Biển số OCR được:")
                    for plate_str in found_plates:
                        with st.expander(f"📄 Biển số: {plate_str}"):
                            exist_info = check_plate_exists_in_csv(plate_str)
                            if exist_info:
                                st.write(f"**ID:** {exist_info['id']}")
                                st.success(f"✅ Biển số '{plate_str}' đã tồn tại trong CSDL.")
                                if exist_info["image_path"] and os.path.exists(exist_info["image_path"]):
                                    st.image(
                                        exist_info["image_path"],
                                        caption="🖼️ Ảnh Biển Số Đã Lưu (Trong CSDL)",
                                        use_container_width=True
                                    )
                            else:
                                st.error(f"❌ Biển số '{plate_str}' chưa có trong CSDL.")

            except Exception as e:
                st.error(f"❌ Lỗi: {e}")

    st.markdown("---")

    # Hiển thị toàn bộ CSV
    st.markdown("### 📊 Database Hiện Tại")
    df_check = load_csv_to_dataframe()
    if df_check is not None:
        st.dataframe(df_check, use_container_width=True)
    else:
        st.info("ℹ️ Chưa có dữ liệu trong CSV.")

def checkout_tab():
    st.subheader("3) Check Out (Xoá Khỏi DB)")
    """
    Xoá khỏi CSV bằng:
    - ID
    - raw_text
    - Ảnh (tương tự kiểm tra, nếu có trong CSV thì xóa)
    """

    # 1) Xóa bằng ID
    st.markdown("#### 🗑️ Xóa Bằng ID")
    with st.form("remove_id_form"):
        remove_id = st.text_input("Nhập ID cần xóa:")
        submitted_remove_id = st.form_submit_button("Xóa theo ID")
    if submitted_remove_id:
        if remove_id:
            removed_item = remove_plate_by_id(remove_id)
            if removed_item:
                st.success(f"✅ Đã xóa ID={removed_item['id']} - Biển số={removed_item['raw_text']}.")
                if removed_item["image_path"] and os.path.exists(removed_item["image_path"]):
                    st.image(removed_item["image_path"], caption="🖼️ Ảnh Biển Số Đã Xoá", use_container_width=True)
                    # Optionally delete the image file
                    # os.remove(removed_item["image_path"])
            else:
                st.error("❌ Không tìm thấy ID trong DB.")
        else:
            st.warning("⚠️ Vui lòng nhập ID cần xóa.")

    st.markdown("---")

    # 2) Xóa bằng raw_text
    st.markdown("#### 🗑️ Xóa Bằng Biển Số (raw_text)")
    with st.form("remove_text_form"):
        remove_text = st.text_input("Nhập biển số cần xóa:")
        submitted_remove_text = st.form_submit_button("Xóa theo Biển Số")
    if submitted_remove_text:
        if remove_text:
            removed_item2 = remove_plate_by_text(remove_text)
            if removed_item2:
                st.success(
                    f"✅ Đã xóa Biển số={removed_item2['raw_text']} - ID={removed_item2['id']} khỏi DB."
                )
                if removed_item2["image_path"] and os.path.exists(removed_item2["image_path"]):
                    st.image(removed_item2["image_path"], caption="🖼️ Ảnh Biển Số Đã Xoá", use_container_width=True)
                    # os.remove(removed_item2["image_path"])
            else:
                st.error("❌ Không tìm thấy biển số này trong DB.")
        else:
            st.warning("⚠️ Vui lòng nhập biển số cần xóa.")

    st.markdown("---")

    # 3) Xóa bằng Ảnh
    st.markdown("#### 🗑️ Xóa Bằng Ảnh")
    with st.form("checkout_image_form"):
        checkout_image = st.file_uploader(
            "Chọn ảnh chứa biển số cần xóa...",
            type=["jpg", "jpeg", "png"],
            key="checkout_image"
        )
        submitted_checkout_image = st.form_submit_button("🗑️ Xoá Khỏi DB Bằng Ảnh Này")
    if submitted_checkout_image and checkout_image is not None:
        file_bytes = np.asarray(bytearray(checkout_image.read()), dtype=np.uint8)
        chkout_img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        chkout_img_rgb = cv2.cvtColor(chkout_img_bgr, cv2.COLOR_BGR2RGB)
        st.image(chkout_img_rgb, caption="🖼️ Ảnh Check Out", channels="RGB", use_container_width=True)

        with st.spinner("🔍 Đang OCR..."):
            try:
                model = YOLO("best.pt")
                reader = easyocr.Reader(['en'], gpu=False)

                raw_texts = detect_plate_in_image(chkout_img_rgb, model, reader, conf_thres=0.25)
                if not raw_texts:
                    st.warning("⚠️ Không phát hiện biển số nào để xóa.")
                else:
                    for rt in raw_texts:
                        removed_item3 = remove_plate_by_text(rt)
                        if removed_item3:
                            st.success(f"✅ Đã xóa biển số='{rt}' (ID: {removed_item3['id']}) khỏi DB.")
                            if removed_item3["image_path"] and os.path.exists(removed_item3["image_path"]):
                                st.image(removed_item3["image_path"], caption="🖼️ Ảnh Đã Xoá", use_container_width=True)
                                # os.remove(removed_item3["image_path"])
                        else:
                            st.warning(f"⚠️ Biển số '{rt}' không có trong DB, không thể xóa.")
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")

    st.markdown("---")

    # Xem lại CSV sau khi xóa
    st.markdown("### 📊 Database Sau Khi Check Out")
    df_out = load_csv_to_dataframe()
    if df_out is not None:
        st.dataframe(df_out, use_container_width=True)
    else:
        st.info("ℹ️ CSV trống hoặc chưa có dữ liệu.")

def verify_tab():
    st.subheader("4) Xác Thực ID & Biển Số")
    """
    - Nhập **ID** (đã có trong DB).
    - Upload **Ảnh mới** chứa biển số.
    - Nếu biển số OCR được **trùng** với `raw_text` trong CSV (ứng với ID này) thì báo "Khớp".
    - Nếu không, báo "Không khớp" và hiển thị ảnh gốc đã lưu (nếu có).
    """

    with st.form("verify_form"):
        verify_id = st.text_input("Nhập ID để xác thực:", key="verify_id").strip()
        verify_image = st.file_uploader("Chọn Ảnh Chứa Biển Số Cần Xác Thực:", type=["jpg", "jpeg", "png"])
        submitted_verify = st.form_submit_button("✅ Xác Thực ID & Biển Số")
    if submitted_verify:
        if not verify_id:
            st.warning("⚠️ Vui lòng nhập ID!")
        elif not verify_image:
            st.warning("⚠️ Vui lòng upload ảnh!")
        else:
            # Kiểm tra ID có trong CSV không
            info_in_csv = check_plate_exists_by_id(verify_id)
            if not info_in_csv:
                st.error(f"❌ Không tìm thấy ID={verify_id} trong database!")
            else:
                # Đọc ảnh, OCR
                file_bytes = np.asarray(bytearray(verify_image.read()), dtype=np.uint8)
                vf_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                vf_rgb = cv2.cvtColor(vf_bgr, cv2.COLOR_BGR2RGB)
                st.image(vf_rgb, caption="🖼️ Ảnh Xác Thực", channels="RGB", use_container_width=True)

                with st.spinner("🔍 Đang chạy YOLO + OCR để xác thực..."):
                    try:
                        model = YOLO("best.pt")
                        reader = easyocr.Reader(['en'], gpu=False)

                        raw_texts_found = detect_plate_in_image(vf_rgb, model, reader, conf_thres=0.25)
                        if not raw_texts_found:
                            st.error("❌ Không phát hiện biển số nào trong ảnh xác thực.")
                        else:
                            matched = False
                            for text_ocr in raw_texts_found:
                                if text_ocr.strip() == info_in_csv["raw_text"].strip():
                                    matched = True
                                    break

                            if matched:
                                st.success(
                                    f"✅ Biển số OCR được **trùng khớp** với ID={verify_id} "
                                    f"(raw_text={info_in_csv['raw_text']})."
                                )
                            else:
                                st.error("❌ Biển số OCR không khớp với biển số đã lưu cho ID này!")
                                if info_in_csv["image_path"] and os.path.exists(info_in_csv["image_path"]):
                                    st.warning("ℹ️ Ảnh gốc đã lưu cho ID này:")
                                    st.image(
                                        info_in_csv["image_path"],
                                        caption=f"🖼️ Ảnh Gốc ID={info_in_csv['id']} - raw_text={info_in_csv['raw_text']}",
                                        use_container_width=True
                                    )
                                else:
                                    st.warning("⚠️ Không tìm thấy ảnh gốc trong DB cho ID này.")

                    except Exception as e:
                        st.error(f"❌ Lỗi khi xác thực: {e}")

    st.markdown("---")

    # Hiển thị toàn bộ CSV
    st.markdown("### 📊 CSDL Hiện Tại (Tham Khảo)")
    df_ver = load_csv_to_dataframe()
    if df_ver is not None:
        st.dataframe(df_ver, use_container_width=True)
    else:
        st.info("ℹ️ CSV rỗng hoặc chưa có dữ liệu.")

def main():
    st.set_page_config(
        page_title="🚗 Phát Hiện Biển Số Xe",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
        # Removed 'theme' parameter
    )

    st.title("🚗 Ứng Dụng Phát Hiện Biển Số Xe - Loại Bỏ Ký Tự Đặc Biệt")

    # Sidebar for navigation
    with st.sidebar:
        # Optionally add a logo
        # st.image("path_to_logo.png", width=200)  # Replace with your logo path
        st.header("Menu")
        tab = st.radio("Chọn Tab", [
            "Xử Lý Ảnh & Video",
            "Kiểm Tra Biển Số",
            "Check Out (Xoá Khỏi DB)",
            "Xác Thực ID & Biển Số"
        ])
        st.markdown("---")
        st.header("Cấu Hình Model")
        model_path = st.text_input(
            "Đường dẫn Model YOLO (best.pt):",
            value="best.pt"
        )
        conf_thres = st.slider("Ngưỡng Confidence", 0.0, 1.0, 0.25, 0.05)

    # Main content based on selected tab
    if tab == "Xử Lý Ảnh & Video":
        process_tab(model_path, conf_thres)
    elif tab == "Kiểm Tra Biển Số":
        check_tab()
    elif tab == "Check Out (Xoá Khỏi DB)":
        checkout_tab()
    elif tab == "Xác Thực ID & Biển Số":
        verify_tab()

if __name__ == "__main__":
    main()
