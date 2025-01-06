import os
import csv
import uuid
import tempfile
from datetime import datetime
import re
import cv2
import easyocr
import numpy as np
import streamlit as st
import pandas as pd
from ultralytics import YOLO
import glob
import pickle
import face_recognition


##########################
# (1) HÀM GIẢ LẬP ANTI-SPOOF
##########################
def test(image, model_dir, device_id):
    # Demo: luôn trả về 1 (face real)
    return 1


##########################
# (2) ĐỊNH NGHĨA CÁC ĐƯỜNG DẪN
##########################
DB_DIR = "./db"  # Thư mục lưu embeddings gương mặt
PLATES_CSV = "plates_database.csv"
PLATES_FOLDER = "plates"  # Lưu ảnh cắt biển số
LOG_FACE_PATH = "./log_face.txt"  # Log login/logout gương mặt
USERS_CSV = "users_database.csv"  # Tệp CSV lưu trữ thông tin User

##########################
# TẠO THƯ MỤC (NẾU CHƯA CÓ)
##########################
os.makedirs(PLATES_FOLDER, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)


##########################
# (3) ĐẢM BẢO CSV BIỂN SỐ CÓ CỘT 'owner_id'
##########################
def ensure_owner_column(csv_path=PLATES_CSV):
    if not os.path.exists(csv_path):
        # Tạo CSV mới với cột owner_id
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "class_name", "confidence", "raw_text",
                "province", "bbox", "timestamp", "image_path", "owner_id"
            ])
        return
    df = pd.read_csv(csv_path)
    if "owner_id" not in df.columns:
        df["owner_id"] = ""
        df.to_csv(csv_path, index=False)


##########################
# (4) BẢNG MAP TỈNH
##########################
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


##########################
# (5) HÀM XỬ LÝ CSV BIỂN SỐ
##########################
def load_csv_to_dataframe(csv_path=PLATES_CSV):
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def save_plate_info_to_csv(data_list, csv_path=PLATES_CSV):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "id", "class_name", "confidence", "raw_text",
                "province", "bbox", "timestamp", "image_path", "owner_id"
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
                item["owner_id"]
            ])


def check_plate_exists_in_csv(plate_text, csv_path=PLATES_CSV):
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            if row[3].strip().upper() == plate_text.strip().upper():
                return {
                    "id": row[0],
                    "class_name": row[1],
                    "confidence": row[2],
                    "raw_text": row[3],
                    "province": row[4],
                    "bbox": row[5],
                    "timestamp": row[6],
                    "image_path": row[7],
                    "owner_id": row[8] if len(row) > 8 else ""
                }
    return None


def check_plate_exists_by_id(plate_id, csv_path=PLATES_CSV):
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
                    "owner_id": row[8] if len(row) > 8 else ""
                }
    return None


def remove_plate_by_id(plate_id, csv_path=PLATES_CSV):
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
                    "owner_id": row[8] if len(row) > 8 else ""
                }
            else:
                rows.append(row)

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    return removed_item


def remove_plate_by_text(plate_text, csv_path=PLATES_CSV):
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
            if row[3].strip().upper() == plate_text.strip().upper():
                removed_item = {
                    "id": row[0],
                    "class_name": row[1],
                    "confidence": row[2],
                    "raw_text": row[3],
                    "province": row[4],
                    "bbox": row[5],
                    "timestamp": row[6],
                    "image_path": row[7],
                    "owner_id": row[8] if len(row) > 8 else ""
                }
            else:
                rows.append(row)

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    return removed_item


def assign_owner_to_plate(plate_text, user_id, csv_path=PLATES_CSV):
    if not os.path.exists(csv_path):
        return False, "CSV chưa có dữ liệu."
    df = pd.read_csv(csv_path)
    if "raw_text" not in df.columns:
        return False, "CSV không có cột raw_text."
    found_idx = df.index[df["raw_text"].str.upper() == plate_text.upper()].tolist()
    if not found_idx:
        return False, f"Không tìm thấy biển số {plate_text} trong CSV."
    for idx in found_idx:
        df.at[idx, "owner_id"] = user_id
    df.to_csv(csv_path, index=False)
    return True, f"Đã gán User ID '{user_id}' cho biển số '{plate_text}'."


##########################
# (6) FACE RECOGNITION
##########################
def load_face_embeddings(db_dir=DB_DIR):
    embeddings_dict = {}
    for file_path in glob.glob(os.path.join(db_dir, "*.pickle")):
        filename = os.path.basename(file_path)
        user_id = os.path.splitext(filename)[0]
        with open(file_path, "rb") as f:
            emb = pickle.load(f)
        embeddings_dict[user_id] = emb
    return embeddings_dict


def recognize_face(frame_bgr, db_embeddings, tolerance=0.6):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(frame_rgb)
    if len(encodings) == 0:
        return ('no_persons_found', None)
    face_encoding = encodings[0]
    known_ids = list(db_embeddings.keys())
    known_encods = list(db_embeddings.values())

    if not known_encods:
        # Nếu DB trống, không so sánh và luôn trả về 'unknown_person'
        return ('unknown_person', None)

    distances = face_recognition.face_distance(known_encods, face_encoding)
    if len(distances) == 0:
        return ('unknown_person', None)

    min_dist_index = np.argmin(distances)
    min_dist = distances[min_dist_index]
    if min_dist <= tolerance:
        return (known_ids[min_dist_index], min_dist)
    else:
        return ('unknown_person', min_dist)


def save_new_face_embedding(frame_bgr, user_id, db_dir=DB_DIR):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(frame_rgb)
    if len(encodings) == 0:
        return False
    face_enc = encodings[0]
    file_path = os.path.join(db_dir, f"{user_id}.pickle")
    with open(file_path, "wb") as f:
        pickle.dump(face_enc, f)
    return True


def spoof_check(frame_bgr):
    label = test(
        image=frame_bgr,
        model_dir="path/to/anti_spoof_models",
        device_id=0
    )
    return (label == 1)


##########################
# (7) YOLO + EASY OCR
##########################
def generate_plate_id():
    return str(uuid.uuid4())


def get_province_from_plate(plate_text: str) -> str:
    plate_text = plate_text.replace(" ", "").upper()
    if len(plate_text) < 2:
        return "Unknown"
    prefix = plate_text[:2]
    if not prefix.isdigit():
        return "Unknown"
    return province_map.get(prefix, "Unknown")


def save_plate_image(plate_img_rgb, plate_id):
    image_path = os.path.join(PLATES_FOLDER, f"{plate_id}.jpg")
    plate_img_bgr = cv2.cvtColor(plate_img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, plate_img_bgr)
    return image_path


def detect_and_read_license_plate_easyocr(frame_rgb, model, reader, conf_thres=0.25):
    results = model.predict(source=frame_rgb, conf=conf_thres)
    detection = results[0]
    class_names = model.names

    ocr_data = []
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    for box_data in detection.boxes.data.tolist():
        x1, y1, x2, y2, conf, class_id = box_data
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_name = class_names[int(class_id)]

        plate_img_rgb = frame_rgb[y1:y2, x1:x2]
        ocr_out = reader.readtext(plate_img_rgb, detail=1)
        recognized_texts = [txt for (_, txt, _) in ocr_out]
        raw_text = " ".join(recognized_texts).strip()

        cleaned_text = re.sub(r'[^A-Za-z0-9]+', '', raw_text).upper()
        province = get_province_from_plate(cleaned_text)

        plate_id = generate_plate_id()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        image_path = ""
        if plate_img_rgb.size > 0:
            image_path = save_plate_image(plate_img_rgb, plate_id)

        ocr_data.append({
            "id": plate_id,
            "class_name": cls_name,
            "confidence": float(conf),
            "raw_text": cleaned_text,
            "province": province,
            "bbox": [x1, y1, x2, y2],
            "timestamp": now_str,
            "image_path": image_path,
            "owner_id": ""  # Initialize owner_id as empty
        })

        color = (0, 255, 0)
        thickness = 2
        display_text = f"{cls_name}: {cleaned_text} ({province})"
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame_bgr, display_text, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    annotated_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, ocr_data


def process_video(video_path, model_path, conf_thres=0.25, progress_callback=None):
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


def detect_plate_in_image(image_rgb, model, reader, conf_thres=0.25):
    results = model.predict(source=image_rgb, conf=conf_thres)
    detection = results[0]

    raw_texts = []
    for box_data in detection.boxes.data.tolist():
        x1, y1, x2, y2, conf, class_id = box_data
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        plate_img_rgb = image_rgb[y1:y2, x1:x2]

        ocr_out = reader.readtext(plate_img_rgb, detail=1)
        recognized_texts = [txt for (_, txt, _) in ocr_out]
        raw_text = " ".join(recognized_texts).strip()

        cleaned_text = re.sub(r'[^A-Za-z0-9]+', '', raw_text).upper()
        raw_texts.append(cleaned_text)

    return raw_texts


##########################
# (9) HÀM HỖ TRỢ QUẢN LÝ USER
##########################
def ensure_users_csv(csv_path=USERS_CSV):
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "user_name", "date_created"])


def save_user_info(user_id, user_name, csv_path=USERS_CSV):
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([user_id, user_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


def user_name_exists(user_name, csv_path=USERS_CSV):
    if not os.path.exists(csv_path):
        return False
    df = pd.read_csv(csv_path)
    if df.empty:
        return False
    return user_name.strip().upper() in df['user_name'].str.upper().values


def get_user_info_by_id(user_id, csv_path=USERS_CSV):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    user_row = df[df['user_id'] == user_id].iloc[0] if not df[df['user_id'] == user_id].empty else None
    if user_row is not None:
        return {
            "user_id": user_row['user_id'],
            "user_name": user_row['user_name'],
            "date_created": user_row['date_created']
        }
    return None


def remove_user_from_csv(user_id, csv_path=USERS_CSV):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    df = df[df['user_id'] != user_id]
    df.to_csv(csv_path, index=False)


def get_user_name_by_id(user_id, csv_path=USERS_CSV):
    user_info = get_user_info_by_id(user_id, csv_path)
    return user_info['user_name'] if user_info else "Unknown"


##########################
# (13) MAIN FUNCTION
##########################
def main():
    st.set_page_config(
        page_title="🚗 App Biển Số + Nhận Diện Khuôn Mặt",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    ensure_owner_column(PLATES_CSV)  # Đảm bảo CSV có cột owner_id
    ensure_users_csv(USERS_CSV)  # Đảm bảo CSV Users tồn tại

    st.title("🚗 Ứng Dụng Biển Số + Gán User & Kiểm Tra")

    with st.sidebar:
        st.header("Menu")
        tab = st.radio(
            "Chọn Tab",
            [
                "Xử Lý Ảnh & Video",
                "Kiểm Tra Biển Số",
                "Check Out (Xoá Khỏi DB)",
                "Xác Thực ID & Biển Số",
                "Gán User → Biển Số",
                "Quản Lý User",  # Loại bỏ tab liên quan đến webcam
                "Kiểm tra User và Biển số"  # Thêm Tab Mới
            ]
        )
        st.markdown("---")
        st.header("Cấu Hình Model")
        model_path = st.text_input("Đường dẫn Model YOLO (best.pt):", value="best.pt")
        conf_thres = st.slider("Ngưỡng Confidence", 0.0, 1.0, 0.25, 0.05)

    if tab == "Xử Lý Ảnh & Video":
        process_tab(model_path, conf_thres)
    elif tab == "Kiểm Tra Biển Số":
        check_tab()
    elif tab == "Check Out (Xoá Khỏi DB)":
        checkout_tab()
    elif tab == "Xác Thực ID & Biển Số":
        verify_tab()
    elif tab == "Gán User → Biển Số":
        assign_tab()
    elif tab == "Quản Lý User":
        manage_user_tab()
    elif tab == "Kiểm tra User và Biển số":
        check_user_plate_tab(model_path, conf_thres)


##########################
# (14) CÁC TAB
##########################
# TAB 1: Xử Lý Ảnh & Video
def process_tab(model_path, conf_thres):
    st.subheader("1) Xử Lý Ảnh & Video + Lưu CSV")
    mode = st.radio("Chế độ", ["Ảnh", "Video"], horizontal=True)

    if mode == "Ảnh":
        st.markdown("### 📷 Xử Lý Ảnh")
        uploaded_file = st.file_uploader("Chọn file ảnh (jpg, jpeg, png)...", type=["jpg", "jpeg", "png"])
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
                with st.spinner("Đang xử lý ảnh..."):
                    try:
                        model = YOLO(model_path)
                        reader = easyocr.Reader(['en'], gpu=False)

                        annotated_rgb, ocr_data = detect_and_read_license_plate_easyocr(
                            image_rgb, model, reader, conf_thres
                        )

                        plates_to_save = []
                        for plate_info in ocr_data:
                            raw_text = plate_info["raw_text"]
                            if raw_text:
                                exist = check_plate_exists_in_csv(raw_text)
                                if exist:
                                    st.warning(f"Biển số '{exist['raw_text']}' đã tồn tại (ID: {exist['id']}).")
                                else:
                                    plates_to_save.append(plate_info)

                        if plates_to_save:
                            save_plate_info_to_csv(plates_to_save)
                            st.success(f"Đã lưu {len(plates_to_save)} biển số mới vào CSV.")

                        with col2:
                            st.subheader("🖼️ Ảnh Đã Annotate")
                            st.image(annotated_rgb, channels="RGB", use_container_width=True)

                        st.markdown("### Kết Quả OCR")
                        if ocr_data:
                            for i, item in enumerate(ocr_data, start=1):
                                with st.expander(f"Biển số #{i}"):
                                    st.write(item)
                        else:
                            st.info("Không phát hiện biển số nào.")

                    except Exception as e:
                        st.error(f"Lỗi: {e}")

    else:  # Video
        video_mode = st.radio("Chọn nguồn video", ["Upload Video"], horizontal=True)  # Loại bỏ "Sử dụng Webcam"
        if video_mode == "Upload Video":
            st.markdown("### Xử Lý Video Từ File")
            uploaded_video = st.file_uploader("Chọn file video (mp4, mov, avi, mkv...)",
                                              type=["mp4", "mov", "avi", "mkv"])
            if uploaded_video is not None:
                temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_video_file.write(uploaded_video.read())
                temp_video_file_path = temp_video_file.name
                temp_video_file.close()

                st.video(uploaded_video)

                if st.button("Phát Hiện + OCR Cho Video"):
                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    def update_progress(progress):
                        progress_bar.progress(progress)
                        progress_text.text(f"Tiến độ: {int(progress * 100)}%")

                    with st.spinner("Đang xử lý video..."):
                        try:
                            processed_video_path, all_frames_data = process_video(
                                temp_video_file_path, model_path, conf_thres, progress_callback=update_progress
                            )
                            st.success("Xử lý xong!")

                            flattened_ocr = [plate for frame in all_frames_data for plate in frame]
                            new_plates = []
                            for plate_info in flattened_ocr:
                                raw_text = plate_info["raw_text"]
                                if raw_text:
                                    exist = check_plate_exists_in_csv(raw_text)
                                    if exist:
                                        st.warning(f"Biển số '{exist['raw_text']}' đã tồn tại (ID: {exist['id']}).")
                                    else:
                                        new_plates.append(plate_info)

                            if new_plates:
                                save_plate_info_to_csv(new_plates)
                                st.success(f"Đã lưu {len(new_plates)} biển số mới.")

                            st.markdown("### Video Đã Annotate")
                            st.video(processed_video_path)

                            with st.expander("Chi Tiết OCR Từng Frame"):
                                for frame_idx, frame_data in enumerate(all_frames_data):
                                    with st.expander(f"Frame {frame_idx + 1}"):
                                        if not frame_data:
                                            st.write("Không phát hiện biển số.")
                                        else:
                                            for i, item in enumerate(frame_data, start=1):
                                                st.write(f"Biển số #{i}: {item}")

                            os.remove(processed_video_path)
                        except Exception as e:
                            st.error(f"Lỗi: {e}")

                    progress_bar.empty()
                    progress_text.empty()


##########################
# TAB 2: Kiểm Tra Biển Số
##########################
def check_tab():
    st.subheader("2) Kiểm Tra Biển Số")
    st.write("Tìm kiếm bằng ID hoặc raw_text, hoặc kiểm tra bằng ảnh (không lưu).")

    # Tìm bằng ID
    st.markdown("#### Tìm Biển Số Bằng ID")
    with st.form("search_id_form"):
        search_id = st.text_input("Nhập ID:")
        submitted_id = st.form_submit_button("Kiểm tra bằng ID")
    if submitted_id:
        if search_id:
            info = check_plate_exists_by_id(search_id, PLATES_CSV)
            if info:
                st.success("Tìm thấy ID trong CSDL!")
                st.write(info)
                if info["image_path"] and os.path.exists(info["image_path"]):
                    st.image(info["image_path"], caption="Ảnh Biển Số Đã Lưu", use_container_width=True)
            else:
                st.error("Không tìm thấy ID này.")
        else:
            st.warning("Vui lòng nhập ID.")

    st.markdown("---")

    # Tìm bằng raw_text
    st.markdown("#### Tìm Biển Số Bằng raw_text")
    with st.form("search_text_form"):
        search_text = st.text_input("Nhập biển số:")
        submitted_text = st.form_submit_button("Kiểm tra")
    if submitted_text:
        if search_text:
            info2 = check_plate_exists_in_csv(search_text, PLATES_CSV)
            if info2:
                st.success("Đã tìm thấy biển số trong CSV!")
                st.write(info2)
                if info2["image_path"] and os.path.exists(info2["image_path"]):
                    st.image(info2["image_path"], caption="Ảnh Biển Số Đã Lưu", use_container_width=True)
            else:
                st.error("Không tìm thấy biển số này.")
        else:
            st.warning("Vui lòng nhập biển số.")

    st.markdown("---")

    # Kiểm tra bằng ảnh (không lưu)
    st.markdown("#### Kiểm Tra Bằng Ảnh (Không Lưu)")
    with st.form("check_image_form"):
        check_image = st.file_uploader("Chọn ảnh ...", type=["jpg", "jpeg", "png"])
        submitted_check_image = st.form_submit_button("Phân Tích Ảnh")
    if submitted_check_image and check_image is not None:
        file_bytes = np.asarray(bytearray(check_image.read()), dtype=np.uint8)
        check_img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        check_img_rgb = cv2.cvtColor(check_img_bgr, cv2.COLOR_BGR2RGB)
        st.image(check_img_rgb, caption="Ảnh Kiểm Tra", channels="RGB", use_container_width=True)

        with st.spinner("Đang OCR..."):
            try:
                model = YOLO("best.pt")
                reader = easyocr.Reader(['en'], gpu=False)
                found_plates = detect_plate_in_image(check_img_rgb, model, reader, conf_thres=0.25)
                if not found_plates:
                    st.warning("Không phát hiện biển số.")
                else:
                    st.success("Biển số OCR được:")
                    for plate_str in found_plates:
                        with st.expander(f"Biển số: {plate_str}"):
                            exist_info = check_plate_exists_in_csv(plate_str, PLATES_CSV)
                            if exist_info:
                                st.write(f"ID: {exist_info['id']}")
                                st.success(f"Biển số '{plate_str}' đã có trong CSDL.")
                                if exist_info["image_path"] and os.path.exists(exist_info["image_path"]):
                                    st.image(exist_info["image_path"], caption="Ảnh Biển Số Đã Lưu Trong CSDL")
                            else:
                                st.error(f"Biển số '{plate_str}' chưa có trong CSDL.")
            except Exception as e:
                st.error(f"Lỗi: {e}")

    st.markdown("---")
    # Hiển thị CSV
    st.markdown("### Database Hiện Tại")
    df_check = load_csv_to_dataframe(PLATES_CSV)
    if df_check is not None:
        st.dataframe(df_check, use_container_width=True)
    else:
        st.info("Chưa có dữ liệu CSV.")


# TAB 3: Check Out (Xóa Khỏi DB)
def checkout_tab():
    st.subheader("3) Check Out (Xoá Khỏi DB)")

    # Xoá bằng ID
    st.markdown("#### Xóa Bằng ID")
    with st.form("remove_id_form"):
        remove_id = st.text_input("Nhập ID cần xóa:")
        submitted_remove_id = st.form_submit_button("Xóa theo ID")
    if submitted_remove_id:
        if remove_id:
            removed_item = remove_plate_by_id(remove_id, PLATES_CSV)
            if removed_item:
                st.success(f"Đã xóa ID={removed_item['id']} - Biển số={removed_item['raw_text']}.")
                if removed_item["image_path"] and os.path.exists(removed_item["image_path"]):
                    st.image(removed_item["image_path"], caption="Ảnh Biển Số Đã Xoá")
            else:
                st.error("Không tìm thấy ID trong DB.")
        else:
            st.warning("Vui lòng nhập ID.")

    st.markdown("---")

    # Xoá bằng raw_text
    st.markdown("#### Xóa Bằng Biển Số")
    with st.form("remove_text_form"):
        remove_text = st.text_input("Nhập biển số cần xóa:")
        submitted_remove_text = st.form_submit_button("Xóa theo Biển Số")
    if submitted_remove_text:
        if remove_text:
            removed_item2 = remove_plate_by_text(remove_text, PLATES_CSV)
            if removed_item2:
                st.success(f"Đã xóa Biển số={removed_item2['raw_text']} - ID={removed_item2['id']}.")
                if removed_item2["image_path"] and os.path.exists(removed_item2["image_path"]):
                    st.image(removed_item2["image_path"], caption="Ảnh Đã Xoá")
            else:
                st.error("Không tìm thấy biển số này.")
        else:
            st.warning("Vui lòng nhập biển số.")

    st.markdown("---")

    # Xoá bằng Ảnh
    st.markdown("#### Xóa Bằng Ảnh")
    with st.form("checkout_image_form"):
        checkout_image = st.file_uploader("Chọn ảnh chứa biển số...", type=["jpg", "jpeg", "png"])
        submitted_checkout_image = st.form_submit_button("Xoá Khỏi DB Bằng Ảnh")
    if submitted_checkout_image and checkout_image is not None:
        file_bytes = np.asarray(bytearray(checkout_image.read()), dtype=np.uint8)
        chkout_img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        chkout_img_rgb = cv2.cvtColor(chkout_img_bgr, cv2.COLOR_BGR2RGB)
        st.image(chkout_img_rgb, caption="Ảnh Check Out", channels="RGB")

        with st.spinner("Đang OCR..."):
            try:
                model = YOLO("best.pt")
                reader = easyocr.Reader(['en'], gpu=False)
                raw_texts = detect_plate_in_image(chkout_img_rgb, model, reader, conf_thres=0.25)
                if not raw_texts:
                    st.warning("Không phát hiện biển số để xóa.")
                else:
                    for rt in raw_texts:
                        removed_item3 = remove_plate_by_text(rt, PLATES_CSV)
                        if removed_item3:
                            st.success(f"Đã xóa biển số='{rt}' (ID: {removed_item3['id']}).")
                            if removed_item3["image_path"] and os.path.exists(removed_item3["image_path"]):
                                st.image(removed_item3["image_path"], caption="Ảnh Đã Xoá")
                        else:
                            st.warning(f"Biển số '{rt}' không có trong DB.")
            except Exception as e:
                st.error(f"Lỗi: {e}")

    st.markdown("---")
    # CSV sau khi xóa
    st.markdown("### Database Sau Khi Check Out")
    df_out = load_csv_to_dataframe(PLATES_CSV)
    if df_out is not None:
        st.dataframe(df_out, use_container_width=True)
    else:
        st.info("CSV trống hoặc chưa có dữ liệu.")


# TAB 4: Xác Thực ID & Biển Số
def verify_tab():
    st.subheader("4) Xác Thực ID & Biển Số")

    with st.form("verify_form"):
        verify_id = st.text_input("Nhập ID:", key="verify_id").strip()
        verify_image = st.file_uploader("Chọn Ảnh Chứa Biển Số:", type=["jpg", "jpeg", "png"])
        submitted_verify = st.form_submit_button("Xác Thực")
    if submitted_verify:
        if not verify_id:
            st.warning("Vui lòng nhập ID.")
        elif not verify_image:
            st.warning("Vui lòng upload ảnh.")
        else:
            info_in_csv = check_plate_exists_by_id(verify_id, PLATES_CSV)
            if not info_in_csv:
                st.error(f"Không tìm thấy ID={verify_id} trong DB!")
            else:
                file_bytes = np.asarray(bytearray(verify_image.read()), dtype=np.uint8)
                vf_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                vf_rgb = cv2.cvtColor(vf_bgr, cv2.COLOR_BGR2RGB)
                st.image(vf_rgb, caption="Ảnh Xác Thực", channels="RGB")

                with st.spinner("Đang OCR..."):
                    try:
                        model = YOLO("best.pt")
                        reader = easyocr.Reader(['en'], gpu=False)
                        raw_texts_found = detect_plate_in_image(vf_rgb, model, reader, conf_thres=0.25)
                        if not raw_texts_found:
                            st.error("Không phát hiện biển số trong ảnh.")
                        else:
                            matched = any(
                                text_ocr.strip().upper() == info_in_csv["raw_text"].strip().upper()
                                for text_ocr in raw_texts_found
                            )
                            if matched:
                                st.success(
                                    f"Biển số OCR trùng khớp với ID={verify_id} (raw_text={info_in_csv['raw_text']})."
                                )
                            else:
                                st.error("Biển số OCR không khớp với biển số đã lưu cho ID này!")
                                if info_in_csv["image_path"] and os.path.exists(info_in_csv["image_path"]):
                                    st.warning("Ảnh gốc của ID này:")
                                    st.image(info_in_csv["image_path"], caption="Ảnh Gốc")
                    except Exception as e:
                        st.error(f"Lỗi khi xác thực: {e}")

    st.markdown("---")
    st.markdown("### CSDL Hiện Tại")
    df_ver = load_csv_to_dataframe(PLATES_CSV)
    if df_ver is not None:
        st.dataframe(df_ver, use_container_width=True)
    else:
        st.info("CSV rỗng hoặc chưa có dữ liệu.")


# TAB 5: Gán User → Biển Số
def assign_tab():
    st.subheader("5) Gán User → Biển Số")

    st.write("Liên kết một User (gương mặt) với một biển số (raw_text) hoặc bằng ảnh. Thêm `owner_id` cho CSV.")

    assign_mode = st.radio("Chọn phương thức gán:", ["Gán bằng Biển Số (raw_text)", "Gán bằng Ảnh"])

    if assign_mode == "Gán bằng Biển Số (raw_text)":
        st.markdown("#### Gán bằng Biển Số (raw_text)")
        plate_to_assign = st.text_input("Nhập biển số (raw_text) cần gán:", "")

        # Lấy danh sách User hiện có
        def get_all_users(db_dir=DB_DIR, users_csv=USERS_CSV):
            users = []
            if os.path.exists(users_csv):
                df = pd.read_csv(users_csv)
                for _, row in df.iterrows():
                    users.append((row['user_id'], row['user_name']))
            return users

        users = get_all_users()
        user_options = {f"{name} (ID: {uid})": uid for uid, name in users}

        user_selected = st.selectbox("Chọn User để gán:", list(user_options.keys()))
        user_id = user_options.get(user_selected, "")

        if st.button("Gán User Cho Biển Số"):
            if not plate_to_assign.strip():
                st.warning("Vui lòng nhập biển số.")
            elif not user_id:
                st.warning("Vui lòng chọn User.")
            else:
                ok, msg = assign_owner_to_plate(plate_to_assign.strip(), user_id, PLATES_CSV)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    elif assign_mode == "Gán bằng Ảnh":
        st.markdown("#### Gán bằng Ảnh của User và Biển Số")
        with st.form("assign_image_form"):
            user_image = st.file_uploader("Chọn ảnh gương mặt User:", type=["jpg", "jpeg", "png"])
            plate_image = st.file_uploader("Chọn ảnh biển số xe:", type=["jpg", "jpeg", "png"])
            submitted_assign_image = st.form_submit_button("Gán User Cho Biển Số")
        if submitted_assign_image:
            if not user_image or not plate_image:
                st.warning("Vui lòng tải lên cả hai ảnh.")
            else:
                # Xử lý ảnh User
                user_file_bytes = np.asarray(bytearray(user_image.read()), dtype=np.uint8)
                user_img_bgr = cv2.imdecode(user_file_bytes, cv2.IMREAD_COLOR)
                user_img_rgb = cv2.cvtColor(user_img_bgr, cv2.COLOR_BGR2RGB)
                st.image(user_img_rgb, caption="Ảnh Gương Mặt User", channels="RGB", use_container_width=True)

                # Kiểm tra giả mạo
                if not spoof_check(user_img_bgr):
                    st.error("Phát hiện giả mạo! Không thể gán User.")
                else:
                    # Nhận diện User
                    db_embeddings = load_face_embeddings(DB_DIR)
                    user_id_recognized, dist = recognize_face(user_img_bgr, db_embeddings, tolerance=0.6)
                    if user_id_recognized in ["unknown_person", "no_persons_found"]:
                        st.error("Không nhận diện được User trong DB. Vui lòng đăng ký User trước.")
                    else:
                        user_info = get_user_info_by_id(user_id_recognized)
                        if not user_info:
                            st.error("Không tìm thấy thông tin User.")
                        else:
                            st.success(f"Đã nhận diện User: {user_info['user_name']} (ID: {user_info['user_id']})")

                            # Xử lý ảnh Biển Số
                            plate_file_bytes = np.asarray(bytearray(plate_image.read()), dtype=np.uint8)
                            plate_img_bgr = cv2.imdecode(plate_file_bytes, cv2.IMREAD_COLOR)
                            plate_img_rgb = cv2.cvtColor(plate_img_bgr, cv2.COLOR_BGR2RGB)
                            st.image(plate_img_rgb, caption="Ảnh Biển Số Xe", use_container_width=True)

                            # OCR Biển Số
                            model = YOLO("best.pt")
                            reader = easyocr.Reader(['en'], gpu=False)
                            raw_texts_found = detect_plate_in_image(plate_img_rgb, model, reader, conf_thres=0.25)

                            if not raw_texts_found:
                                st.error("Không phát hiện biển số trong ảnh biển số.")
                            else:
                                plate_text = raw_texts_found[0]
                                st.info(f"Biển số OCR được: {plate_text}")

                                # Kiểm tra biển số đã tồn tại
                                plate_info = check_plate_exists_in_csv(plate_text, PLATES_CSV)
                                if plate_info:
                                    if plate_info["owner_id"]:
                                        st.warning(
                                            f"Biển số '{plate_text}' đã được gán với User ID '{plate_info['owner_id']}'.")
                                    else:
                                        # Gán User cho Biển Số
                                        ok, msg = assign_owner_to_plate(plate_text, user_id_recognized, PLATES_CSV)
                                        if ok:
                                            st.success(msg)
                                        else:
                                            st.error(msg)
                                else:
                                    st.error(f"Biển số '{plate_text}' chưa có trong DB. Vui lòng xử lý thêm.")
                                    # Optionally, bạn có thể thêm biển số mới vào DB tại đây
                                    # Ví dụ:
                                    # plate_id = generate_plate_id()
                                    # province = get_province_from_plate(plate_text)
                                    # image_path = save_plate_image(plate_img_rgb, plate_id)
                                    # new_plate = {
                                    #     "id": plate_id,
                                    #     "class_name": "license_plate",
                                    #     "confidence": 1.0,  # Giả sử
                                    #     "raw_text": plate_text,
                                    #     "province": province,
                                    #     "bbox": [0, 0, 0, 0],  # Không có bounding box
                                    #     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    #     "image_path": image_path,
                                    #     "owner_id": user_id_recognized
                                    # }
                                    # save_plate_info_to_csv([new_plate])
                                    # st.success(f"Đã thêm biển số '{plate_text}' và gán với User '{user_info['user_name']}'.")

    st.markdown("---")
    st.markdown("### Kiểm Tra CSDL `owner_id`")
    df_a = load_csv_to_dataframe(PLATES_CSV)
    if df_a is not None:
        # Thêm cột tên User dựa trên owner_id
        df_a['owner_name'] = df_a['owner_id'].apply(lambda x: get_user_name_by_id(x) if x else "")
        st.dataframe(df_a, use_container_width=True)
    else:
        st.info("Chưa có dữ liệu CSV.")


# TAB 6: Quản Lý User (Bao Gồm Đăng Ký User)
def manage_user_tab():
    st.subheader("6) Quản Lý User")

    st.markdown("### 📌 Đăng Ký User Mới")
    with st.form("register_user_form"):
        user_name_input = st.text_input("Nhập tên User:", "")
        uploaded_user_image = st.file_uploader("Chọn ảnh gương mặt User (.jpg/.png):", type=["jpg", "png", "jpeg"])
        submitted_register = st.form_submit_button("Đăng Ký")
    if submitted_register:
        if not user_name_input.strip():
            st.warning("Vui lòng nhập tên User.")
        elif not uploaded_user_image:
            st.warning("Vui lòng tải lên ảnh gương mặt.")
        else:
            # Kiểm tra xem tên User đã tồn tại chưa
            if user_name_exists(user_name_input.strip()):
                st.error(f"User '{user_name_input}' đã tồn tại. Vui lòng chọn tên khác.")
            else:
                # Xử lý ảnh User
                file_bytes = np.asarray(bytearray(uploaded_user_image.read()), dtype=np.uint8)
                user_img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                user_img_rgb = cv2.cvtColor(user_img_bgr, cv2.COLOR_BGR2RGB)
                st.image(user_img_rgb, caption="Ảnh Gương Mặt Đã Tải Lên", channels="RGB", use_container_width=True)

                # Kiểm tra giả mạo
                if not spoof_check(user_img_bgr):
                    st.error("Phát hiện giả mạo! Không thể đăng ký User.")
                else:
                    # Nhận diện xem User đã tồn tại trong DB chưa
                    db_embeddings = load_face_embeddings(DB_DIR)
                    user_id_recognized, dist = recognize_face(user_img_bgr, db_embeddings, tolerance=0.6)
                    if user_id_recognized not in ["unknown_person", "no_persons_found"]:
                        st.warning("Ảnh tải lên đã được đăng ký với User khác.")
                        user_info = get_user_info_by_id(user_id_recognized)
                        if user_info:
                            st.info(f"User đã tồn tại: {user_info['user_name']} (ID: {user_info['user_id']})")
                    else:
                        # Tạo user_id duy nhất
                        user_id = str(uuid.uuid4())

                        # Lưu embeddings mới
                        success = save_new_face_embedding(user_img_bgr, user_id, DB_DIR)
                        if success:
                            # Lưu thông tin User vào CSV
                            save_user_info(user_id, user_name_input.strip(), USERS_CSV)
                            st.success(f"Đăng ký User '{user_name_input}' thành công!")
                            with open(LOG_FACE_PATH, "a", encoding="utf-8") as f:
                                f.write(f"{user_id},{user_name_input.strip()},{datetime.now()},registered\n")
                        else:
                            st.error("Không phát hiện khuôn mặt trong ảnh. Vui lòng thử lại.")

    st.markdown("---")
    st.markdown("### 📝 Danh Sách User Hiện Có")

    # Lấy danh sách User hiện có
    def get_all_users(db_dir=DB_DIR, users_csv=USERS_CSV):
        users = []
        if os.path.exists(users_csv):
            df = pd.read_csv(users_csv)
            for _, row in df.iterrows():
                users.append((row['user_id'], row['user_name']))
        return users

    users = get_all_users()
    if users:
        st.write("#### Danh sách User:")
        user_options = {f"{name} (ID: {uid})": uid for uid, name in users}
        selected_user = st.selectbox("Chọn User để xem chi tiết hoặc xóa:", list(user_options.keys()))

        user_id = user_options.get(selected_user, "")
        user_info = get_user_info_by_id(user_id)

        if user_info:
            st.markdown("##### Thông Tin User")
            st.write(f"**ID:** {user_info['user_id']}")
            st.write(f"**Tên User:** {user_info['user_name']}")
            st.write(f"**Ngày Tạo:** {user_info['date_created']}")

            # Hiển thị ảnh User từ embeddings
            embedding_file = os.path.join(DB_DIR, f"{user_info['user_id']}.pickle")
            if os.path.exists(embedding_file):
                # Không thể dễ dàng chuyển embeddings thành ảnh, nhưng có thể hiển thị thông tin
                st.write("**Ảnh User:** Đã được lưu trữ trong DB.")
            else:
                st.write("**Ảnh User:** Không tìm thấy trong DB.")

            st.markdown("---")
            st.markdown("##### Xóa User")

            # Nút "Xóa User" với key duy nhất
            delete_button = st.button("Xóa User", key=f"delete_user_{user_id}")
            if delete_button:
                st.session_state["user_to_delete"] = user_id

            # Hiển thị thông báo xác nhận nếu cần
            if "user_to_delete" in st.session_state and st.session_state["user_to_delete"] == user_id:
                st.warning(f"Bạn có chắc chắn muốn xóa User '{user_info['user_name']}' không?")
                confirm_delete = st.button("Xác nhận Xóa", key=f"confirm_delete_{user_id}")
                cancel_delete = st.button("Hủy", key=f"cancel_delete_{user_id}")
                if confirm_delete:
                    # Thực hiện xóa
                    # Xóa file embeddings
                    embedding_file = os.path.join(DB_DIR, f"{user_id}.pickle")
                    if os.path.exists(embedding_file):
                        os.remove(embedding_file)
                        st.success(f"Đã xóa embeddings của User '{user_info['user_name']}'.")
                    else:
                        st.error("Không tìm thấy file embeddings của User.")

                    # Cập nhật CSV: loại bỏ hoặc cập nhật owner_id cho các biển số liên quan
                    df = load_csv_to_dataframe(PLATES_CSV)
                    if df is not None:
                        # Tìm các biển số liên quan đến User và loại bỏ owner_id
                        affected_rows = df['owner_id'].str.strip().str.upper() == user_id.strip().upper()
                        if affected_rows.any():
                            df.loc[affected_rows, 'owner_id'] = ""
                            df.to_csv(PLATES_CSV, index=False)
                            st.success(
                                f"Cập nhật 'owner_id' cho các biển số liên quan đến User '{user_info['user_name']}'.")
                        else:
                            st.info(f"Không tìm thấy biển số nào liên kết với User '{user_info['user_name']}'.")
                    else:
                        st.error("Không thể tải dữ liệu từ CSV.")

                    # Xóa các log liên quan đến User
                    if os.path.exists(LOG_FACE_PATH):
                        with open(LOG_FACE_PATH, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                        with open(LOG_FACE_PATH, "w", encoding="utf-8") as f:
                            for line in lines:
                                if not line.startswith(f"{user_id},"):
                                    f.write(line)
                        st.success(f"Đã xóa các log liên quan đến User '{user_info['user_name']}'.")
                    else:
                        st.info("Không tìm thấy file log để cập nhật.")

                    # Xóa thông tin User khỏi users CSV
                    remove_user_from_csv(user_id, USERS_CSV)
                    st.success(f"Đã xóa thông tin User '{user_info['user_name']}' khỏi hệ thống.")

                    # Reset trạng thái xác nhận
                    del st.session_state["user_to_delete"]

                if cancel_delete:
                    st.info("Đã hủy xóa User.")
                    del st.session_state["user_to_delete"]
    else:
        st.info("Không có User nào để quản lý.")

    st.markdown("---")
    st.markdown("### 📋 CSDL User Hiện Tại")
    if os.path.exists(USERS_CSV):
        df_users = pd.read_csv(USERS_CSV)
        st.dataframe(df_users, use_container_width=True)
    else:
        st.info("Chưa có dữ liệu User.")


# TAB 7: Kiểm tra User và Biển số (Chức Năng Mới)
def check_user_plate_tab(model_path, conf_thres):
    st.subheader("7) Kiểm tra User và Biển số")

    st.write("Upload ảnh gương mặt của User và ảnh biển số xe để kiểm tra khớp nhau.")

    with st.form("check_user_plate_form"):
        user_image = st.file_uploader("Chọn ảnh gương mặt User:", type=["jpg", "jpeg", "png"])
        plate_image = st.file_uploader("Chọn ảnh biển số xe:", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Kiểm Tra")
    if submitted:
        if not user_image or not plate_image:
            st.warning("Vui lòng tải lên cả hai ảnh.")
        else:
            # Process Plate Image
            file_bytes_plate = np.asarray(bytearray(plate_image.read()), dtype=np.uint8)
            plate_img_bgr = cv2.imdecode(file_bytes_plate, cv2.IMREAD_COLOR)
            plate_img_rgb = cv2.cvtColor(plate_img_bgr, cv2.COLOR_BGR2RGB)
            st.image(plate_img_rgb, caption="Ảnh Biển Số", use_container_width=True)

            # Extract plate text
            with st.spinner("Đang OCR biển số..."):
                try:
                    model = YOLO(model_path)
                    reader = easyocr.Reader(['en'], gpu=False)
                    raw_texts_found = detect_plate_in_image(plate_img_rgb, model, reader, conf_thres=conf_thres)
                    if not raw_texts_found:
                        st.error("Không phát hiện biển số trong ảnh biển số.")
                    else:
                        # Assuming one plate per image
                        plate_text = raw_texts_found[0]
                        st.info(f"Biển số OCR được: {plate_text}")

                        # Check if plate exists
                        plate_info = check_plate_exists_in_csv(plate_text, PLATES_CSV)
                        if not plate_info:
                            st.error("Biển số chưa có trong DB.")
                        else:
                            owner_id = plate_info["owner_id"]
                            if not owner_id:
                                st.warning("Biển số chưa được gán `owner_id`.")
                            else:
                                # Now, process User Image
                                file_bytes_user = np.asarray(bytearray(user_image.read()), dtype=np.uint8)
                                user_img_bgr = cv2.imdecode(file_bytes_user, cv2.IMREAD_COLOR)
                                user_img_rgb = cv2.cvtColor(user_img_bgr, cv2.COLOR_BGR2RGB)
                                st.image(user_img_rgb, caption="Ảnh Gương Mặt User", use_container_width=True)

                                # Check anti-spoof
                                if not spoof_check(user_img_bgr):
                                    st.error("Phát hiện giả mạo gương mặt!")
                                else:
                                    # Recognize user_id from face
                                    db_embeddings = load_face_embeddings(DB_DIR)
                                    user_id_recognized, dist = recognize_face(user_img_bgr, db_embeddings,
                                                                              tolerance=0.6)
                                    if user_id_recognized in ["unknown_person", "no_persons_found"]:
                                        st.error("Không nhận diện được User trong DB.")
                                    else:
                                        # Compare user_id_recognized with owner_id
                                        if user_id_recognized.strip().upper() == owner_id.strip().upper():
                                            # Get user name
                                            user_info = get_user_info_by_id(user_id_recognized)
                                            user_name = user_info['user_name'] if user_info else "User"
                                            st.success(f"Biển số '{plate_text}' thuộc về User '{user_name}'.")
                                        else:
                                            # Get owner user info
                                            owner_info = get_user_info_by_id(owner_id)
                                            owner_name = owner_info['user_name'] if owner_info else "Unknown"
                                            user_info = get_user_info_by_id(user_id_recognized)
                                            user_name = user_info['user_name'] if user_info else "User"
                                            st.error(
                                                f"Biển số '{plate_text}' thuộc về User '{owner_name}', không phải '{user_name}'.")

                except Exception as e:
                    st.error(f"Lỗi khi OCR biển số: {e}")

    st.markdown("---")
    st.markdown("### Database Hiện Tại")
    df_check = load_csv_to_dataframe(PLATES_CSV)
    if df_check is not None:
        st.dataframe(df_check, use_container_width=True)
    else:
        st.info("Chưa có dữ liệu CSV.")


##########################
# (14) MAIN FUNCTION
##########################
if __name__ == "__main__":
    main()
