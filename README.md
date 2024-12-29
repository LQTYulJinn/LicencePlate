# 🚗 Ứng Dụng Phát Hiện Biển Số Xe

Ứng dụng này sử dụng mô hình YOLO và EasyOCR để phát hiện và nhận diện biển số xe từ ảnh và video. Với giao diện Streamlit thân thiện, người dùng có thể dễ dàng xử lý ảnh, video, kiểm tra, quản lý và xác thực thông tin biển số xe.

## 🛠️ Tính Năng

1. **Xử Lý Ảnh & Video**
   - **Ảnh:** Upload ảnh để phát hiện và nhận diện biển số xe. Kết quả sẽ được hiển thị kèm ảnh đã annotate và lưu thông tin vào CSV.
   - **Video:** Upload video hoặc sử dụng webcam để phát hiện và nhận diện biển số xe trong từng khung hình. Kết quả sẽ được hiển thị kèm video đã annotate và lưu thông tin vào CSV.

2. **Kiểm Tra Biển Số**
   - Tìm kiếm thông tin biển số bằng **ID**, **biển số (raw_text)** hoặc **ảnh**.
   - Hiển thị thông tin chi tiết và ảnh biển số đã lưu.

3. **Check Out (Xoá Khỏi DB)**
   - Xóa thông tin biển số khỏi cơ sở dữ liệu (CSV) bằng **ID**, **biển số (raw_text)** hoặc **ảnh**.
   - Hiển thị thông tin đã xóa và ảnh biển số tương ứng.

4. **Xác Thực ID & Biển Số**
   - Nhập **ID** và upload **ảnh mới** chứa biển số để xác thực.
   - So sánh biển số OCR từ ảnh mới với thông tin đã lưu để xác nhận khớp hay không.

## 📦 Cài Đặt

### Yêu Cầu

- Python 3.7 trở lên
- Các thư viện Python sau:
  - `os`
  - `csv`
  - `uuid`
  - `tempfile`
  - `datetime`
  - `re`
  - `cv2` (OpenCV)
  - `easyocr`
  - `numpy`
  - `streamlit`
  - `pandas`
  - `ultralytics`
  - `streamlit_webrtc`
  - `av`

### Cài Đặt Thư Viện

Bạn có thể cài đặt các thư viện cần thiết bằng cách sử dụng `pip`. Tạo một file `requirements.txt` với nội dung sau:

```txt
opencv-python
easyocr
numpy
streamlit
pandas
ultralytics
streamlit-webrtc
av
