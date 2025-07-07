# Công Cụ Gán Nhãn Vai Trò Ngữ Nghĩa (Semantic Role Labeling Tool)

## Tổng Quan

Công cụ này là một ứng dụng web giúp gán nhãn vai trò ngữ nghĩa cho dữ liệu văn bản (ví dụ: Chủ thể, Đối tượng, Chủ đề, v.v.). Công cụ hỗ trợ người dùng gán nhãn cho các câu hoặc đoạn văn bản, đặc biệt hữu ích cho các tác vụ Xử lý Ngôn ngữ Tự nhiên (NLP) như xây dựng bộ dữ liệu Semantic Role Labeling (SRL).

## Hướng Dẫn Sử Dụng

1. **Tải lên tệp CSV**:
   - Nhấn nút **Upload** và chọn tệp CSV chứa các văn bản cần gán nhãn. Mỗi dòng nên chứa một đoạn văn bản ở cột đầu tiên.
2. **Duyệt văn bản**:
   - Sử dụng các nút **Previous** và **Next** để chuyển qua lại giữa các văn bản đã tải lên. Văn bản hiện tại sẽ hiển thị ở trung tâm.
3. **Gán nhãn**:
   - Với mỗi văn bản, chọn vai trò ngữ nghĩa từ danh sách (ví dụ: ASPECT, OPINION, HOLDER, v.v.).
   - Nhấn **Save Label** để lưu nhãn cho văn bản hiện tại. Nhãn sẽ xuất hiện ở bảng bên phải.
4. **Xuất kết quả**:
   - Sau khi hoàn thành, nhấn **Export to CSV** để tải về tệp CSV chứa các văn bản và nhãn đã gán.

## Kết Quả Xuất Ra

Tệp CSV xuất ra sẽ chứa thông tin chi tiết cho từng đoạn văn bản và các nhãn đã gán. Các cột bao gồm:

- `sentence`: Câu/văn bản gốc.
- Với mỗi nhãn (ví dụ: label_1, label_2, ...):
  - `label_X_r`: Vai trò ngữ nghĩa (ví dụ: ASPECT, OPINION, ...).
  - `label_X_te`: Đoạn văn bản được gán nhãn.
  - `label_X_ir`: Vị trí hoặc chỉ số từ trong câu được gán nhãn.

### Ví dụ Kết Quả

| sentence                                | label_1_r   | label_1_te     | label_1_ir | label_2_r   | label_2_te | label_2_ir | label_3_r | label_3_te    | label_3_ir |
| --------------------------------------- | ----------- | -------------- | ---------- | ----------- | ---------- | ---------- | --------- | ------------- | ---------- |
| Tôi rất thích diễn xuất trong bộ phim này. | ASPECT      | diễn xuất      | 2,3        | OPINION     | thích      | 1          | TARGET    | bộ phim này   | 5,6,7      |
| Nội dung phim khá hấp dẫn.                | ASPECT      | Nội dung phim  | 0,1,2      | OPINION     | hấp dẫn    | 4          |           |               |            |

- Mỗi nhóm `label_X_r`/`label_X_te`/`label_X_ir` tương ứng với một nhãn được gán cho câu.
- Số lượng cột nhãn phụ thuộc vào số nhãn được gán cho mỗi câu.

Kết quả này có thể dùng để huấn luyện hoặc đánh giá các mô hình NLP cho bài toán gán nhãn vai trò ngữ nghĩa.

## Yêu Cầu

- Python 3.x
- Flask (`flask==2.0.1`)
- Flask-CORS (`flask-cors==3.0.10`)
- Werkzeug (`werkzeug==2.0.3`)
- Pandas (`pandas==1.3.5`)

Tất cả các thư viện Python cần thiết đã được liệt kê trong `requirements.txt`.

Không cần cài đặt thêm thư viện JavaScript; các thư viện giao diện (ví dụ: Bootstrap) được tải qua CDN.

## Chạy Ứng Dụng

1. Cài đặt các thư viện:
   ```bash
   pip install -r requirements.txt
   ```
2. Khởi động ứng dụng:
   ```bash
   python app.py
   ```
3. Mở trình duyệt và truy cập `http://localhost:5000` để sử dụng công cụ.

---

## Mô tả Nhãn (Label Descriptions)

| Nhãn         | Mô tả                                                                 |
| ------------ | ---------------------------------------------------------------------- |
| **ASPECT**   | Phần của bộ phim được đề cập (ví dụ: diễn xuất, cốt truyện, nhạc phim) |
| **OPINION**  | Ý kiến, cảm xúc (ví dụ: nhàm chán, hấp dẫn, tệ)                        |
| **HOLDER**   | Người đưa ra ý kiến (thường là người đánh giá, có thể ẩn)              |
| **TARGET**   | Đối tượng được nói đến (thường là bộ phim hoặc một khía cạnh nào đó)   |
| **NEGATION** | Từ phủ định làm thay đổi ý nghĩa (ví dụ: không, chưa từng)             |
| **TIME**     | Thông tin thời gian nếu có (ví dụ: "tối qua", "năm 2020")            |
| **MODALITY** | Mức độ, sắc thái (ví dụ: "có thể tốt hơn", "rất thích")              |
| **EMOTION**  | (Tùy chọn) Từ chỉ cảm xúc rõ ràng (ví dụ: "yêu thích", "ghét")        |

---

Nếu có thắc mắc hoặc cần hỗ trợ, vui lòng liên hệ người phát triển.
