# 🚀 AI ETL DATA

Công cụ xử lý dữ liệu text đa năng với Google Gemini AI, được thiết kế để xử lý đa dạng các loại dữ liệu text theo nhu cầu của người dùng.

## 🎯 Mục tiêu

Tạo ra một công cụ ETL (Extract, Transform, Load) linh hoạt sử dụng AI Gemini để:
- Xử lý dữ liệu text từ file Excel/CSV
- Thực hiện đa dạng các tác vụ AI: tóm tắt, phân loại, phân tích cảm xúc, trích xuất thông tin...
- Hỗ trợ checkpoint để xử lý file lớn
- Giao diện người dùng thân thiện

## ✨ Tính năng chính

### 🤖 Kết nối AI linh hoạt
- ✅ Hỗ trợ multiple Gemini models (gemma-3-27b-it, gemini-2.0-flash-lite, etc.)
- ✅ User tự nhập API Key và chọn model
- ✅ Xử lý lỗi và retry thông minh
- ✅ Rate limiting tự động

### 📁 Xử lý file đa dạng
- ✅ Hỗ trợ Excel (.xlsx, .xls) và CSV
- ✅ Tự động phát hiện encoding
- ✅ Tự động phát hiện cột text
- ✅ Preview dữ liệu trước khi xử lý

### 💾 Checkpoint thông minh
- ✅ Lưu tiến trình tự động
- ✅ Tiếp tục từ điểm dừng khi bị gián đoạn
- ✅ Tự động xóa checkpoint khi hoàn thành

### ✍️ Prompt templates
- ✅ Templates có sẵn: tóm tắt, phân loại, cảm xúc, từ khóa, dịch thuật
- ✅ Hỗ trợ prompt tùy chỉnh
- ✅ Validation prompt input

### 📊 Monitoring & Reporting
- ✅ Progress bar real-time
- ✅ Báo cáo tiến trình định kỳ
- ✅ Ước tính thời gian hoàn thành
- ✅ Thống kê lỗi và thành công

## 🚀 Cài đặt và sử dụng

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy chương trình

```bash
python main.py
```

### 3. Quy trình sử dụng

#### Bước 1: Cấu hình API
```
📡 BƯỚC 1: Cấu hình API Gemini
Nhập Gemini API Key: [YOUR_API_KEY]
```

#### Bước 2: Chọn model
```
🤖 BƯỚC 2: Chọn model AI
Các model có sẵn:
  1. gemma-3-27b-it
  2. gemma-3n-e4b-it  
  3. gemini-2.0-flash-lite
  4. gemini-2.5-flash
Chọn model (1-4) hoặc nhập tên model khác: 1
```

#### Bước 3: Chọn file dữ liệu
```
📁 BƯỚC 3: Chọn file dữ liệu
Nhập đường dẫn file cần xử lý (.xlsx, .csv): data.xlsx
```

#### Bước 4: Chọn cột dữ liệu
```
📊 BƯỚC 4: Chọn cột dữ liệu
File có 1000 dòng dữ liệu
Các cột có sẵn:
  1. ID (VD: 1)
  2. MESSAGE (VD: Khách hàng hỏi về sản phẩm...)
  3. DATE (VD: 2025-01-20)

💡 Tự động phát hiện cột text: 'MESSAGE'
Chọn cột cần xử lý: 2
```

#### Bước 5: Cấu hình checkpoint
```
💾 BƯỚC 5: Cấu hình checkpoint
Sử dụng checkpoint để có thể tiếp tục khi bị dừng? (y/n) [y]: y
```

#### Bước 6: Cấu hình prompt
```
✍️ BƯỚC 6: Cấu hình prompt AI
Các template prompt có sẵn:
  1. summarize: Hãy tóm tắt nội dung sau...
  2. classify: Hãy phân loại nội dung sau...
  3. sentiment: Hãy phân tích cảm xúc...
  4. extract_keywords: Hãy trích xuất từ khóa...
  5. translate: Hãy dịch nội dung sau...
  6. custom: Hãy xử lý nội dung theo yêu cầu
  7. Tự nhập prompt

Chọn template (1-7): 1
```

#### Bước 7: Xác nhận và chạy
```
📋 BƯỚC 7: Tổng kết cấu hình
🤖 Model: gemma-3-27b-it
📁 File input: data.xlsx
📊 Cột xử lý: MESSAGE
💾 Checkpoint: Có
✍️ Prompt: Hãy tóm tắt nội dung sau...

Xác nhận bắt đầu xử lý? (y/n): y
```

## 📊 Output

### File kết quả
- **Tên**: `<tên_file_gốc>_ai_result_<timestamp>.<định_dạng>`
- **Vị trí**: Cùng thư mục với file input
- **Nội dung**: File gốc + cột `AI_RESULT` chứa kết quả AI

### File checkpoint (nếu sử dụng)
- **Tên**: `<tên_file_gốc>_checkpoint.<định_dạng>`
- **Mục đích**: Lưu tiến trình để tiếp tục khi bị dừng
- **Tự động xóa**: Khi hoàn thành xử lý

### Log file
- **Tên**: `ai_etl_data.log`
- **Nội dung**: Chi tiết quá trình xử lý, lỗi, debug info

## 🔧 Cấu hình nâng cao

### Chỉnh sửa config.py

```python
# Model parameters
MAX_OUTPUT_TOKENS = 1024  # Số token tối đa cho output
TEMPERATURE = 0.3         # Độ sáng tạo (0-1)
TOP_P = 0.8              # Nucleus sampling
TOP_K = 40               # Top-k sampling

# Processing
CHECKPOINT_INTERVAL = 50          # Lưu checkpoint mỗi N records
PROGRESS_REPORT_INTERVAL = 100    # Báo cáo mỗi N records
REQUEST_DELAY = 2.0              # Delay giữa requests (giây)

# Retry
MAX_RETRIES = 3          # Số lần thử lại
RETRY_DELAY = 60         # Thời gian chờ retry (giây)
```

### Custom prompt templates

```python
DEFAULT_PROMPT_TEMPLATES = {
    "my_task": "Hãy thực hiện tác vụ custom của tôi:",
    # ... thêm templates khác
}
```

## 📋 Ví dụ sử dụng

### 1. Tóm tắt bình luận khách hàng
```
Input: "Sản phẩm này rất tốt, tôi đã sử dụng 3 tháng và cảm thấy hài lòng..."
Prompt: "Hãy tóm tắt ý chính của bình luận khách hàng"
Output: "Khách hàng hài lòng với sản phẩm sau 3 tháng sử dụng"
```

### 2. Phân loại email
```
Input: "Tôi muốn khiếu nại về dịch vụ giao hàng chậm..."
Prompt: "Phân loại email này vào: Khiếu nại, Hỏi đáp, Đề xuất"
Output: "Khiếu nại"
```

### 3. Phân tích cảm xúc
```
Input: "Dịch vụ khách hàng rất tệ, tôi rất thất vọng"
Prompt: "Phân tích cảm xúc: Tích cực/Tiêu cực/Trung tính"
Output: "Tiêu cực"
```

## 🛠️ Troubleshooting

### Lỗi thường gặp

**1. "Lỗi khởi tạo Gemini"**
```
- Kiểm tra API Key có đúng không
- Kiểm tra model name có được hỗ trợ
- Kiểm tra kết nối internet
```

**2. "File không tồn tại"**
```
- Kiểm tra đường dẫn file có đúng
- Kiểm tra file có quyền đọc
- Sử dụng đường dẫn tuyệt đối
```

**3. "Rate limit exceeded"**
```
- Chương trình sẽ tự động retry
- Tăng REQUEST_DELAY trong config.py
- Kiểm tra quota API
```

**4. "Không tìm thấy cột"**
```
- Kiểm tra tên cột trong file
- Sử dụng tính năng auto-detect
- Kiểm tra encoding file CSV
```

## 🔒 Bảo mật

- ❌ Không lưu API Key trong code
- ✅ API Key chỉ tồn tại trong runtime
- ✅ Log không chứa thông tin nhạy cảm
- ✅ Hỗ trợ .env file cho development

## 📈 Performance

### Tốc độ xử lý
- **Gemini Flash**: ~1-2 records/giây
- **Gemini Pro**: ~0.5-1 records/giây
- **Phụ thuộc**: Độ dài text, độ phức tạp prompt

### Khuyến nghị
- Sử dụng checkpoint cho file > 100 records
- Tối ưu prompt để giảm token
- Chọn model phù hợp với yêu cầu

## 🤝 Đóng góp

Project này được thiết kế để dễ mở rộng:

1. **Thêm model mới**: Cập nhật `AVAILABLE_MODELS` trong config.py
2. **Thêm prompt template**: Cập nhật `DEFAULT_PROMPT_TEMPLATES`
3. **Thêm file format**: Cập nhật hàm `load_data()` và `save_data()`

## 📄 License

MIT License - Sử dụng tự do cho mục đích cá nhân và thương mại.

---

**Phiên bản**: 1.0.0  
**Ngày tạo**: 2025-01-24 