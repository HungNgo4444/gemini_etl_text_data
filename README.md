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
- ⚡ **Batch Processing**: Xử lý 5-10 records/API call (tăng tốc 5-10x)

### 📁 Xử lý file đa dạng
- ✅ Hỗ trợ Excel (.xlsx, .xls) và CSV
- ✅ Tự động phát hiện encoding
- ✅ Tự động phát hiện cột text
- ✅ Hỗ trợ xử lý 1 cột hoặc nhiều cột cùng lúc
- ✅ Preview dữ liệu trước khi xử lý

### 💾 Checkpoint thông minh
- ✅ Lưu tiến trình tự động
- ✅ Tiếp tục từ điểm dừng khi bị gián đoạn
- ✅ Tự động xóa checkpoint khi hoàn thành

### ✍️ Prompt templates
- ✅ Templates có sẵn: tóm tắt, phân loại, cảm xúc, từ khóa, dịch thuật
- ✅ Hỗ trợ đọc prompt từ file .txt
- ✅ Hỗ trợ prompt tùy chỉnh
- ✅ Validation prompt input
- ✅ Tự động định nghĩa cột cho multi-column

### 📊 Monitoring & Reporting
- ✅ Progress bar real-time (batch-aware)
- ✅ Báo cáo tiến trình định kỳ
- ✅ Ước tính thời gian hoàn thành (với batch optimization)
- ✅ Thống kê lỗi và thành công
- ✅ Automatic fallback khi batch processing thất bại

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
  1. ID (VD: MSG_001)
  2. MESSAGE (VD: Khách hàng hỏi về sản phẩm...)
  3. AUTHOR (VD: Nguyễn Văn A)
  4. PRODUCT (VD: ChocoPie Truyền Thống)
  5. CHANNEL (VD: Facebook)
  6. DATE (VD: 2025-01-20)

💡 Tự động phát hiện cột text: 'MESSAGE'

🎯 Tùy chọn lựa chọn cột:
  1️⃣  Chọn 1 cột duy nhất
  2️⃣  Chọn nhiều cột để ghép lại

Chọn chế độ (1 hoặc 2): 2
📝 Nhập các số cột cần xử lý, cách nhau bằng dấu phẩy
   Ví dụ: 1,3,5 hoặc 2,4,6,8

Nhập các số cột (1-6): 2,3,4
✅ Đã chọn 3 cột:
  1. MESSAGE
  2. AUTHOR  
  3. PRODUCT
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
  7. Đọc prompt từ file (.txt)
  8. Tự nhập prompt

Chọn template (1-8): 7
Nhập đường dẫn file prompt (.txt): Prompt mẫu.txt
✅ Đã đọc thành công file: Prompt mẫu.txt
📝 Độ dài prompt: 6234 ký tự
Prompt preview: Bạn là một hệ thống phân loại và trích xuất thông tin sản phẩm...
```

#### Bước 7: Xác nhận và chạy
```
📋 BƯỚC 7: Tổng kết cấu hình
🤖 Model: gemma-3-27b-it
📁 File input: data.xlsx
📊 Chế độ: Nhiều cột (3 cột)
     1. MESSAGE
     2. AUTHOR
     3. PRODUCT
💾 Checkpoint: Có
✍️ Prompt: Bạn là chuyên gia phân tích dữ liệu khách hàng...

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

# Batch Processing
ENABLE_BATCH_PROCESSING = True  # Bật/tắt batch processing
BATCH_SIZE = 5                  # Số records xử lý cùng lúc
MAX_BATCH_SIZE = 20             # Giới hạn tối đa batch size
MIN_BATCH_SIZE = 1              # Giới hạn tối thiểu batch size
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

### 4. Sử dụng prompt từ file
```
# Tạo file prompt.txt với nội dung phức tạp
Bạn là một hệ thống phân loại sản phẩm chuyên nghiệp.
Hãy phân tích tin nhắn và trả về kết quả theo format:
Category|Product|Service|Tag|Note

Quy tắc phân loại:
1. Category: Chỉ được chọn từ danh sách...
2. Product: Tên sản phẩm cụ thể...
3. Service: Loại dịch vụ...
...

# Chọn option "Đọc prompt từ file (.txt)" và nhập đường dẫn
Input: "Tôi muốn mua ChocoPie vị đào"
Output: "Bánh ChocoPie|ChocoPie Vị Đào|Sản phẩm|Hỏi/Thảo luận nhắc đến sản phẩm|2"
```

### 5. Xử lý nhiều cột (Multi-column)
```
# Tạo data với nhiều cột thông tin
ID | MESSAGE | AUTHOR | PRODUCT | CHANNEL
1  | "Sản phẩm tuyệt vời!" | Nguyễn A | ChocoPie | Facebook
2  | "Cần hỗ trợ đổi trả" | Trần B | Bánh quy | Instagram

# Chọn chế độ "Nhiều cột để ghép lại"
# Nhập: 2,3,4 (MESSAGE, AUTHOR, PRODUCT)

# AI sẽ nhận được prompt với định nghĩa rõ ràng:
THÔNG TIN CÁC CỘT:
- Cột 1 (MESSAGE): MESSAGE
- Cột 2 (AUTHOR): AUTHOR  
- Cột 3 (PRODUCT): PRODUCT

DỮ LIỆU CẦN XỬ LÝ:
1. MESSAGE: Sản phẩm tuyệt vời!
2. AUTHOR: Nguyễn A
3. PRODUCT: ChocoPie

# Kết quả tích hợp thông tin từ cả 3 cột
Output: "Sentiment: Tích cực | Chủ đề: Review | Tóm tắt: Khách hàng đánh giá cao sản phẩm | Ghi chú: Nguyễn A review ChocoPie"
```

### 6. Batch Processing (Tối ưu tốc độ)
```
🚀 BẮT ĐẦU XỬ LÝ DỮ LIỆU
🎯 Sẽ xử lý 50 records  
⚡ Chế độ: Batch Processing
📦 Batch size: 5 records/batch
🔢 Số batch ước tính: 10
⏱️ Ước tính thời gian: ~0.1 giờ (cải thiện 5-10x)

# AI nhận được batch prompt:
HƯỚNG DẪN BATCH PROCESSING:
- Xử lý 5 mục dữ liệu dưới đây
- Trả về kết quả theo thứ tự tương ứng
- Format: [1] Kết quả 1\n[2] Kết quả 2\n[3] Kết quả 3...

DỮ LIỆU CẦN XỬ LÝ:
[1] Sản phẩm tuyệt vời!
[2] Giao hàng chậm
[3] Chất lượng ổn
[4] Rất hài lòng
[5] Bình thường

# AI trả về:
[1] Tích cực | Hài lòng
[2] Tiêu cực | Giao hàng  
[3] Trung tính | Chất lượng
[4] Tích cực | Hài lòng
[5] Trung tính | Bình thường

→ 1 API call xử lý 5 records thay vì 5 API calls
→ Giảm 80% thời gian xử lý
```

## 🚀 Batch Processing

**Tính năng mới**: Xử lý hàng loạt để tăng tốc độ 5-10 lần!

### Cách hoạt động
- Thay vì xử lý từng record, gom 5-20 records thành 1 batch
- 1 API call xử lý nhiều records cùng lúc
- Giảm 80-90% số lượng API calls

### Cấu hình
```python
# config.py
ENABLE_BATCH_PROCESSING = True
BATCH_SIZE = 5  # Số records mỗi batch
MAX_BATCH_SIZE = 20  # Giới hạn tối đa
```

### Performance
- **Trước**: 100 records = 100 API calls
- **Sau**: 100 records = 20 API calls (batch size 5)
- **Tăng tốc**: 5-10x nhanh hơn

## 🧵 Parallel Processing

**Tính năng tiên tiến**: Xử lý song song để tăng tốc độ 15-30 lần!

### Cách hoạt động
- Chạy nhiều threads đồng thời
- Mỗi thread xử lý 1 batch riêng biệt
- Kết hợp Parallel + Batch Processing

### Cấu hình an toàn
```python
# config.py
ENABLE_PARALLEL_PROCESSING = True
MAX_CONCURRENT_THREADS = 2         # Số threads song song
THREAD_BATCH_SIZE = 5              # Batch size cho mỗi thread
RATE_LIMIT_DELAY = 2.0             # Delay tránh rate limit
CIRCUIT_BREAKER_THRESHOLD = 5      # Bảo vệ khỏi lỗi liên tiếp
```

### Tính năng bảo vệ
- **Circuit Breaker**: Tự động ngừng khi quá nhiều lỗi
- **Rate Limiting**: Delay giữa các requests
- **Auto Fallback**: Tự động chuyển về batch/single mode khi lỗi
- **Timeout Protection**: Timeout cho mỗi thread

### Performance Comparison
```
📊 HIỆU SUẤT XỬ LÝ 100 RECORDS:

Single Processing:    ~5 phút    (baseline)
Batch Processing:     ~1 phút    (5x faster)
Parallel Processing:  ~20-30s    (15-30x faster)
```

### Ví dụ sử dụng
```python
# Tự động detect mode dựa trên số lượng dữ liệu:
# < 2 records: Single Processing
# < MAX_CONCURRENT_THREADS: Batch Processing  
# >= MAX_CONCURRENT_THREADS: Parallel Processing
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