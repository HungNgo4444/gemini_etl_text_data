import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===========================
# API PROVIDER CONFIGURATION
# ===========================
API_PROVIDER = ""  # Sẽ được input từ user: "gemini" hoặc "openai"

# ===========================
# GEMINI API CONFIGURATION
# ===========================
# Người dùng sẽ điền thông tin này khi chạy chương trình
GEMINI_API_KEY = ""  # Sẽ được input từ user
GEMINI_MODEL_NAME = ""      # Sẽ được input từ user (mặc định: gemma-3-27b-it)

# Các model Gemini phổ biến
AVAILABLE_GEMINI_MODELS = [
    "gemma-3-27b-it",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash"
]

# ===========================
# OPENAI API CONFIGURATION
# ===========================
OPENAI_API_KEY = ""  # Sẽ được input từ user
OPENAI_MODEL_NAME = ""  # Sẽ được input từ user (mặc định: gpt-3.5-turbo)

# Các model OpenAI phổ biến
AVAILABLE_OPENAI_MODELS = [
    "gpt-4.1-nano",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1"
]

# ===========================
# LEGACY CONFIGURATION (để tương thích ngược)
# ===========================
MODEL_NAME = ""      # Deprecated, sử dụng GEMINI_MODEL_NAME hoặc OPENAI_MODEL_NAME
AVAILABLE_MODELS = AVAILABLE_GEMINI_MODELS  # Deprecated

# ===========================
# FILE CONFIGURATION
# ===========================
INPUT_FILE = ""        # Sẽ được input từ user
OUTPUT_FILE = ""       # Sẽ được tự động tạo dựa trên input file
MESSAGE_COLUMN = "MESSAGE"  # Mặc định, có thể thay đổi
AI_RESULT_COLUMN = "AI_RESULT"  # Cột kết quả AI

# ===========================
# PROCESSING CONFIGURATION
# ===========================
USE_CHECKPOINT = True   # Mặc định sử dụng checkpoint
CHECKPOINT_FILE = ""    # Sẽ được tự động tạo dựa trên input file
CHECKPOINT_INTERVAL = 30   # Lưu checkpoint mỗi 60 records (khớp với ASYNC_CHUNK_SIZE)
PROGRESS_REPORT_INTERVAL = 30  # Báo cáo tiến trình mỗi 60 records

# User prompt - sẽ được input từ user
USER_PROMPT = ""

# ===========================
# BATCH PROCESSING CONFIGURATION
# ===========================
ENABLE_BATCH_PROCESSING = True  # Bật/tắt batch processing
BATCH_SIZE = 10  # Số records xử lý cùng lúc trong 1 batch
MAX_BATCH_SIZE = 100  # Giới hạn tối đa batch size
MIN_BATCH_SIZE = 1   # Giới hạn tối thiểu batch size

# ===========================
# PARALLEL PROCESSING CONFIGURATION
# ===========================
ENABLE_PARALLEL_PROCESSING = False  # Bật/tắt parallel processing (legacy)
MAX_CONCURRENT_THREADS = 2         # Số threads chạy song song (legacy)
THREAD_BATCH_SIZE = 25              # Số records mỗi thread xử lý trong 1 batch (legacy)
RATE_LIMIT_DELAY = 3.0             # Delay giữa các requests (seconds) (legacy)
MAX_RETRIES_PER_THREAD = 2         # Số lần retry cho mỗi thread (legacy)
THREAD_TIMEOUT = 120               # Timeout cho mỗi thread (seconds) (legacy)
CIRCUIT_BREAKER_THRESHOLD = 3      # Số lỗi liên tiếp để kích hoạt circuit breaker (legacy)

# ===========================
# ASYNC PROCESSING CONFIGURATION - OPTIMIZED FOR GPT-4.1 MINI PAID TIER
# ===========================
ENABLE_ASYNC_PROCESSING = True     # Bật/tắt async processing (MẶC ĐỊNH BẬT)
MAX_CONCURRENT_REQUESTS = 10        # Giảm concurrent để tránh burst (từ 5 → 3)
ASYNC_BATCH_SIZE = 5              # Giảm batch size để ổn định (từ 5 → 3)
ASYNC_CHUNK_SIZE = 100             # Giảm chunk size để checkpoint thường xuyên (từ 100 → 60)
ASYNC_RATE_LIMIT_RPM = 1000        # An toàn với 500 RPM limit (80% của limit)
ASYNC_TIMEOUT = 90                # Tăng timeout cho GPT-4.1 Mini (từ 60 → 90)
ASYNC_MAX_RETRIES = 3             # Giữ nguyên số lần retry
ASYNC_RETRY_DELAY = 5             # Tăng delay base cho exponential backoff (từ 2 → 5)
ASYNC_ENABLE_RATE_LIMITER = True  # BẮT BUỘC bật rate limiter

# ===========================
# MODEL PARAMETERS
# ===========================
MAX_OUTPUT_TOKENS = 30000
TEMPERATURE = 0
TOP_P = 0.8
TOP_K = 40

# ===========================
# RETRY & RATE LIMIT CONFIGURATION
# ===========================
MAX_RETRIES = 3
RETRY_DELAY = 60  # seconds
REQUEST_DELAY = 5.0  # seconds between requests for safety

# ===========================
# FILE FORMAT SUPPORT
# ===========================
SUPPORTED_FORMATS = ['.xlsx', '.csv', '.xls']

# ===========================
# DEFAULT PROMPTS TEMPLATES
# ===========================
DEFAULT_PROMPT_TEMPLATES = {
    "summarize": "Hãy tóm tắt nội dung sau một cách ngắn gọn và rõ ràng bằng tiếng Việt:",
    "classify": "Hãy phân loại nội dung sau vào một trong các danh mục phù hợp:",
    "sentiment": "Hãy phân tích cảm xúc (tích cực/tiêu cực/trung tính) của nội dung sau:",
    "extract_keywords": "Hãy trích xuất các từ khóa quan trọng từ nội dung sau:",
    "translate": "Hãy dịch nội dung sau sang tiếng Việt:",
    "custom": "Hãy xử lý nội dung sau theo yêu cầu:",
    "json_classify": """Bạn là một hệ thống phân loại dữ liệu. Hãy phân tích nội dung và trả về kết quả theo định dạng JSON chính xác như sau:

{
    "category": "string hoặc null", 
    "product": "string hoặc null",
    "service": "string hoặc null", 
    "tag": "string hoặc null",
    "note_1": "string hoặc null"
}

QUY TẮC QUAN TRỌNG:
- Chỉ trả về JSON object duy nhất, không có text khác
- Sử dụng null cho các trường không tìm thấy thông tin
- Đảm bảo JSON syntax chính xác với double quotes
- Không sử dụng comments trong JSON"""
}

# ===========================
# LOGGING CONFIGURATION
# ===========================
LOG_FILE = "ai_etl_data.log"
LOG_LEVEL = "INFO"

# ===========================
# ERROR RETRY CONFIGURATION
# ===========================
ENABLE_ERROR_RETRY = True         # Bật/tắt tính năng retry failed rows trước khi hoàn thành
ERROR_RETRY_MAX_ATTEMPTS = 2      # Số lần retry tối đa cho mỗi row bị lỗi
ERROR_RETRY_DELAY_BASE = 2        # Delay cơ bản giữa các lần retry (seconds)
ERROR_RETRY_EXPONENTIAL = True    # Sử dụng exponential backoff (2s, 4s, 8s...)

# ===========================
# JSON OUTPUT CONFIGURATION
# ===========================
ENABLE_JSON_OUTPUT = False  # Sẽ được input từ user
JSON_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": ["string", "null"]},
        "product": {"type": ["string", "null"]}, 
        "service": {"type": ["string", "null"]},
        "tag": {"type": ["string", "null"]},
        "note_1": {"type": ["string", "null"]}
    },
    "required": ["category", "product", "service", "tag", "note_1"]
}

# JSON Parsing configuration
JSON_PARSE_FALLBACK_TO_TEXT = True  # Fallback về text parsing nếu JSON thất bại
JSON_VALIDATE_SCHEMA = True         # Validate JSON schema
JSON_REPAIR_MALFORMED = True        # Thử sửa JSON bị lỗi format

# =========================== 