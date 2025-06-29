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
    "gemma-3n-e4b-it",
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
CHECKPOINT_INTERVAL = 100  # Lưu checkpoint mỗi 100 records
PROGRESS_REPORT_INTERVAL = 100  # Báo cáo tiến trình mỗi 100 records

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
# ASYNC PROCESSING CONFIGURATION
# ===========================
ENABLE_ASYNC_PROCESSING = True     # Bật/tắt async processing (MẶC ĐỊNH BẬT)
MAX_CONCURRENT_REQUESTS = 5       # Số requests đồng thời tối đa (semaphore limit)
ASYNC_BATCH_SIZE = 5             # Số items xử lý trong 1 batch async (TĂNG ĐỂ NHANH HƠN!)
ASYNC_CHUNK_SIZE = 300             # Số items xử lý trong 1 chunk (để chia nhỏ workload)
ASYNC_RATE_LIMIT_RPM = 300          # Rate limit: requests per minute
ASYNC_TIMEOUT = 60                 # Timeout cho mỗi async request (seconds)
ASYNC_MAX_RETRIES = 3              # Số lần retry cho async requests
ASYNC_RETRY_DELAY = 2              # Delay base cho exponential backoff (seconds)
ASYNC_ENABLE_RATE_LIMITER = True   # Bật dynamic rate limiter

# ===========================
# MODEL PARAMETERS
# ===========================
MAX_OUTPUT_TOKENS = 30000
TEMPERATURE = 0.3
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
    "custom": "Hãy xử lý nội dung sau theo yêu cầu:"
}

# ===========================
# LOGGING CONFIGURATION
# ===========================
LOG_FILE = "ai_etl_data.log"
LOG_LEVEL = "INFO" 