import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===========================
# API CONFIGURATION
# ===========================
# Người dùng sẽ điền thông tin này khi chạy chương trình
GEMINI_API_KEY = ""  # Sẽ được input từ user
MODEL_NAME = ""      # Sẽ được input từ user (mặc định: gemma-3-27b-it)

# Các model Gemini phổ biến
AVAILABLE_MODELS = [
    "gemma-3-27b-it",
    "gemma-3n-e4b-it",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash"
]

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
BATCH_SIZE = 20  # Số records xử lý cùng lúc trong 1 batch
MAX_BATCH_SIZE = 100  # Giới hạn tối đa batch size
MIN_BATCH_SIZE = 1   # Giới hạn tối thiểu batch size

# ===========================
# MODEL PARAMETERS
# ===========================
MAX_OUTPUT_TOKENS = 1024
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