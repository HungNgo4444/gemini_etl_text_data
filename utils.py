import pandas as pd
import numpy as np
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import google.generativeai as genai
from tqdm import tqdm

# Import config
from config import (
    MAX_OUTPUT_TOKENS,
    TEMPERATURE, 
    TOP_P,
    TOP_K,
    MAX_RETRIES,
    RETRY_DELAY,
    REQUEST_DELAY,
    SUPPORTED_FORMATS,
    LOG_FILE,
    LOG_LEVEL
)

def setup_logging():
    """Thiết lập logging cho ứng dụng"""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def initialize_gemini(api_key, model_name):
    """Khởi tạo Gemini API với cấu hình"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
            }
        )
        logger.info(f"✅ Đã khởi tạo Gemini model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"❌ Lỗi khởi tạo Gemini: {str(e)}")
        return None

def validate_file_path(file_path):
    """Kiểm tra tính hợp lệ của đường dẫn file"""
    if not file_path:
        return False, "Đường dẫn file không được để trống"
    
    path = Path(file_path)
    
    # Kiểm tra file có tồn tại
    if not path.exists():
        return False, f"File không tồn tại: {file_path}"
    
    # Kiểm tra định dạng file
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        return False, f"Định dạng file không được hỗ trợ. Chỉ hỗ trợ: {', '.join(SUPPORTED_FORMATS)}"
    
    return True, "File hợp lệ"

def load_data(file_path):
    """Load dữ liệu từ file Excel hoặc CSV"""
    try:
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_ext == '.csv':
            # Thử nhiều encoding để tránh lỗi
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise Exception("Không thể đọc file CSV với các encoding thông dụng")
        else:
            raise Exception(f"Định dạng file không được hỗ trợ: {file_ext}")
        
        logger.info(f"✅ Đã load file: {file_path} ({len(df)} records)")
        return df
        
    except Exception as e:
        logger.error(f"❌ Lỗi load file: {str(e)}")
        return None

def save_data(df, file_path):
    """Lưu dữ liệu ra file Excel hoặc CSV"""
    try:
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.xlsx', '.xls']:
            df.to_excel(file_path, index=False)
        elif file_ext == '.csv':
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
        else:
            raise Exception(f"Định dạng file output không được hỗ trợ: {file_ext}")
        
        logger.info(f"✅ Đã lưu file: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi lưu file: {str(e)}")
        return False

def detect_message_column(df):
    """Tự động phát hiện cột chứa message"""
    possible_columns = ['MESSAGE']
    
    for col in possible_columns:
        if col in df.columns:
            return col
    
    # Nếu không tìm thấy, trả về cột đầu tiên có kiểu text
    for col in df.columns:
        if df[col].dtype == 'object':
            return col
    
    return None

def generate_output_filename(input_file):
    """Tạo tên file output dựa trên file input"""
    input_path = Path(input_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{input_path.stem}_ai_result_{timestamp}{input_path.suffix}"
    return str(input_path.parent / output_name)

def generate_checkpoint_filename(input_file):
    """Tạo tên file checkpoint dựa trên file input"""
    input_path = Path(input_file)
    checkpoint_name = f"{input_path.stem}_checkpoint{input_path.suffix}"
    return str(input_path.parent / checkpoint_name)

def load_checkpoint(checkpoint_file):
    """Load checkpoint nếu tồn tại"""
    try:
        if os.path.exists(checkpoint_file):
            df = load_data(checkpoint_file)
            if df is not None:
                logger.info(f"🔄 Đã load checkpoint: {checkpoint_file}")
                return df
    except Exception as e:
        logger.warning(f"⚠️ Lỗi load checkpoint: {str(e)}")
    
    return None

def save_checkpoint(df, checkpoint_file):
    """Lưu checkpoint"""
    try:
        save_data(df, checkpoint_file)
        logger.info(f"💾 Đã lưu checkpoint: {checkpoint_file}")
    except Exception as e:
        logger.error(f"❌ Lỗi lưu checkpoint: {str(e)}")

def process_text_with_ai(model, text, prompt, max_retries=MAX_RETRIES):
    """Xử lý text với AI model"""
    for attempt in range(max_retries):
        try:
            # Tạo prompt đầy đủ
            full_prompt = f"{prompt}\n\nNội dung cần xử lý:\n{text}\n\nKết quả:"
            
            # Gọi API
            response = model.generate_content(full_prompt)
            
            # Delay để tránh rate limit
            time.sleep(REQUEST_DELAY)
            
            if response and response.text:
                result = response.text.strip()
                logger.debug(f"✅ Xử lý thành công: {text[:50]}...")
                return result
            else:
                raise Exception("Không nhận được response từ AI")
                
        except Exception as e:
            error_msg = str(e)
            
            # Xử lý lỗi rate limit
            if "429" in error_msg or "quota" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = RETRY_DELAY * (attempt + 1)  # Tăng dần thời gian chờ
                    logger.warning(f"⏳ Rate limit reached. Chờ {wait_time} giây...")
                    time.sleep(wait_time)
                    continue
            
            # Xử lý lỗi khác
            logger.error(f"❌ Lỗi xử lý text (attempt {attempt + 1}): {error_msg}")
            
            if attempt < max_retries - 1:
                time.sleep(5)  # Chờ ngắn trước khi thử lại
                continue
    
    return f"Lỗi xử lý sau {max_retries} lần thử"

def process_multicolumn_with_ai(model, row_data, column_names, prompt, max_retries=MAX_RETRIES):
    """Xử lý nhiều cột với AI model"""
    try:
        # Tạo cấu trúc dữ liệu cho prompt
        data_structure = "\n".join([
            f"{i+1}. {col_name}: {clean_text(str(row_data.get(col_name, 'N/A')))}"
            for i, col_name in enumerate(column_names)
        ])
        
        # Tạo phần định nghĩa cột cho prompt
        column_definitions = "\n".join([
            f"- Cột {i+1} ({col_name}): {_get_column_description(col_name)}"
            for i, col_name in enumerate(column_names)
        ])
        
        # Tạo prompt đầy đủ với định nghĩa cột
        full_prompt = f"""
{prompt}

THÔNG TIN CÁC CỘT:
{column_definitions}

DỮ LIỆU CẦN XỬ LÝ:
{data_structure}

Kết quả:"""
        
        # Gọi hàm xử lý AI
        return process_text_with_ai(model, "", full_prompt, max_retries)
        
    except Exception as e:
        logger.error(f"❌ Lỗi xử lý multi-column: {str(e)}")
        return f"Lỗi xử lý multi-column: {str(e)}"

def _get_column_description(column_name):
    """Tạo mô tả cho cột dựa trên tên cột"""
    return column_name

def estimate_completion_time(processed, total, start_time):
    """Ước tính thời gian hoàn thành"""
    if processed == 0:
        return "Không thể ước tính"
    
    elapsed = time.time() - start_time
    rate = processed / elapsed  # records per second
    remaining = total - processed
    
    if rate > 0:
        eta_seconds = remaining / rate
        eta = datetime.now() + timedelta(seconds=eta_seconds)
        return f"Dự kiến hoàn thành: {eta.strftime('%H:%M:%S %d/%m/%Y')}"
    
    return "Không thể ước tính"

def print_progress_summary(processed, total, start_time, errors=0):
    """In báo cáo tiến trình chi tiết"""
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"📊 BÁO CÁO TIẾN TRÌNH:")
    print(f"✅ Đã xử lý: {processed}/{total} ({processed/total*100:.1f}%)")
    print(f"⚡ Tốc độ: {rate:.2f} records/giây")
    print(f"⏱️ Thời gian đã chạy: {elapsed/3600:.1f} giờ")
    print(f"❌ Lỗi: {errors}")
    print(f"⏰ {estimate_completion_time(processed, total, start_time)}")
    print(f"{'='*60}\n")

def validate_prompt(prompt):
    """Kiểm tra tính hợp lệ của prompt"""
    if not prompt or prompt.strip() == "":
        return False, "Prompt không được để trống"
    
    if len(prompt.strip()) < 10:
        return False, "Prompt quá ngắn (tối thiểu 10 ký tự)"
    
    return True, "Prompt hợp lệ"

def clean_text(text):
    """Làm sạch text trước khi xử lý"""
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # Loại bỏ các ký tự không mong muốn
    text = text.replace('\n\n\n', '\n\n')
    text = text.replace('\t', ' ')
    
    return text

def get_processing_stats(df, result_column):
    """Lấy thống kê quá trình xử lý"""
    if result_column not in df.columns:
        return {
            'total': len(df),
            'processed': 0,
            'remaining': len(df),
            'errors': 0
        }
    
    total = len(df)
    processed = len(df[df[result_column].notna() & (df[result_column] != "")])
    errors = len(df[df[result_column].str.contains("Lỗi", na=False)])
    remaining = total - processed
    
    return {
        'total': total,
        'processed': processed,
        'remaining': remaining,
        'errors': errors
    } 