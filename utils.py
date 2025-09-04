import pandas as pd
import numpy as np
import os
import time
import logging
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
import google.generativeai as genai
import asyncio
import jsonschema
from jsonschema import validate, ValidationError

# Import async processor
try:
    from async_processor import process_data_async
    ASYNC_AVAILABLE = True
except ImportError as e:
    ASYNC_AVAILABLE = False
    print(f"⚠️ Async processing không khả dụng: {e}")

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
    LOG_LEVEL,
    BATCH_SIZE,
    ENABLE_BATCH_PROCESSING,
    ENABLE_PARALLEL_PROCESSING,
    ENABLE_ASYNC_PROCESSING,
    MAX_CONCURRENT_THREADS,
    THREAD_BATCH_SIZE,
    RATE_LIMIT_DELAY,
    MAX_RETRIES_PER_THREAD,
    THREAD_TIMEOUT,
    CIRCUIT_BREAKER_THRESHOLD,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    OPENAI_API_KEY,
    OPENAI_MODEL_NAME,
    JSON_OUTPUT_SCHEMA,
    JSON_VALIDATE_SCHEMA,
    JSON_REPAIR_MALFORMED,
    JSON_PARSE_FALLBACK_TO_TEXT
)

def check_openai_availability():
    """Kiểm tra OpenAI availability động"""
    try:
        from openai import OpenAI
        return True, OpenAI
    except ImportError:
        try:
            import openai
            if hasattr(openai, 'OpenAI'):
                return True, openai.OpenAI
            else:
                return False, None
        except ImportError:
            return False, None

# Kiểm tra ban đầu
OPENAI_AVAILABLE, OpenAI = check_openai_availability()
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import traceback

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

def initialize_gemini(api_key, model_name, fine_tuned_model_info=None):
    """Khởi tạo Gemini API với cấu hình (hỗ trợ fine-tuned models)"""
    try:
        genai.configure(api_key=api_key)
        
        # Nếu là fine-tuned model, sử dụng model_id
        if fine_tuned_model_info and fine_tuned_model_info.get('model_id'):
            actual_model_name = fine_tuned_model_info['model_id']
            logger.info(f"🎯 Sử dụng fine-tuned model: {actual_model_name}")
        else:
            actual_model_name = model_name
            logger.info(f"🤖 Sử dụng standard model: {actual_model_name}")
        
        model = genai.GenerativeModel(
            model_name=actual_model_name,
            generation_config={
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
            }
        )
        
        # Attach fine-tuned model info nếu có
        if fine_tuned_model_info:
            model._fine_tuned_info = fine_tuned_model_info
        
        # Attach API provider info
        model._api_provider = 'gemini'
        model._api_key = api_key
        model._model_name = actual_model_name
        
        logger.info(f"✅ Đã khởi tạo Gemini model: {actual_model_name}")
        return model
    except Exception as e:
        logger.error(f"❌ Lỗi khởi tạo Gemini: {str(e)}")
        return None

def initialize_openai(api_key, model_name):
    """Khởi tạo OpenAI API với cấu hình"""
    # Kiểm tra lại động
    available, openai_class = check_openai_availability()
    if not available or openai_class is None:
        raise ImportError("OpenAI package không có sẵn hoặc version không tương thích. Vui lòng cài đặt: pip install openai>=1.0.0")
    
    try:
        client = openai_class(api_key=api_key)
        
        # Test connection
        test_response = client.models.list()
        
        # Create a wrapper object that behaves like Gemini model
        class OpenAIModelWrapper:
            def __init__(self, client, model_name):
                self.client = client
                self.model_name = model_name
                self._api_provider = 'openai'
                self._fine_tuned_info = None
                self._api_key = client.api_key
                self._model_name = model_name
            
            def generate_content(self, prompt):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=MAX_OUTPUT_TOKENS,
                        temperature=TEMPERATURE,
                        top_p=TOP_P
                    )
                    
                    if response.choices and response.choices[0].message:
                        # Create response object similar to Gemini
                        class ResponseWrapper:
                            def __init__(self, text):
                                self.text = text
                        
                        return ResponseWrapper(response.choices[0].message.content)
                    else:
                        return None
                        
                except Exception as e:
                    raise Exception(f"OpenAI API error: {str(e)}")
        
        model = OpenAIModelWrapper(client, model_name)
        logger.info(f"✅ Đã khởi tạo OpenAI model: {model_name}")
        return model
        
    except Exception as e:
        logger.error(f"❌ Lỗi khởi tạo OpenAI: {str(e)}")
        return None

def initialize_ai_model(api_provider, api_key, model_name, fine_tuned_model_info=None):
    """Khởi tạo AI model dựa trên provider"""
    if api_provider == 'gemini':
        model = initialize_gemini(api_key, model_name, fine_tuned_model_info)
        if model:
            # Thêm thông tin API provider cho async processing
            model._api_provider = "gemini"
        return model
    elif api_provider == 'openai':
        # Kiểm tra lại động
        available, _ = check_openai_availability()
        if not available:
            logger.error("❌ OpenAI package không có sẵn. Vui lòng cài đặt: pip install openai>=1.0.0")
            return None
        model = initialize_openai(api_key, model_name)
        if model:
            # Thêm thông tin API provider cho async processing
            model._api_provider = "openai"
        return model
    else:
        logger.error(f"❌ API provider không được hỗ trợ: {api_provider}")
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

def generate_output_filename(input_file, output_format=None):
    """Tạo tên file output dựa trên file input và định dạng output"""
    input_path = Path(input_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_format == 'csv':
        output_name = f"{input_path.stem}_ai_result_{timestamp}.csv"
    else:
        output_name = f"{input_path.stem}_ai_result_{timestamp}.xlsx"
    return str(input_path.parent / output_name)

def generate_checkpoint_filename(input_file, output_format=None):
    """Tạo tên file checkpoint dựa trên file input và định dạng output"""
    input_path = Path(input_file)
    if output_format == 'csv':
        checkpoint_name = f"{input_path.stem}_checkpoint.csv"
    else:
        checkpoint_name = f"{input_path.stem}_checkpoint.xlsx"
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
    """Xử lý text với AI model (hỗ trợ fine-tuned models với prompt context)"""
    for attempt in range(max_retries):
        try:
            # Kiểm tra nếu là fine-tuned model với prompt context
            if hasattr(model, '_fine_tuned_info') and model._fine_tuned_info:
                fine_tuned_info = model._fine_tuned_info
                
                # Nếu model có prompt context, sử dụng format đã training
                if fine_tuned_info.get('requires_context', False):
                    prompt_context = fine_tuned_info.get('prompt_context', {})
                    context_template = prompt_context.get('template', '')
                    
                    if context_template:
                        # Apply prompt context template
                        full_prompt = apply_fine_tuned_prompt_context(
                            text, prompt, context_template
                        )
                        logger.debug(f"🎯 Sử dụng fine-tuned prompt context")
                    else:
                        full_prompt = f"{prompt}\n\nNội dung cần xử lý:\n{text}\n\nKết quả:"
                else:
                    full_prompt = f"{prompt}\n\nNội dung cần xử lý:\n{text}\n\nKết quả:"
            else:
                # Standard model processing
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
            
            # Xử lý lỗi rate limit (cho cả Gemini và OpenAI)
            if ("429" in error_msg or "quota" in error_msg.lower() or 
                "rate_limit" in error_msg.lower() or "too_many_requests" in error_msg.lower()):
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

def process_batch_with_ai(model, batch_data, prompt, max_retries=MAX_RETRIES):
    """Xử lý batch data với AI model - single column mode"""
    try:
        # Tạo prompt cho batch processing
        batch_items = []
        for i, item in enumerate(batch_data, 1):
            cleaned_text = clean_text(str(item))
            batch_items.append(f"[{i}] {cleaned_text}")
        
        batch_content = "\n\n".join(batch_items)
        
        # Tạo prompt đầy đủ cho batch
        full_prompt = f"""{prompt}

HƯỚNG DẪN BATCH PROCESSING:
- Xử lý {len(batch_data)} mục dữ liệu dưới đây
- Trả về kết quả theo thứ tự tương ứng
- Format: [1] Kết quả 1\n[2] Kết quả 2\n[3] Kết quả 3...

DỮ LIỆU CẦN XỬ LÝ:
{batch_content}

KẾT QUẢ:"""

        # Gọi AI
        for attempt in range(max_retries):
            try:
                response = model.generate_content(full_prompt)
                
                # Delay để tránh rate limit  
                time.sleep(REQUEST_DELAY)
                
                if response and response.text:
                    # Parse kết quả batch
                    results = parse_batch_results(response.text.strip(), len(batch_data))
                    logger.debug(f"✅ Xử lý batch thành công: {len(results)} results")
                    return results
                else:
                    raise Exception("Không nhận được response từ AI")
                    
            except Exception as e:
                error_msg = str(e)
                
                # Xử lý lỗi rate limit (cho cả Gemini và OpenAI)
                if ("429" in error_msg or "quota" in error_msg.lower() or 
                    "rate_limit" in error_msg.lower() or "too_many_requests" in error_msg.lower()):
                    if attempt < max_retries - 1:
                        wait_time = RETRY_DELAY * (attempt + 1)
                        logger.warning(f"⏳ Rate limit reached. Chờ {wait_time} giây...")
                        time.sleep(wait_time)
                        continue
                
                # Xử lý lỗi khác
                logger.error(f"❌ Lỗi xử lý batch (attempt {attempt + 1}): {error_msg}")
                
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
        
        # Nếu batch thất bại, fallback về single processing
        logger.warning("⚠️ Batch processing thất bại, fallback về single processing")
        results = []
        for item in batch_data:
            result = process_text_with_ai(model, str(item), prompt, max_retries)
            results.append(result)
        return results
        
    except Exception as e:
        logger.error(f"❌ Lỗi xử lý batch: {str(e)}")
        # Fallback về single processing
        results = []
        for item in batch_data:
            result = process_text_with_ai(model, str(item), prompt, max_retries)
            results.append(result)
        return results

def process_multicolumn_batch_with_ai(model, batch_rows, column_names, prompt, max_retries=MAX_RETRIES):
    """Xử lý batch data với AI model - multi-column mode"""
    try:
        # Tạo prompt cho multi-column batch processing
        batch_items = []
        
        # Tạo phần định nghĩa cột
        column_definitions = "\n".join([
            f"- Cột {i+1} ({col_name}): {_get_column_description(col_name)}"
            for i, col_name in enumerate(column_names)
        ])
        
        # Tạo dữ liệu batch
        for i, row_data in enumerate(batch_rows, 1):
            data_structure = "\n".join([
                f"  {j+1}. {col_name}: {clean_text(str(row_data.get(col_name, 'N/A')))}"
                for j, col_name in enumerate(column_names)
            ])
            batch_items.append(f"[{i}]\n{data_structure}")
        
        batch_content = "\n\n".join(batch_items)
        
        # Tạo prompt đầy đủ cho multi-column batch
        full_prompt = f"""{prompt}

THÔNG TIN CÁC CỘT:
{column_definitions}

HƯỚNG DẪN BATCH PROCESSING:
- Xử lý {len(batch_rows)} mục dữ liệu dưới đây
- Mỗi mục có {len(column_names)} cột thông tin
- Trả về kết quả theo thứ tự tương ứng
- Format: [1] Kết quả 1\n[2] Kết quả 2\n[3] Kết quả 3...

DỮ LIỆU CẦN XỬ LÝ:
{batch_content}

KẾT QUẢ:"""

        # Gọi AI
        for attempt in range(max_retries):
            try:
                response = model.generate_content(full_prompt)
                
                # Delay để tránh rate limit
                time.sleep(REQUEST_DELAY)
                
                if response and response.text:
                    # Parse kết quả batch
                    results = parse_batch_results(response.text.strip(), len(batch_rows))
                    logger.debug(f"✅ Xử lý multi-column batch thành công: {len(results)} results")
                    return results
                else:
                    raise Exception("Không nhận được response từ AI")
                    
            except Exception as e:
                error_msg = str(e)
                
                # Xử lý lỗi rate limit (cho cả Gemini và OpenAI)
                if ("429" in error_msg or "quota" in error_msg.lower() or 
                    "rate_limit" in error_msg.lower() or "too_many_requests" in error_msg.lower()):
                    if attempt < max_retries - 1:
                        wait_time = RETRY_DELAY * (attempt + 1)
                        logger.warning(f"⏳ Rate limit reached. Chờ {wait_time} giây...")
                        time.sleep(wait_time)
                        continue
                
                # Xử lý lỗi khác
                logger.error(f"❌ Lỗi xử lý multi-column batch (attempt {attempt + 1}): {error_msg}")
                
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
        
        # Nếu batch thất bại, fallback về single processing
        logger.warning("⚠️ Multi-column batch processing thất bại, fallback về single processing")
        results = []
        for row_data in batch_rows:
            result = process_multicolumn_with_ai(model, row_data, column_names, prompt, max_retries)
            results.append(result)
        return results
        
    except Exception as e:
        logger.error(f"❌ Lỗi xử lý multi-column batch: {str(e)}")
        # Fallback về single processing
        results = []
        for row_data in batch_rows:
            result = process_multicolumn_with_ai(model, row_data, column_names, prompt, max_retries)
            results.append(result)
        return results

def parse_batch_results(response_text, expected_count):
    """Parse kết quả từ batch response"""
    try:
        results = []
        lines = response_text.split('\n')
        
        current_result = ""
        current_index = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Kiểm tra nếu là dòng bắt đầu với [số]
            if line.startswith('[') and ']' in line:
                # Lưu kết quả trước đó nếu có
                if current_result and current_index > 0:
                    results.append(current_result.strip())
                
                # Bắt đầu kết quả mới
                bracket_end = line.find(']')
                try:
                    index = int(line[1:bracket_end])
                    current_index = index
                    current_result = line[bracket_end + 1:].strip()
                except ValueError:
                    # Nếu không parse được số, coi như là nội dung
                    current_result += " " + line
            else:
                # Tiếp tục nội dung của kết quả hiện tại
                current_result += " " + line
        
        # Lưu kết quả cuối cùng
        if current_result and current_index > 0:
            results.append(current_result.strip())
        
        # Đảm bảo số lượng kết quả đúng
        while len(results) < expected_count:
            results.append("Lỗi parse kết quả")
        
        # Cắt bớt nếu có quá nhiều kết quả
        results = results[:expected_count]
        
        logger.debug(f"✅ Parse batch results: {len(results)}/{expected_count}")
        return results
        
    except Exception as e:
        logger.error(f"❌ Lỗi parse batch results: {str(e)}")
        # Fallback: trả về response nguyên cho tất cả
        return [response_text] * expected_count

# ===========================
# PARALLEL PROCESSING FUNCTIONS
# ===========================

class CircuitBreaker:
    """Circuit breaker pattern cho parallel processing"""
    def __init__(self, threshold=CIRCUIT_BREAKER_THRESHOLD):
        self.threshold = threshold
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.reset_timeout = 60  # seconds
        self._lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Gọi function với circuit breaker protection"""
        with self._lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.reset_timeout:
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    self.failure_count = 0
                return result
            
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.threshold:
                    self.state = 'OPEN'
                    logger.warning(f"🔴 Circuit breaker OPEN - {self.failure_count} failures")
                
                raise e

def process_parallel_batch_worker(thread_id, model, batch_data, column_names, prompt, is_multicolumn, circuit_breaker, progress_queue):
    """Worker function cho parallel processing"""
    try:
        logger.info(f"🚀 Thread {thread_id} bắt đầu xử lý {len(batch_data)} items")
        
        # Delay staggered để tránh rate limit
        time.sleep(thread_id * RATE_LIMIT_DELAY)
        
        # Xử lý với circuit breaker protection
        if is_multicolumn:
            results = circuit_breaker.call(
                process_multicolumn_batch_with_ai,
                model, batch_data, column_names, prompt, MAX_RETRIES_PER_THREAD
            )
        else:
            results = circuit_breaker.call(
                process_batch_with_ai,
                model, batch_data, prompt, MAX_RETRIES_PER_THREAD
            )
        
        # Report progress
        progress_queue.put(('success', thread_id, len(batch_data)))
        logger.info(f"✅ Thread {thread_id} hoàn thành thành công")
        
        return results
        
    except Exception as e:
        error_msg = f"Thread {thread_id} lỗi: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.debug(traceback.format_exc())
        
        # Report error
        progress_queue.put(('error', thread_id, str(e)))
        
        # Fallback về single processing cho batch này
        results = []
        try:
            if is_multicolumn:
                for row_data in batch_data:
                    result = process_multicolumn_with_ai(model, row_data, column_names, prompt, MAX_RETRIES_PER_THREAD)
                    results.append(result)
            else:
                for item in batch_data:
                    result = process_text_with_ai(model, str(item), prompt, MAX_RETRIES_PER_THREAD)
                    results.append(result)
        except Exception as fallback_error:
            logger.error(f"❌ Thread {thread_id} fallback cũng thất bại: {str(fallback_error)}")
            results = [f"Lỗi xử lý: {str(e)}"] * len(batch_data)
        
        return results

def process_data_with_async(model, data, column_names, prompt, is_multicolumn=False, checkpoint_callback=None, checkpoint_interval=None, use_json=False):
    """Xử lý dữ liệu với Async Processing (thay thế parallel processing) với checkpoint support"""
    # Import config tại runtime để tránh lỗi circular import
    try:
        from config import ENABLE_ASYNC_PROCESSING
    except ImportError as e:
        logger.error(f"❌ Không thể import config: {e}")
        return process_data_batch_only(model, data, column_names, prompt, is_multicolumn)
    
    if not ASYNC_AVAILABLE:
        logger.warning("⚠️ Async processing không khả dụng, fallback về batch processing")
        return process_data_batch_only(model, data, column_names, prompt, is_multicolumn)
    
    if not ENABLE_ASYNC_PROCESSING:
        logger.info("🔄 Async processing bị tắt, chuyển về batch processing")
        return process_data_batch_only(model, data, column_names, prompt, is_multicolumn)
    
    try:
        total_items = len(data)
        logger.info(f"🚀 Starting async processing: {total_items} items")
        
        # Extract API info từ model - LẤY TRỰC TIẾP TỪ MODEL OBJECT
        api_provider = getattr(model, '_api_provider', 'gemini')
        
        # Lấy API key và model name từ model object (đã được set khi initialize)
        if hasattr(model, '_api_key') and hasattr(model, '_model_name'):
            api_key = model._api_key
            model_name = model._model_name
            logger.info(f"🔑 Sử dụng API key từ model: {api_provider} - {model_name}")
        else:
            # Không có API key trong model, fallback về batch processing
            logger.warning("⚠️ Model không có API key cho async processing, fallback về batch processing")
            return process_data_batch_only(model, data, column_names, prompt, is_multicolumn)
        
        # Prepare data cho async processing
        if is_multicolumn:
            # Convert DataFrame rows to dict cho multicolumn
            data_for_async = []
            for _, row in data.iterrows() if hasattr(data, 'iterrows') else enumerate(data):
                if hasattr(data, 'iterrows'):
                    data_for_async.append(row.to_dict())
                else:
                    # Nếu data là list of dicts
                    data_for_async.append(row)
        else:
            # Single column data
            if hasattr(data, 'tolist'):
                data_for_async = data.tolist()
            else:
                data_for_async = list(data)
        
        # Run async processing với checkpoint support
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                process_data_async(
                    api_provider=api_provider,
                    api_key=api_key,
                    model_name=model_name,
                    data=data_for_async,
                    prompt_template=prompt,
                    is_multicolumn=is_multicolumn,
                    column_names=column_names,
                    checkpoint_callback=checkpoint_callback,
                    checkpoint_interval=checkpoint_interval,
                    use_json=use_json
                )
            )
            
            logger.info(f"✅ Async processing completed: {len(results)} results")
            return results
            
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"❌ Async processing failed: {str(e)}")
        logger.debug(traceback.format_exc())
        logger.info("🔄 Fallback to batch processing")
        return process_data_batch_only(model, data, column_names, prompt, is_multicolumn)

def process_data_parallel(model, data, column_names, prompt, is_multicolumn=False):
    """Xử lý dữ liệu với parallel processing (DEPRECATED - sử dụng async thay thế)"""
    logger.warning("⚠️ process_data_parallel is deprecated, using async processing instead")
    return process_data_with_async(model, data, column_names, prompt, is_multicolumn)

def process_data_batch_only(model, data, column_names, prompt, is_multicolumn=False):
    """Xử lý dữ liệu chỉ với batch processing (không parallel/async)"""
    try:
        total_items = len(data)
        batch_size = BATCH_SIZE
        all_results = []
        
        logger.info(f"📦 Batch processing: {total_items} items với batch size {batch_size}")
        
        with tqdm(total=total_items, desc="🔄 Batch Processing") as pbar:
            for i in range(0, total_items, batch_size):
                batch = data[i:i+batch_size]
                
                if is_multicolumn:
                    results = process_multicolumn_batch_with_ai(model, batch, column_names, prompt)
                else:
                    results = process_batch_with_ai(model, batch, prompt)
                
                all_results.extend(results)
                pbar.update(len(batch))
        
        return all_results
        
    except Exception as e:
        logger.error(f"❌ Lỗi batch processing: {str(e)}")
        # Fallback về single processing
        results = []
        for item in data:
            if is_multicolumn:
                result = process_multicolumn_with_ai(model, item, column_names, prompt)
            else:
                result = process_text_with_ai(model, str(item), prompt)
            results.append(result)
        return results

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
    
    # Tìm tất cả error patterns
    error_patterns = [
        "Lỗi",           # Vietnamese error
        "Batch error",   # Async batch error
        "failed after",  # Batch failed after X attempts
        "HTTP 429",      # Rate limit error
        "Timeout",       # Timeout error
        "Connection error"  # Connection error
    ]
    
    # Tạo mask cho errors
    error_mask = pd.Series([False] * len(df))
    for pattern in error_patterns:
        pattern_mask = df[result_column].astype(str).str.contains(pattern, case=False, na=False)
        error_mask = error_mask | pattern_mask
    
    errors = error_mask.sum()
    
    # Processed = có data và không phải error
    has_data_mask = df[result_column].notna() & (df[result_column] != "")
    processed = (has_data_mask & ~error_mask).sum()
    remaining = total - has_data_mask.sum()  # Chưa có data gì cả
    
    return {
        'total': total,
        'processed': processed,  # Thành công thực sự
        'remaining': remaining,  # Chưa xử lý
        'errors': errors        # Có lỗi
    }

def apply_fine_tuned_prompt_context(text, prompt, context_template):
    """
    Apply prompt context template cho fine-tuned model
    
    Args:
        text: Input text
        prompt: User prompt  
        context_template: Template từ fine-tuning
        
    Returns:
        str: Formatted prompt theo fine-tuned model
    """
    try:
        # Tạo context data
        context_data = {
            'input_data': text,
            'input_text': text,
            'content': text,
            'message': text,
            'context_info': prompt,
            'context_data': prompt,
            'metadata': prompt,
            'additional_context': prompt
        }
        
        # Apply template với placeholder replacement
        formatted_prompt = context_template
        
        for key, value in context_data.items():
            placeholder = f"{{{key}}}"
            if placeholder in formatted_prompt:
                formatted_prompt = formatted_prompt.replace(placeholder, str(value))
        
        logger.debug(f"📝 Applied fine-tuned context template")
        return formatted_prompt
        
    except Exception as e:
        logger.warning(f"⚠️ Lỗi apply context template: {str(e)}, fallback to standard")
        return f"{prompt}\n\nNội dung cần xử lý:\n{text}\n\nKết quả:"

def load_fine_tuned_models():
    """Load danh sách fine-tuned models từ registry"""
    try:
        registry_file = "fine_tuned_models.json"
        if os.path.exists(registry_file):
            with open(registry_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('fine_tuned_models', {})
        else:
            return {}
    except Exception as e:
        logger.warning(f"⚠️ Không thể load fine-tuned models: {str(e)}")
        return {}

def check_and_retry_failed_rows(df, result_column, model, column_names, prompt, is_multicolumn=False, max_retry_attempts=2):
    """
    Kiểm tra và xử lý lại các row bị lỗi trước khi hoàn thành ETL
    
    Args:
        df: DataFrame chứa dữ liệu
        result_column: Tên cột chứa kết quả AI
        model: AI model đã khởi tạo
        column_names: Danh sách tên cột (cho multicolumn) hoặc tên cột message
        prompt: Prompt template
        is_multicolumn: True nếu xử lý nhiều cột
        max_retry_attempts: Số lần retry tối đa
        
    Returns:
        dict: Thống kê kết quả retry
    """
    logger.info("🔍 Kiểm tra các row bị lỗi...")
    
    # Tìm tất cả error patterns
    error_patterns = [
        "Lỗi",           # Vietnamese error
        "Batch error",   # Async batch error  
        "failed after",  # Batch failed after X attempts
        "HTTP 429",      # Rate limit error
        "Timeout",       # Timeout error
        "Connection error",  # Connection error
        "Rate limit",    # Rate limit variations
        "Request failed", # Request failed
        "API error",     # API error
        "Exception",     # General exception
        "Error:",        # Error with colon
        "❌"            # Error emoji
    ]
    
    # Tạo mask cho errors
    error_mask = pd.Series([False] * len(df))
    for pattern in error_patterns:
        pattern_mask = df[result_column].astype(str).str.contains(pattern, case=False, na=False)
        error_mask = error_mask | pattern_mask
    
    # Tìm các row có lỗi
    error_indices = df[error_mask].index.tolist()
    
    if not error_indices:
        logger.info("✅ Không có row nào bị lỗi!")
        return {
            'total_errors': 0,
            'retry_attempted': 0,
            'retry_success': 0,
            'retry_failed': 0
        }
    
    logger.info(f"🔥 Tìm thấy {len(error_indices)} row bị lỗi, bắt đầu retry...")
    
    retry_stats = {
        'total_errors': len(error_indices),
        'retry_attempted': 0,
        'retry_success': 0,
        'retry_failed': 0
    }
    
    # Progress bar cho retry process
    with tqdm(error_indices, desc="🔄 Retry Failed Rows", ncols=100) as pbar:
        for idx in pbar:
            try:
                retry_stats['retry_attempted'] += 1
                
                # Lấy thông tin row hiện tại
                current_error = str(df.at[idx, result_column])
                pbar.set_description(f"🔄 Retry row {idx} (Error: {current_error[:30]}...)")
                
                success = False
                
                # Thử retry với delay tăng dần
                for attempt in range(max_retry_attempts):
                    try:
                        # Delay tăng dần: 2s, 5s, 10s
                        if attempt > 0:
                            delay = 2 * (attempt + 1)
                            logger.debug(f"⏳ Waiting {delay}s before retry attempt {attempt + 1}")
                            time.sleep(delay)
                        
                        # Xử lý lại row này
                        if is_multicolumn:
                            # Multi-column mode
                            row_data = {}
                            has_data = False
                            
                            for col in column_names:
                                if col in df.columns:
                                    value = df.at[idx, col]
                                    row_data[col] = value
                                    if pd.notna(value) and str(value).strip():
                                        has_data = True
                            
                            if not has_data:
                                df.at[idx, result_column] = "Không có dữ liệu"
                                success = True
                                break
                            
                            # Xử lý với AI multi-column
                            result = process_multicolumn_with_ai(
                                model,
                                row_data,
                                column_names,
                                prompt,
                                max_retries=1  # Giảm retry trong function để tránh nested retry
                            )
                        else:
                            # Single column mode
                            message_col = column_names if isinstance(column_names, str) else column_names[0]
                            
                            if pd.isna(df.at[idx, message_col]):
                                df.at[idx, result_column] = "Không có dữ liệu"
                                success = True
                                break
                            
                            # Lấy và làm sạch text
                            text = clean_text(df.at[idx, message_col])
                            
                            if not text:
                                df.at[idx, result_column] = "Không có dữ liệu"
                                success = True
                                break
                            
                            # Xử lý với AI
                            result = process_text_with_ai(
                                model, 
                                text, 
                                prompt,
                                max_retries=1  # Giảm retry trong function
                            )
                        
                        # Kiểm tra kết quả có phải error không
                        if result and not any(pattern.lower() in str(result).lower() for pattern in error_patterns):
                            df.at[idx, result_column] = result
                            success = True
                            logger.debug(f"✅ Row {idx} retry success")
                            break
                        else:
                            logger.debug(f"⚠️ Row {idx} retry attempt {attempt + 1} still failed: {str(result)[:50]}...")
                            
                    except Exception as retry_error:
                        logger.debug(f"❌ Row {idx} retry attempt {attempt + 1} exception: {str(retry_error)}")
                        continue
                
                # Cập nhật stats
                if success:
                    retry_stats['retry_success'] += 1
                    pbar.set_description(f"✅ Row {idx} retry success")
                else:
                    retry_stats['retry_failed'] += 1
                    # Giữ nguyên error message cũ hoặc cập nhật
                    df.at[idx, result_column] = f"Retry failed: {current_error}"
                    pbar.set_description(f"❌ Row {idx} retry failed")
                
            except Exception as e:
                retry_stats['retry_failed'] += 1
                logger.error(f"💥 Unexpected error retrying row {idx}: {str(e)}")
                pbar.set_description(f"💥 Row {idx} unexpected error")
    
    # Log kết quả retry
    logger.info(f"🎯 Retry Results:")
    logger.info(f"   - Total errors found: {retry_stats['total_errors']}")
    logger.info(f"   - Retry attempted: {retry_stats['retry_attempted']}")
    logger.info(f"   - Retry success: {retry_stats['retry_success']}")
    logger.info(f"   - Retry failed: {retry_stats['retry_failed']}")
    
    if retry_stats['retry_success'] > 0:
        success_rate = (retry_stats['retry_success'] / retry_stats['retry_attempted']) * 100
        logger.info(f"   - Retry success rate: {success_rate:.1f}%")
    
    return retry_stats 

# ===========================
# JSON PARSING UTILITIES  
# ===========================

def parse_json_response(response_text, schema=None, repair_malformed=True):
    """
    Parse JSON response từ AI với error handling và repair
    
    Args:
        response_text: Text response từ AI
        schema: JSON schema để validate (optional)
        repair_malformed: Có thử sửa JSON lỗi format không
        
    Returns:
        dict: Parsed JSON object hoặc None nếu thất bại
    """
    # Sử dụng config nếu không override
    if repair_malformed is None:
        repair_malformed = JSON_REPAIR_MALFORMED
    
    try:
        # Step 1: Clean và extract JSON từ response
        json_text = extract_json_from_text(response_text)
        
        if not json_text:
            logger.debug("❌ Không tìm thấy JSON trong response")
            return None
        
        # Step 2: Parse JSON
        try:
            json_obj = json.loads(json_text)
            logger.debug("✅ JSON parse thành công")
        except json.JSONDecodeError as e:
            if repair_malformed:
                logger.debug(f"⚠️ JSON parse failed, thử repair: {str(e)}")
                repaired_json = repair_json_text(json_text)
                if repaired_json:
                    json_obj = json.loads(repaired_json)
                    logger.debug("✅ JSON repair và parse thành công")
                else:
                    logger.debug("❌ JSON repair thất bại")
                    return None
            else:
                logger.debug(f"❌ JSON parse failed: {str(e)}")
                return None
        
        # Step 3: Validate schema nếu được bật và có schema phù hợp
        if JSON_VALIDATE_SCHEMA and schema and _is_compatible_schema(json_obj, schema):
            try:
                validate(instance=json_obj, schema=schema)
                logger.debug("✅ JSON schema validation thành công")
            except ValidationError as e:
                logger.debug(f"⚠️ JSON schema validation failed: {str(e)}")
                # Không return None, vẫn trả về object để caller có thể xử lý
        else:
            logger.debug("🔄 Bỏ qua schema validation (schema không tương thích hoặc không có)")
        
        return json_obj
        
    except Exception as e:
        logger.debug(f"❌ JSON parsing error: {str(e)}")
        return None

def _is_compatible_schema(json_obj, schema):
    """
    Kiểm tra xem JSON object có tương thích với schema không
    
    Args:
        json_obj: JSON object đã parse
        schema: JSON schema
        
    Returns:
        bool: True nếu tương thích
    """
    if not isinstance(json_obj, dict) or not isinstance(schema, dict):
        return False
    
    # Schema cũ (category, product, service, tag, note_1)
    old_schema_fields = ['category', 'product', 'service', 'tag', 'note_1']
    has_old_fields = any(field in json_obj for field in old_schema_fields)
    
    # Schema mới (Masterise Group - AI_SACTHAI, AI_TOPICS, etc.)
    new_schema_fields = ['AI_SACTHAI', 'AI_TOPICS', 'AI_DOITUONG', 'AI_THELOAINOIDUNG']
    has_new_fields = any(field in json_obj for field in new_schema_fields)
    
    # Chỉ validate nếu có old schema fields
    return has_old_fields and not has_new_fields

def extract_json_from_text(text):
    """
    Trích xuất JSON object từ text response của AI
    Hỗ trợ xử lý JSON được bọc trong markdown code blocks
    
    Args:
        text: Text chứa JSON (có thể có text khác xung quanh)
        
    Returns:
        str: JSON text được extract hoặc None
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.strip()
    
    # Method 1: Xử lý markdown code blocks (```json ... ```)
    markdown_patterns = [
        r'```json\s*(.*?)\s*```',  # ```json ... ```
        r'```\s*(.*?)\s*```',      # ``` ... ```
        r'`(.*?)`',                # ` ... `
    ]
    
    for pattern in markdown_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            # Lấy match đầu tiên và tìm JSON object trong đó
            for match in matches:
                json_text = _extract_json_object(match.strip())
                if json_text:
                    logger.debug("✅ Tìm thấy JSON trong markdown code block")
                    return json_text
    
    # Method 2: Tìm JSON object trực tiếp trong text
    json_text = _extract_json_object(text)
    if json_text:
        logger.debug("✅ Tìm thấy JSON object trực tiếp")
        return json_text
    
    logger.debug("❌ Không tìm thấy JSON object trong text")
    return None

def _extract_json_object(text):
    """
    Helper function để trích xuất JSON object từ text
    
    Args:
        text: Text chứa JSON
        
    Returns:
        str: JSON text được extract hoặc None
    """
    if not text:
        return None
    
    # Tìm vị trí { đầu tiên
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    # Tìm matching } bằng cách đếm brackets
    bracket_count = 0
    end_idx = -1
    
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            bracket_count += 1
        elif text[i] == '}':
            bracket_count -= 1
            if bracket_count == 0:
                end_idx = i
                break
    
    if end_idx == -1:
        # Không tìm thấy matching }, thử lấy từ start đến cuối
        json_text = text[start_idx:]
    else:
        json_text = text[start_idx:end_idx + 1]
    
    return json_text.strip()

def repair_json_text(json_text):
    """
    Thử sửa JSON text bị lỗi format phổ biến
    
    Args:
        json_text: JSON text có thể bị lỗi
        
    Returns:
        str: JSON text đã được sửa hoặc None nếu không sửa được
    """
    if not json_text:
        return None
    
    try:
        original = json_text.strip()
        repaired = original
        
        # Repair 1: Thêm quotes cho keys không có quotes
        repaired = re.sub(r'(\w+):', r'"\1":', repaired)
        
        # Repair 2: Thay single quotes thành double quotes
        repaired = repaired.replace("'", '"')
        
        # Repair 3: Sửa trailing comma
        repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
        
        # Repair 4: Sửa null không đúng format
        repaired = re.sub(r'\bnull\b', 'null', repaired, flags=re.IGNORECASE)
        repaired = re.sub(r'\bNone\b', 'null', repaired)
        repaired = re.sub(r'\bundefined\b', 'null', repaired)
        
        # Repair 5: Sửa boolean values
        repaired = re.sub(r'\bTrue\b', 'true', repaired)
        repaired = re.sub(r'\bFalse\b', 'false', repaired)
        
        # Repair 6: Remove comments
        repaired = re.sub(r'//.*$', '', repaired, flags=re.MULTILINE)
        repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
        
        # Test nếu repair thành công
        try:
            json.loads(repaired)
            logger.debug("✅ JSON repair thành công")
            return repaired
        except json.JSONDecodeError:
            logger.debug("❌ JSON repair thất bại")
            return None
            
    except Exception as e:
        logger.debug(f"❌ JSON repair error: {str(e)}")
        return None

def convert_json_to_text_format(json_obj, delimiter="|"):
    """
    Chuyển JSON object thành text format cũ để tương thích
    Hoặc trả về JSON string nếu schema phức tạp
    
    Args:
        json_obj: JSON object đã parse
        delimiter: Delimiter để nối các field
        
    Returns:
        str: Text format theo pattern cũ hoặc JSON string
    """
    if not isinstance(json_obj, dict):
        return str(json_obj)
    
    try:
        # Kiểm tra nếu là schema cũ (category, product, service, tag, note_1)
        old_schema_fields = ['category', 'product', 'service', 'tag', 'note_1']
        if all(field in json_obj for field in old_schema_fields):
            # Theo format: Category|Sản phẩm|Service|Tag|Note 1
            fields = [
                json_obj.get('category') or 'null',
                json_obj.get('product') or 'null', 
                json_obj.get('service') or 'null',
                json_obj.get('tag') or 'null',
                json_obj.get('note_1') or 'null'
            ]
            return delimiter.join(fields)
        
        # Nếu là schema phức tạp (như Masterise Group), trả về JSON string
        else:
            import json
            return json.dumps(json_obj, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.debug(f"❌ JSON to text conversion error: {str(e)}")
        # Fallback: trả về JSON string
        try:
            import json
            return json.dumps(json_obj, ensure_ascii=False)
        except:
            return str(json_obj)

def parse_response_with_fallback(response_text, use_json=False, schema=None):
    """
    Parse response với fallback từ JSON về text parsing
    
    Args:
        response_text: Response text từ AI
        use_json: Có thử parse JSON trước không
        schema: JSON schema để validate
        
    Returns:
        str: Parsed result text
    """
    if not response_text:
        return "Không có response"
    
    # Nếu được bật JSON mode, thử parse JSON trước
    if use_json:
        json_obj = parse_json_response(response_text, schema)
        
        if json_obj:
            # Convert JSON về text format để tương thích với code cũ
            return convert_json_to_text_format(json_obj)
        elif JSON_PARSE_FALLBACK_TO_TEXT:
            logger.debug("🔄 JSON parse thất bại, fallback về text parsing")
            # Cải thiện: Thử extract text có ý nghĩa từ response
            cleaned_text = _extract_meaningful_text(response_text)
            if cleaned_text and cleaned_text != "```json":
                return cleaned_text
            else:
                # Nếu vẫn không có text có ý nghĩa, trả về thông báo lỗi rõ ràng
                return f"JSON parse failed - Response: {response_text[:100]}..."
        else:
            return f"JSON parse failed: {response_text[:200]}..."
    
    # Fallback về text parsing cũ hoặc trả về raw text
    return response_text.strip()

def _extract_meaningful_text(response_text):
    """
    Trích xuất text có ý nghĩa từ response, loại bỏ markdown và formatting
    
    Args:
        response_text: Response text từ AI
        
    Returns:
        str: Text có ý nghĩa hoặc None
    """
    if not response_text:
        return None
    
    text = response_text.strip()
    
    # Loại bỏ markdown code blocks
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*', '', text)
    text = re.sub(r'`\s*', '', text)
    
    # Loại bỏ các pattern không có ý nghĩa
    text = re.sub(r'^\s*```json\s*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*```\s*$', '', text)
    
    # Loại bỏ các dòng trống và whitespace thừa
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)
    
    # Nếu text quá ngắn hoặc chỉ chứa formatting, return None
    if len(text) < 5 or text.lower() in ['json', '```', '```json']:
        return None
    
    return text

# Test utility
def test_json_parsing():
    """Test JSON parsing utilities"""
    test_cases = [
        # Valid JSON
        '{"category": "Bánh", "product": "ChocoPie", "service": "Review", "tag": "Chất lượng tốt", "note_1": "2"}',
        
        # JSON với text xung quanh  
        'Kết quả phân tích:\n{"category": "Nước khoáng", "product": null, "service": "Sản phẩm", "tag": "Hỏi thảo luận", "note_1": "2"}\nHết.',
        
        # JSON với lỗi format
        "{category: 'Bánh', product: 'ChocoPie', service: 'Review', tag: 'Chất lượng tốt', note_1: '2'}",
        
        # Text không có JSON
        "Category|ChocoPie|Review|Chất lượng tốt|2"
    ]
    
    print("🧪 Testing JSON parsing utilities...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case[:50]}...")
        
        # Test JSON parsing
        json_result = parse_json_response(test_case)
        print(f"  JSON parse: {json_result}")
        
        # Test với fallback
        fallback_result = parse_response_with_fallback(test_case, use_json=True)
        print(f"  With fallback: {fallback_result}")

if __name__ == "__main__":
    test_json_parsing() 