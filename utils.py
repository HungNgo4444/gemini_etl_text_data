import pandas as pd
import numpy as np
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import google.generativeai as genai
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import traceback

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
    MAX_CONCURRENT_THREADS,
    THREAD_BATCH_SIZE,
    RATE_LIMIT_DELAY,
    MAX_RETRIES_PER_THREAD,
    THREAD_TIMEOUT,
    CIRCUIT_BREAKER_THRESHOLD
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
                
                # Xử lý lỗi rate limit
                if "429" in error_msg or "quota" in error_msg.lower():
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
                
                # Xử lý lỗi rate limit
                if "429" in error_msg or "quota" in error_msg.lower():
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

def process_data_parallel(model, data, column_names, prompt, is_multicolumn=False):
    """Xử lý dữ liệu với parallel processing"""
    if not ENABLE_PARALLEL_PROCESSING:
        logger.info("🔄 Parallel processing bị tắt, chuyển về batch processing")
        return process_data_batch_only(model, data, column_names, prompt, is_multicolumn)
    
    try:
        total_items = len(data)
        logger.info(f"🚀 Bắt đầu parallel processing: {total_items} items với {MAX_CONCURRENT_THREADS} threads")
        
        # Chia dữ liệu thành các batches cho các threads
        thread_batches = []
        batch_size = THREAD_BATCH_SIZE
        
        for i in range(0, total_items, batch_size):
            batch = data[i:i+batch_size]
            thread_batches.append(batch)
        
        logger.info(f"📦 Đã chia thành {len(thread_batches)} batches, mỗi batch {batch_size} items")
        
        # Setup circuit breaker và progress tracking
        circuit_breaker = CircuitBreaker()
        progress_queue = Queue()
        all_results = [None] * len(thread_batches)
        
        # Progress tracking
        completed_threads = 0
        total_processed = 0
        start_time = time.time()
        
        # Chạy parallel processing với ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_THREADS) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, batch in enumerate(thread_batches):
                future = executor.submit(
                    process_parallel_batch_worker,
                    i, model, batch, column_names, prompt, is_multicolumn, circuit_breaker, progress_queue
                )
                future_to_index[future] = i
            
            # Collect results với progress tracking
            with tqdm(total=total_items, desc="🔄 Parallel Processing") as pbar:
                for future in as_completed(future_to_index, timeout=THREAD_TIMEOUT):
                    try:
                        index = future_to_index[future]
                        results = future.result()
                        all_results[index] = results
                        
                        # Update progress
                        batch_size_actual = len(thread_batches[index])
                        total_processed += batch_size_actual
                        completed_threads += 1
                        pbar.update(batch_size_actual)
                        
                        # Progress report
                        elapsed = time.time() - start_time
                        rate = total_processed / elapsed if elapsed > 0 else 0
                        logger.info(f"📈 Thread {index} hoàn thành - Tổng: {total_processed}/{total_items} ({rate:.2f} items/s)")
                        
                    except Exception as e:
                        index = future_to_index[future]
                        logger.error(f"❌ Thread {index} timeout hoặc lỗi: {str(e)}")
                        # Tạo kết quả lỗi cho batch này
                        batch_size_error = len(thread_batches[index])
                        all_results[index] = [f"Lỗi timeout: {str(e)}"] * batch_size_error
        
        # Flatten results
        final_results = []
        for batch_results in all_results:
            if batch_results:
                final_results.extend(batch_results)
        
        # Validation
        if len(final_results) != total_items:
            logger.warning(f"⚠️ Số lượng kết quả không khớp: {len(final_results)} vs {total_items}")
            # Pad with error messages if needed
            while len(final_results) < total_items:
                final_results.append("Lỗi: thiếu kết quả")
        
        # Final summary
        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        logger.info(f"🎉 Parallel processing hoàn thành: {total_processed}/{total_items} trong {elapsed:.1f}s ({rate:.2f} items/s)")
        
        return final_results
        
    except Exception as e:
        logger.error(f"❌ Lỗi parallel processing: {str(e)}")
        logger.info("🔄 Fallback về batch processing")
        return process_data_batch_only(model, data, column_names, prompt, is_multicolumn)

def process_data_batch_only(model, data, column_names, prompt, is_multicolumn=False):
    """Xử lý dữ liệu chỉ với batch processing (không parallel)"""
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
    processed = len(df[df[result_column].notna() & (df[result_column] != "")])
    errors = len(df[df[result_column].str.contains("Lỗi", na=False)])
    remaining = total - processed
    
    return {
        'total': total,
        'processed': processed,
        'remaining': remaining,
        'errors': errors
    } 