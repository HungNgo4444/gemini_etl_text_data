import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path

from utils import (
    initialize_ai_model,
    load_data,
    save_data,
    load_checkpoint,
    save_checkpoint,
    process_text_with_ai,
    process_multicolumn_with_ai,
    process_batch_with_ai,
    process_multicolumn_batch_with_ai,
    process_data_parallel,
    process_data_with_async,
    process_data_batch_only,
    generate_output_filename,
    generate_checkpoint_filename,
    clean_text,
    get_processing_stats,
    print_progress_summary,
    logger,
    check_and_retry_failed_rows
)

from config import (
    AI_RESULT_COLUMN,
    CHECKPOINT_INTERVAL,
    PROGRESS_REPORT_INTERVAL,
    ENABLE_BATCH_PROCESSING,
    BATCH_SIZE,
    ENABLE_PARALLEL_PROCESSING,
    ENABLE_ASYNC_PROCESSING,
    MAX_CONCURRENT_THREADS,
    THREAD_BATCH_SIZE,
    MAX_CONCURRENT_REQUESTS,
    ASYNC_RATE_LIMIT_RPM,
    ASYNC_CHUNK_SIZE,
    ENABLE_ERROR_RETRY,
    ERROR_RETRY_MAX_ATTEMPTS
)

class AIDataProcessor:
    """Lớp xử lý dữ liệu với AI"""
    
    def __init__(self, config):
        """Khởi tạo processor với cấu hình từ user"""
        self.config = config
        self.model = None
        self.df = None
        self.stats = {
            'processed': 0,
            'errors': 0,
            'start_time': None
        }
        
        # Thiết lập file paths
        self.input_file = config['input_file']
        self.output_file = generate_output_filename(self.input_file)
        self.checkpoint_file = generate_checkpoint_filename(self.input_file) if config['use_checkpoint'] else None
        
    def initialize(self):
        """Khởi tạo các thành phần cần thiết"""
        print("\n🔧 KHỞI TẠO PROCESSOR")
        print("="*50)
        
        # 1. Khởi tạo AI model
        api_provider_name = "Gemini" if self.config['api_provider'] == 'gemini' else "OpenAI"
        print(f"🤖 Đang khởi tạo {api_provider_name} model...")
        fine_tuned_model_info = self.config.get('fine_tuned_model_info')
        self.model = initialize_ai_model(
            self.config['api_provider'],
            self.config['api_key'], 
            self.config['model_name'],
            fine_tuned_model_info
        )
        
        if not self.model:
            print(f"❌ Không thể khởi tạo {api_provider_name} model!")
            return False
        
        # 2. Load dữ liệu
        print("📊 Đang load dữ liệu...")
        
        # Thử load checkpoint trước
        if self.config['use_checkpoint'] and self.checkpoint_file:
            self.df = load_checkpoint(self.checkpoint_file)
            
        # Nếu không có checkpoint, load file gốc
        if self.df is None:
            self.df = load_data(self.input_file)
            
        if self.df is None:
            print("❌ Không thể load dữ liệu!")
            return False
        
        # 3. Kiểm tra cột cần xử lý
        if self.config.get('multi_column_mode', False):
            # Kiểm tra tất cả cột đã chọn
            missing_columns = []
            for col in self.config['selected_columns']:
                if col not in self.df.columns:
                    missing_columns.append(col)
            
            if missing_columns:
                print(f"❌ Không tìm thấy các cột: {missing_columns}")
                return False
        else:
            # Kiểm tra cột đơn
            if self.config['message_column'] not in self.df.columns:
                print(f"❌ Không tìm thấy cột '{self.config['message_column']}' trong file!")
                return False
        
        # 4. Thêm cột kết quả nếu chưa có
        if AI_RESULT_COLUMN not in self.df.columns:
            self.df[AI_RESULT_COLUMN] = ""
        
        # 5. Thống kê dữ liệu
        stats = get_processing_stats(self.df, AI_RESULT_COLUMN)
        
        print(f"✅ Khởi tạo thành công!")
        print(f"📈 Tổng số records: {stats['total']}")
        print(f"✅ Đã xử lý: {stats['processed']}")
        print(f"⏳ Còn lại: {stats['remaining']}")
        print(f"❌ Lỗi: {stats['errors']}")
        
        if stats['remaining'] == 0:
            print("🎉 Tất cả dữ liệu đã được xử lý!")
            return False
        
        return True
    
    def process_data(self):
        """Xử lý dữ liệu chính với Parallel + Batch Processing"""
        print(f"\n🚀 BẮT ĐẦU XỬ LÝ DỮ LIỆU")
        print("="*50)
        
        stats = get_processing_stats(self.df, AI_RESULT_COLUMN)
        self.stats['start_time'] = time.time()
        
        # Xác định chế độ xử lý - ưu tiên async processing
        is_async_mode = ENABLE_ASYNC_PROCESSING and stats['remaining'] > 10
        is_parallel_mode = ENABLE_PARALLEL_PROCESSING and stats['remaining'] > MAX_CONCURRENT_THREADS and not is_async_mode
        is_batch_mode = ENABLE_BATCH_PROCESSING and stats['remaining'] > 1 and not is_async_mode and not is_parallel_mode
        
        print(f"🎯 Sẽ xử lý {stats['remaining']} records")
        
        # Hiển thị chế độ xử lý
        if is_async_mode:
            print(f"🚀 Chế độ: Async Processing (SEMAPHORE + RATE LIMITER)")
            print(f"⚡ Max concurrent: {MAX_CONCURRENT_REQUESTS}")
            print(f"📊 Rate limit: {ASYNC_RATE_LIMIT_RPM} RPM")
            print(f"📦 Chunk size: {ASYNC_CHUNK_SIZE}")
            estimated_time_saving = "50-100x faster"
        elif is_parallel_mode:
            print(f"🚀 Chế độ: Parallel + Batch Processing (LEGACY)")
            print(f"🧵 Threads: {MAX_CONCURRENT_THREADS}")
            print(f"📦 Thread batch size: {THREAD_BATCH_SIZE}")
            estimated_time_saving = "15-30x faster"
        elif is_batch_mode:
            print(f"📦 Chế độ: Batch Processing")
            print(f"📦 Batch size: {BATCH_SIZE}")
            estimated_time_saving = "5-10x faster"
        else:
            print(f"⚡ Chế độ: Single Processing")
            estimated_time_saving = "baseline"
        
        print(f"⏱️ Ước tính cải thiện tốc độ: {estimated_time_saving}")
        print(f"✍️ Prompt: {self.config['prompt'][:100]}...")
        print("-" * 50)
        
        try:
            if is_async_mode:
                return self._process_with_parallel()  # Sử dụng lại function này nhưng đã được cập nhật cho async
            elif is_parallel_mode:
                return self._process_with_parallel()
            elif is_batch_mode:
                return self._process_with_batch()
            else:
                return self._process_without_batch()
                
        except KeyboardInterrupt:
            print("\n⏹️ Người dùng dừng chương trình...")
            if self.config['use_checkpoint'] and self.checkpoint_file:
                save_checkpoint(self.df, self.checkpoint_file)
                print("💾 Đã lưu checkpoint. Chạy lại để tiếp tục.")
            return False
            
        except Exception as e:
            print(f"\n💥 Lỗi không mong muốn: {str(e)}")
            logger.error(f"Lỗi xử lý dữ liệu: {str(e)}")
            if self.config['use_checkpoint'] and self.checkpoint_file:
                save_checkpoint(self.df, self.checkpoint_file)
                print("💾 Đã lưu checkpoint. Chạy lại để tiếp tục.")
            return False
    
    def _process_with_parallel(self):
        """Xử lý dữ liệu với Async Processing (thay thế parallel processing)"""
        print("🚀 Khởi động Async Processing...")
        
        # Chuẩn bị dữ liệu cho async processing
        unprocessed_data = []
        unprocessed_indices = []
        
        for idx in self.df.index:
            if (pd.isna(self.df.at[idx, AI_RESULT_COLUMN]) or 
                self.df.at[idx, AI_RESULT_COLUMN] == ""):
                
                if self.config.get('multi_column_mode', False):
                    # Multi-column mode
                    row_data = {}
                    has_data = False
                    
                    for col in self.config['selected_columns']:
                        value = self.df.at[idx, col]
                        row_data[col] = value
                        if pd.notna(value) and str(value).strip():
                            has_data = True
                    
                    if has_data:
                        unprocessed_data.append(row_data)
                        unprocessed_indices.append(idx)
                    else:
                        self.df.at[idx, AI_RESULT_COLUMN] = "Không có dữ liệu"
                else:
                    # Single column mode
                    if pd.notna(self.df.at[idx, self.config['message_column']]):
                        text = clean_text(self.df.at[idx, self.config['message_column']])
                        if text:
                            unprocessed_data.append(text)
                            unprocessed_indices.append(idx)
                        else:
                            self.df.at[idx, AI_RESULT_COLUMN] = "Không có dữ liệu"
                    else:
                        self.df.at[idx, AI_RESULT_COLUMN] = "Không có dữ liệu"
        
        if not unprocessed_data:
            print("✅ Không có dữ liệu cần xử lý")
            return True
        
        print(f"📊 Chuẩn bị xử lý {len(unprocessed_data)} items với async processing")
        
        # 🔥 TẠO CHECKPOINT CALLBACK FUNCTION
        def async_checkpoint_callback(results_so_far, chunk_completed, total_chunks):
            """Callback để lưu checkpoint trong quá trình async processing"""
            try:
                # Cập nhật kết quả vào DataFrame
                for i, result in enumerate(results_so_far):
                    if i < len(unprocessed_indices):
                        idx = unprocessed_indices[i]
                        self.df.at[idx, AI_RESULT_COLUMN] = result
                        
                # Lưu checkpoint
                if self.config['use_checkpoint'] and self.checkpoint_file:
                    save_checkpoint(self.df, self.checkpoint_file)
                    logger.info(f"💾 Async checkpoint saved: chunk {chunk_completed}/{total_chunks}")
                    
            except Exception as e:
                logger.error(f"❌ Error in async checkpoint callback: {e}")
        
        # Gọi async processing với checkpoint support
        try:
            results = process_data_with_async(
                self.model,
                unprocessed_data,
                self.config.get('selected_columns', []),
                self.config['prompt'],
                self.config.get('multi_column_mode', False),
                checkpoint_callback=async_checkpoint_callback,
                checkpoint_interval=CHECKPOINT_INTERVAL,
                use_json=self.config.get('use_json_output', False)
            )
            
            # Lưu kết quả vào DataFrame
            for i, result in enumerate(results):
                if i < len(unprocessed_indices):
                    idx = unprocessed_indices[i]
                    self.df.at[idx, AI_RESULT_COLUMN] = result
                    self.stats['processed'] += 1
            
            # Lưu checkpoint cuối cùng
            if self.config['use_checkpoint'] and self.checkpoint_file:
                save_checkpoint(self.df, self.checkpoint_file)
            
            print(f"✅ Async processing hoàn thành: {len(results)} kết quả")
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi async processing: {str(e)}")
            print(f"⚠️ Async processing thất bại, fallback về batch processing")
            
            # Fallback về batch processing
            return self._process_with_batch_fallback(unprocessed_data, unprocessed_indices)

    def _process_with_batch_fallback(self, unprocessed_data, unprocessed_indices):
        """Fallback processing khi parallel thất bại"""
        try:
            print("📦 Chuyển sang Batch Processing fallback...")
            
            results = process_data_batch_only(
                self.model,
                unprocessed_data,
                self.config.get('selected_columns', []),
                self.config['prompt'],
                self.config.get('multi_column_mode', False)
            )
            
            # Lưu kết quả vào DataFrame
            for i, result in enumerate(results):
                if i < len(unprocessed_indices):
                    idx = unprocessed_indices[i]
                    self.df.at[idx, AI_RESULT_COLUMN] = result
                    self.stats['processed'] += 1
            
            # Lưu checkpoint
            if self.config['use_checkpoint'] and self.checkpoint_file:
                save_checkpoint(self.df, self.checkpoint_file)
            
            print(f"✅ Batch processing fallback hoàn thành: {len(results)} kết quả")
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch processing fallback cũng thất bại: {str(e)}")
            print("⚠️ Fallback về single processing...")
            
            # Final fallback về single processing
            return self._process_single_fallback(unprocessed_data, unprocessed_indices)
    
    def _process_single_fallback(self, unprocessed_data, unprocessed_indices):
        """Final fallback về single processing"""
        try:
            print("⚡ Single processing fallback...")
            
            for i, data in enumerate(unprocessed_data):
                if i >= len(unprocessed_indices):
                    break
                    
                idx = unprocessed_indices[i]
                
                try:
                    if self.config.get('multi_column_mode', False):
                        result = process_multicolumn_with_ai(
                            self.model,
                            data,
                            self.config['selected_columns'],
                            self.config['prompt']
                        )
                    else:
                        result = process_text_with_ai(
                            self.model,
                            str(data),
                            self.config['prompt']
                        )
                    
                    self.df.at[idx, AI_RESULT_COLUMN] = result
                    self.stats['processed'] += 1
                    
                except Exception as single_error:
                    self.stats['errors'] += 1
                    self.df.at[idx, AI_RESULT_COLUMN] = f"Lỗi xử lý: {str(single_error)}"
                    logger.error(f"Lỗi single processing tại {idx}: {str(single_error)}")
            
            # Lưu checkpoint
            if self.config['use_checkpoint'] and self.checkpoint_file:
                save_checkpoint(self.df, self.checkpoint_file)
            
            print(f"✅ Single processing fallback hoàn thành: {self.stats['processed']} records")
            return True
            
        except Exception as e:
            logger.error(f"❌ Single processing fallback thất bại: {str(e)}")
            return False

    def _process_with_batch(self):
        """Xử lý dữ liệu với batch processing"""
        # Lấy danh sách các record cần xử lý
        unprocessed_indices = []
        for idx in self.df.index:
            if (pd.isna(self.df.at[idx, AI_RESULT_COLUMN]) or 
                self.df.at[idx, AI_RESULT_COLUMN] == ""):
                unprocessed_indices.append(idx)
        
        total_batches = (len(unprocessed_indices) + BATCH_SIZE - 1) // BATCH_SIZE
        
        # Tạo progress bar cho batches
        progress_bar = tqdm(
            range(total_batches),
            desc="Xử lý batches",
            ncols=100
        )
        
        for batch_idx in progress_bar:
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(unprocessed_indices))
            batch_indices = unprocessed_indices[start_idx:end_idx]
            
            try:
                if self.config.get('multi_column_mode', False):
                    # Batch processing cho multi-column
                    batch_rows = []
                    valid_indices = []
                    
                    for idx in batch_indices:
                        row_data = {}
                        has_data = False
                        
                        for col in self.config['selected_columns']:
                            value = self.df.at[idx, col]
                            row_data[col] = value
                            if pd.notna(value) and str(value).strip():
                                has_data = True
                        
                        if has_data:
                            batch_rows.append(row_data)
                            valid_indices.append(idx)
                        else:
                            self.df.at[idx, AI_RESULT_COLUMN] = "Không có dữ liệu"
                    
                    if batch_rows:
                        # Xử lý batch multi-column
                        results = process_multicolumn_batch_with_ai(
                            self.model,
                            batch_rows,
                            self.config['selected_columns'],
                            self.config['prompt']
                        )
                        
                        # Lưu kết quả
                        for i, result in enumerate(results):
                            if i < len(valid_indices):
                                self.df.at[valid_indices[i], AI_RESULT_COLUMN] = result
                                self.stats['processed'] += 1
                
                else:
                    # Batch processing cho single column
                    batch_data = []
                    valid_indices = []
                    
                    for idx in batch_indices:
                        if pd.notna(self.df.at[idx, self.config['message_column']]):
                            text = clean_text(self.df.at[idx, self.config['message_column']])
                            if text:
                                batch_data.append(text)
                                valid_indices.append(idx)
                            else:
                                self.df.at[idx, AI_RESULT_COLUMN] = "Không có dữ liệu"
                        else:
                            self.df.at[idx, AI_RESULT_COLUMN] = "Không có dữ liệu"
                    
                    if batch_data:
                        # Xử lý batch single column
                        results = process_batch_with_ai(
                            self.model,
                            batch_data,
                            self.config['prompt']
                        )
                        
                        # Lưu kết quả
                        for i, result in enumerate(results):
                            if i < len(valid_indices):
                                self.df.at[valid_indices[i], AI_RESULT_COLUMN] = result
                                self.stats['processed'] += 1
                
                # Cập nhật progress bar
                progress_bar.set_description(f"Đã xử lý: {self.stats['processed']} records")
                
                # Lưu checkpoint định kỳ
                if (self.config['use_checkpoint'] and 
                    self.checkpoint_file and
                    (batch_idx + 1) % (CHECKPOINT_INTERVAL // BATCH_SIZE) == 0):
                    
                    save_checkpoint(self.df, self.checkpoint_file)
                
                # Báo cáo tiến trình định kỳ
                if (batch_idx + 1) % (PROGRESS_REPORT_INTERVAL // BATCH_SIZE) == 0:
                    print_progress_summary(
                        self.stats['processed'], 
                        len(self.df), 
                        self.stats['start_time'], 
                        self.stats['errors']
                    )
                
            except Exception as e:
                logger.error(f"❌ Lỗi xử lý batch {batch_idx + 1}: {str(e)}")
                # Fallback: xử lý từng record trong batch này
                for idx in batch_indices:
                    try:
                        if self.config.get('multi_column_mode', False):
                            row_data = {}
                            has_data = False
                            for col in self.config['selected_columns']:
                                value = self.df.at[idx, col]
                                row_data[col] = value
                                if pd.notna(value) and str(value).strip():
                                    has_data = True
                            
                            if has_data:
                                result = process_multicolumn_with_ai(
                                    self.model,
                                    row_data,
                                    self.config['selected_columns'],
                                    self.config['prompt']
                                )
                                self.df.at[idx, AI_RESULT_COLUMN] = result
                                self.stats['processed'] += 1
                            else:
                                self.df.at[idx, AI_RESULT_COLUMN] = "Không có dữ liệu"
                        else:
                            if pd.notna(self.df.at[idx, self.config['message_column']]):
                                text = clean_text(self.df.at[idx, self.config['message_column']])
                                if text:
                                    result = process_text_with_ai(
                                        self.model, 
                                        text, 
                                        self.config['prompt']
                                    )
                                    self.df.at[idx, AI_RESULT_COLUMN] = result
                                    self.stats['processed'] += 1
                                else:
                                    self.df.at[idx, AI_RESULT_COLUMN] = "Không có dữ liệu"
                            else:
                                self.df.at[idx, AI_RESULT_COLUMN] = "Không có dữ liệu"
                    except Exception as single_error:
                        self.stats['errors'] += 1
                        self.df.at[idx, AI_RESULT_COLUMN] = f"Lỗi xử lý: {str(single_error)}"
                        logger.error(f"Lỗi tại row {idx}: {str(single_error)}")
        
        progress_bar.close()
        
        # Báo cáo cuối cùng
        print_progress_summary(
            self.stats['processed'], 
            len(self.df), 
            self.stats['start_time'], 
            self.stats['errors']
        )
        
        return True

    def _process_without_batch(self):
        """Xử lý dữ liệu không dùng batch processing (legacy mode)"""
        stats = get_processing_stats(self.df, AI_RESULT_COLUMN)
        
        # Tạo progress bar
        progress_bar = tqdm(
            self.df.index, 
            desc="Xử lý dữ liệu", 
            ncols=100,
            initial=stats['processed']
        )
        
        for idx in progress_bar:
            # Bỏ qua nếu đã xử lý
            if (pd.notna(self.df.at[idx, AI_RESULT_COLUMN]) and 
                self.df.at[idx, AI_RESULT_COLUMN] != ""):
                continue
            
            try:
                # Xử lý theo chế độ
                if self.config.get('multi_column_mode', False):
                    # Chế độ nhiều cột
                    row_data = {}
                    has_data = False
                    
                    for col in self.config['selected_columns']:
                        value = self.df.at[idx, col]
                        row_data[col] = value
                        if pd.notna(value) and str(value).strip():
                            has_data = True
                    
                    if not has_data:
                        self.df.at[idx, AI_RESULT_COLUMN] = "Không có dữ liệu"
                        continue
                    
                    # Xử lý với AI multi-column
                    result = process_multicolumn_with_ai(
                        self.model,
                        row_data,
                        self.config['selected_columns'],
                        self.config['prompt']
                    )
                else:
                    # Chế độ cột đơn
                    if pd.isna(self.df.at[idx, self.config['message_column']]):
                        continue
                    
                    # Lấy và làm sạch text
                    text = clean_text(self.df.at[idx, self.config['message_column']])
                    
                    if not text:
                        self.df.at[idx, AI_RESULT_COLUMN] = "Không có dữ liệu"
                        continue
                    
                    # Xử lý với AI
                    result = process_text_with_ai(
                        self.model, 
                        text, 
                        self.config['prompt']
                    )
                
                # Lưu kết quả
                self.df.at[idx, AI_RESULT_COLUMN] = result
                self.stats['processed'] += 1
                
                # Cập nhật progress bar
                progress_bar.set_description(f"Đã xử lý: {self.stats['processed']}")
                
            except Exception as e:
                self.stats['errors'] += 1
                self.df.at[idx, AI_RESULT_COLUMN] = f"Lỗi xử lý: {str(e)}"
                logger.error(f"Lỗi tại row {idx}: {str(e)}")
            
            # Lưu checkpoint định kỳ
            if (self.config['use_checkpoint'] and 
                self.checkpoint_file and
                self.stats['processed'] % CHECKPOINT_INTERVAL == 0 and 
                self.stats['processed'] > 0):
                
                save_checkpoint(self.df, self.checkpoint_file)
            
            # Báo cáo tiến trình định kỳ
            if (self.stats['processed'] % PROGRESS_REPORT_INTERVAL == 0 and 
                self.stats['processed'] > 0):
                
                print_progress_summary(
                    self.stats['processed'], 
                    stats['total'], 
                    self.stats['start_time'], 
                    self.stats['errors']
                )
        
        progress_bar.close()
        
        # Báo cáo cuối cùng
        print_progress_summary(
            self.stats['processed'], 
            stats['total'], 
            self.stats['start_time'], 
            self.stats['errors']
        )
        
        return True
    
    def save_results(self):
        """Lưu kết quả cuối cùng"""
        print(f"\n💾 LƯU KẾT QUẢ")
        print("="*50)
        
        # BƯỚC MỚI: Kiểm tra và retry các row bị lỗi trước khi lưu
        if ENABLE_ERROR_RETRY:
            print("\n🔍 KIỂM TRA VÀ XỬ LÝ LẠI CÁC ROW BỊ LỖI...")
            print("-"*50)
            
            try:
                # Xác định thông số cho retry
                if self.config.get('multi_column_mode', False):
                    column_names = self.config['selected_columns']
                    is_multicolumn = True
                else:
                    column_names = self.config['message_column']
                    is_multicolumn = False
                
                # Chạy retry failed rows với config
                retry_stats = check_and_retry_failed_rows(
                    df=self.df,
                    result_column=AI_RESULT_COLUMN,
                    model=self.model,
                    column_names=column_names,
                    prompt=self.config['prompt'],
                    is_multicolumn=is_multicolumn,
                    max_retry_attempts=ERROR_RETRY_MAX_ATTEMPTS
                )
                
                # Hiển thị kết quả retry
                if retry_stats['total_errors'] > 0:
                    print(f"\n📊 KẾT QUẢ RETRY:")
                    print(f"   🔥 Tổng lỗi tìm thấy: {retry_stats['total_errors']}")
                    print(f"   🔄 Đã thử retry: {retry_stats['retry_attempted']}")
                    print(f"   ✅ Retry thành công: {retry_stats['retry_success']}")
                    print(f"   ❌ Retry thất bại: {retry_stats['retry_failed']}")
                    
                    if retry_stats['retry_success'] > 0:
                        success_rate = (retry_stats['retry_success'] / retry_stats['retry_attempted']) * 100
                        print(f"   📈 Tỷ lệ retry thành công: {success_rate:.1f}%")
                        
                        # Cập nhật stats tổng
                        self.stats['processed'] += retry_stats['retry_success']
                        if retry_stats['retry_failed'] > retry_stats['retry_success']:
                            self.stats['errors'] += (retry_stats['retry_failed'] - retry_stats['retry_success'])
                        else:
                            self.stats['errors'] = max(0, self.stats['errors'] - retry_stats['retry_success'])
                else:
                    print("✅ Không có row nào bị lỗi cần retry!")
                    
            except Exception as retry_error:
                print(f"⚠️ Lỗi trong quá trình retry: {str(retry_error)}")
                print("Tiếp tục lưu file với dữ liệu hiện tại...")
        
        print("\n💾 Lưu file kết quả...")
        success = save_data(self.df, self.output_file)
        
        if success:
            print(f"✅ Đã lưu kết quả: {Path(self.output_file).name}")
            
            # Thống kê cuối cùng (sau retry)
            final_stats = get_processing_stats(self.df, AI_RESULT_COLUMN)
            print(f"\n📊 THỐNG KÊ CUỐI CÙNG:")
            print(f"   - Tổng records: {final_stats['total']}")
            print(f"   - Đã xử lý thành công: {final_stats['processed']}")
            print(f"   - Còn lỗi: {final_stats['errors']}")
            print(f"   - Chưa xử lý: {final_stats['remaining']}")
            
            # Tính tỷ lệ thành công thực tế
            actual_success = final_stats['processed']
            total_records = final_stats['total']
            success_rate = (actual_success / total_records * 100) if total_records > 0 else 0
            print(f"   - Tỷ lệ thành công: {success_rate:.1f}%")
            
            # Xóa checkpoint nếu hoàn thành tốt
            remaining_errors = final_stats['errors'] + final_stats['remaining']
            if (self.config['use_checkpoint'] and 
                self.checkpoint_file and 
                remaining_errors == 0):  # Không còn lỗi và chưa xử lý
                try:
                    Path(self.checkpoint_file).unlink(missing_ok=True)
                    print("🗑️ Đã xóa checkpoint file (hoàn thành 100%)")
                except:
                    pass
            elif remaining_errors > 0:
                print(f"💾 Giữ checkpoint file (còn {remaining_errors} records chưa hoàn thành)")
            
            return True
        else:
            print("❌ Lỗi lưu file kết quả!")
            return False
    
    def run(self):
        """Chạy toàn bộ quá trình xử lý"""
        print("\n🎯 BẮT ĐẦU QUÁ TRÌNH XỬ LÝ AI ETL DATA")
        print("="*60)
        
        # 1. Khởi tạo
        if not self.initialize():
            return False
        
        # 2. Xử lý dữ liệu
        if not self.process_data():
            return False
        
        # 3. Lưu kết quả
        if not self.save_results():
            return False
        
        # 4. Tổng kết
        elapsed_hours = (time.time() - self.stats['start_time']) / 3600
        print(f"\n🎉 HOÀN THÀNH!")
        print("="*60)
        print(f"⏱️ Tổng thời gian: {elapsed_hours:.2f} giờ")
        print(f"⚡ Tốc độ trung bình: {self.stats['processed']/elapsed_hours:.0f} records/giờ")
        print(f"📁 File kết quả: {Path(self.output_file).name}")
        print(f"📍 Vị trí: {Path(self.output_file).parent}")
        
        return True

def run_processor(config):
    """Hàm tiện ích để chạy processor"""
    processor = AIDataProcessor(config)
    return processor.run() 