import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path

from utils import (
    initialize_gemini,
    load_data,
    save_data,
    load_checkpoint,
    save_checkpoint,
    process_text_with_ai,
    generate_output_filename,
    generate_checkpoint_filename,
    clean_text,
    get_processing_stats,
    print_progress_summary,
    logger
)

from config import (
    AI_RESULT_COLUMN,
    CHECKPOINT_INTERVAL,
    PROGRESS_REPORT_INTERVAL
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
        print("🤖 Đang khởi tạo Gemini model...")
        self.model = initialize_gemini(
            self.config['api_key'], 
            self.config['model_name']
        )
        
        if not self.model:
            print("❌ Không thể khởi tạo Gemini model!")
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
        """Xử lý dữ liệu chính"""
        print(f"\n🚀 BẮT ĐẦU XỬ LÝ DỮ LIỆU")
        print("="*50)
        
        stats = get_processing_stats(self.df, AI_RESULT_COLUMN)
        self.stats['start_time'] = time.time()
        
        print(f"🎯 Sẽ xử lý {stats['remaining']} records")
        print(f"✍️ Prompt: {self.config['prompt'][:100]}...")
        print(f"⏱️ Ước tính thời gian: ~{stats['remaining'] * 3 / 3600:.1f} giờ")
        print("-" * 50)
        
        try:
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
                
                # Bỏ qua nếu không có dữ liệu
                if pd.isna(self.df.at[idx, self.config['message_column']]):
                    continue
                
                try:
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
    
    def save_results(self):
        """Lưu kết quả cuối cùng"""
        print(f"\n💾 LƯU KẾT QUẢ")
        print("="*50)
        
        success = save_data(self.df, self.output_file)
        
        if success:
            print(f"✅ Đã lưu kết quả: {Path(self.output_file).name}")
            
            # Thống kê cuối cùng
            final_stats = get_processing_stats(self.df, AI_RESULT_COLUMN)
            print(f"📊 Thống kê cuối cùng:")
            print(f"   - Tổng records: {final_stats['total']}")
            print(f"   - Đã xử lý: {final_stats['processed']}")
            print(f"   - Lỗi: {final_stats['errors']}")
            print(f"   - Tỷ lệ thành công: {(final_stats['processed']-final_stats['errors'])/final_stats['total']*100:.1f}%")
            
            # Xóa checkpoint nếu hoàn thành
            if (self.config['use_checkpoint'] and 
                self.checkpoint_file and 
                final_stats['remaining'] == 0):
                try:
                    Path(self.checkpoint_file).unlink(missing_ok=True)
                    print("🗑️ Đã xóa checkpoint file")
                except:
                    pass
            
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