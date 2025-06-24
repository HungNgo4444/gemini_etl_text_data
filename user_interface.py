import os
import sys
from pathlib import Path
from config import AVAILABLE_MODELS, DEFAULT_PROMPT_TEMPLATES
from utils import validate_file_path, validate_prompt, detect_message_column, load_data

def get_user_input():
    """Thu thập thông tin từ người dùng"""
    print("🚀 AI ETL DATA - Xử lý dữ liệu text với Gemini AI")
    print("="*60)
    
    user_config = {}
    
    # 1. Nhập API Key
    print("\n📡 BƯỚC 1: Cấu hình API Gemini")
    print("-" * 40)
    
    while True:
        api_key = input("Nhập Gemini API Key: ").strip()
        if api_key:
            user_config['api_key'] = api_key
            print("✅ API Key đã được nhập")
            break
        print("❌ API Key không được để trống!")
    
    # 2. Chọn model
    print("\n🤖 BƯỚC 2: Chọn model AI")
    print("-" * 40)
    print("Các model có sẵn:")
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        print(f"  {i}. {model}")
    
    while True:
        try:
            choice = input(f"Chọn model (1-{len(AVAILABLE_MODELS)}) hoặc nhập tên model khác: ").strip()
            
            # Kiểm tra nếu là số
            if choice.isdigit():
                model_index = int(choice) - 1
                if 0 <= model_index < len(AVAILABLE_MODELS):
                    user_config['model_name'] = AVAILABLE_MODELS[model_index]
                    break
                else:
                    print(f"❌ Vui lòng chọn từ 1 đến {len(AVAILABLE_MODELS)}")
            else:
                # Người dùng nhập tên model khác
                if choice:
                    user_config['model_name'] = choice
                    break
                print("❌ Tên model không được để trống!")
        except ValueError:
            print("❌ Vui lòng nhập số hợp lệ!")
    
    print(f"✅ Đã chọn model: {user_config['model_name']}")
    
    # 3. Nhập đường dẫn file input
    print("\n📁 BƯỚC 3: Chọn file dữ liệu")
    print("-" * 40)
    
    while True:
        file_path = input("Nhập đường dẫn file cần xử lý (.xlsx, .csv): ").strip()
        
        # Loại bỏ dấu ngoặc kép nếu có
        file_path = file_path.strip('"').strip("'")
        
        is_valid, message = validate_file_path(file_path)
        if is_valid:
            user_config['input_file'] = file_path
            print(f"✅ {message}")
            break
        else:
            print(f"❌ {message}")
    
    # 4. Chọn cột cần xử lý
    print("\n📊 BƯỚC 4: Chọn cột dữ liệu")
    print("-" * 40)
    
    # Load file để hiển thị các cột
    print("🔍 Đang phân tích file...")
    df = load_data(user_config['input_file'])
    
    if df is not None:
        print(f"📈 File có {len(df)} dòng dữ liệu")
        print("Các cột có sẵn:")
        for i, col in enumerate(df.columns, 1):
            # Hiển thị preview dữ liệu
            sample_data = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "N/A"
            if len(str(sample_data)) > 50:
                sample_data = str(sample_data)[:50] + "..."
            print(f"  {i}. {col} (VD: {sample_data})")
        
        # Tự động detect cột message
        detected_col = detect_message_column(df)
        if detected_col:
            print(f"\n💡 Tự động phát hiện cột text: '{detected_col}'")
        
        while True:
            choice = input(f"\nChọn cột cần xử lý (1-{len(df.columns)}) hoặc nhập tên cột: ").strip()
            
            if choice.isdigit():
                col_index = int(choice) - 1
                if 0 <= col_index < len(df.columns):
                    user_config['message_column'] = df.columns[col_index]
                    break
                else:
                    print(f"❌ Vui lòng chọn từ 1 đến {len(df.columns)}")
            else:
                if choice in df.columns:
                    user_config['message_column'] = choice
                    break
                print("❌ Tên cột không tồn tại!")
        
        print(f"✅ Đã chọn cột: '{user_config['message_column']}'")
    else:
        print("❌ Không thể load file để phân tích cột")
        return None
    
    # 5. Cấu hình checkpoint
    print("\n💾 BƯỚC 5: Cấu hình checkpoint")
    print("-" * 40)
    
    while True:
        checkpoint_choice = input("Sử dụng checkpoint để có thể tiếp tục khi bị dừng? (y/n) [y]: ").strip().lower()
        if checkpoint_choice in ['', 'y', 'yes', 'có']:
            user_config['use_checkpoint'] = True
            print("✅ Sẽ sử dụng checkpoint")
            break
        elif checkpoint_choice in ['n', 'no', 'không']:
            user_config['use_checkpoint'] = False
            print("✅ Không sử dụng checkpoint")
            break
        else:
            print("❌ Vui lòng nhập 'y' hoặc 'n'")
    
    # 6. Nhập prompt
    print("\n✍️ BƯỚC 6: Cấu hình prompt AI")
    print("-" * 40)
    
    # Hiển thị các template có sẵn
    print("Các template prompt có sẵn:")
    template_keys = list(DEFAULT_PROMPT_TEMPLATES.keys())
    for i, key in enumerate(template_keys, 1):
        print(f"  {i}. {key}: {DEFAULT_PROMPT_TEMPLATES[key]}")
    
    print(f"  {len(template_keys) + 1}. Tự nhập prompt")
    
    while True:
        try:
            choice = input(f"\nChọn template (1-{len(template_keys) + 1}): ").strip()
            
            if choice.isdigit():
                template_index = int(choice) - 1
                if 0 <= template_index < len(template_keys):
                    # Sử dụng template có sẵn
                    template_key = template_keys[template_index]
                    user_config['prompt'] = DEFAULT_PROMPT_TEMPLATES[template_key]
                    print(f"✅ Đã chọn template: {template_key}")
                    print(f"Prompt: {user_config['prompt']}")
                    break
                elif template_index == len(template_keys):
                    # Tự nhập prompt
                    while True:
                        custom_prompt = input("\nNhập prompt tùy chỉnh: ").strip()
                        is_valid, message = validate_prompt(custom_prompt)
                        if is_valid:
                            user_config['prompt'] = custom_prompt
                            print(f"✅ {message}")
                            break
                        else:
                            print(f"❌ {message}")
                    break
                else:
                    print(f"❌ Vui lòng chọn từ 1 đến {len(template_keys) + 1}")
            else:
                print("❌ Vui lòng nhập số hợp lệ!")
        except ValueError:
            print("❌ Vui lòng nhập số hợp lệ!")
    
    # 7. Tổng kết
    print("\n📋 BƯỚC 7: Tổng kết cấu hình")
    print("="*60)
    print(f"🤖 Model: {user_config['model_name']}")
    print(f"📁 File input: {Path(user_config['input_file']).name}")
    print(f"📊 Cột xử lý: {user_config['message_column']}")
    print(f"💾 Checkpoint: {'Có' if user_config['use_checkpoint'] else 'Không'}")
    print(f"✍️ Prompt: {user_config['prompt'][:100]}...")
    
    while True:
        confirm = input("\nXác nhận bắt đầu xử lý? (y/n): ").strip().lower()
        if confirm in ['y', 'yes', 'có']:
            print("🚀 Bắt đầu xử lý...")
            return user_config
        elif confirm in ['n', 'no', 'không']:
            print("❌ Đã hủy")
            return None
        else:
            print("❌ Vui lòng nhập 'y' hoặc 'n'")

def display_help():
    """Hiển thị hướng dẫn sử dụng"""
    print("""
🚀 AI ETL DATA - Hướng dẫn sử dụng
================================

📋 MÔ TẢ:
Công cụ xử lý dữ liệu text bằng AI Gemini, hỗ trợ đa dạng các tác vụ:
- Tóm tắt văn bản
- Phân loại nội dung  
- Phân tích cảm xúc
- Trích xuất từ khóa
- Dịch thuật
- Và nhiều tác vụ khác tùy chỉnh

📁 ĐỊNH DẠNG FILE HỖ TRỢ:
- Excel (.xlsx, .xls)
- CSV (.csv)

⚙️ TÍNH NĂNG:
✅ Kết nối Gemini API với nhiều model
✅ Tự động phát hiện cột dữ liệu
✅ Checkpoint để tiếp tục khi bị dừng
✅ Prompt templates có sẵn
✅ Báo cáo tiến trình real-time
✅ Xử lý lỗi thông minh
✅ Xuất kết quả cùng thư mục

🚀 CÁCH SỬ DỤNG:
1. Chạy: python main.py
2. Nhập API Key Gemini
3. Chọn model AI
4. Chọn file dữ liệu (.xlsx/.csv)
5. Chọn cột cần xử lý
6. Cấu hình checkpoint
7. Chọn/nhập prompt
8. Xác nhận và bắt đầu

📊 KẾT QUẢ:
- File kết quả: <tên_file>_ai_result_<timestamp>.<định_dạng>
- File checkpoint: <tên_file>_checkpoint.<định_dạng>
- Log file: ai_etl_data.log

💡 MẸO:
- Sử dụng checkpoint để xử lý file lớn
- Prompt càng cụ thể kết quả càng tốt
- Kiểm tra API Key có quyền truy cập model
""")

if __name__ == "__main__":
    display_help() 