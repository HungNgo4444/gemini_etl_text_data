import os
import sys
from pathlib import Path
from config import AVAILABLE_GEMINI_MODELS, AVAILABLE_OPENAI_MODELS, DEFAULT_PROMPT_TEMPLATES
from utils import validate_file_path, validate_prompt, detect_message_column, load_data, load_fine_tuned_models

# Kiểm tra OpenAI availability
def check_openai_available():
    """Kiểm tra OpenAI availability động"""
    try:
        from utils import check_openai_availability
        available, _ = check_openai_availability()
        return available
    except ImportError:
        return False

OPENAI_AVAILABLE = check_openai_available()

def load_prompt_from_file():
    """Load prompt từ file text"""
    while True:
        file_path = input("\nNhập đường dẫn file prompt (.txt): ").strip()
        
        # Loại bỏ dấu ngoặc kép nếu có
        file_path = file_path.strip('"').strip("'")
        
        # Kiểm tra file tồn tại
        if not os.path.exists(file_path):
            print(f"❌ File không tồn tại: {file_path}")
            retry = input("Thử lại? (y/n): ").strip().lower()
            if retry not in ['y', 'yes', 'có']:
                return None
            continue
        
        # Kiểm tra phần mở rộng
        if not file_path.lower().endswith('.txt'):
            print(f"❌ Chỉ hỗ trợ file .txt")
            retry = input("Thử lại? (y/n): ").strip().lower()
            if retry not in ['y', 'yes', 'có']:
                return None
            continue
        
        try:
            # Thử đọc file với các encoding khác nhau
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read().strip()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print(f"❌ Không thể đọc file với các encoding thông dụng")
                retry = input("Thử lại? (y/n): ").strip().lower()
                if retry not in ['y', 'yes', 'có']:
                    return None
                continue
            
            if not content:
                print(f"❌ File rỗng hoặc không có nội dung")
                retry = input("Thử lại? (y/n): ").strip().lower()
                if retry not in ['y', 'yes', 'có']:
                    return None
                continue
            
            # Validate prompt
            is_valid, message = validate_prompt(content)
            if not is_valid:
                print(f"❌ {message}")
                retry = input("Thử lại? (y/n): ").strip().lower()
                if retry not in ['y', 'yes', 'có']:
                    return None
                continue
            
            print(f"✅ Đã đọc thành công file: {Path(file_path).name}")
            print(f"📝 Độ dài prompt: {len(content)} ký tự")
            
            return content
            
        except Exception as e:
            print(f"❌ Lỗi khi đọc file: {str(e)}")
            retry = input("Thử lại? (y/n): ").strip().lower()
            if retry not in ['y', 'yes', 'có']:
                return None

def get_user_input():
    """Thu thập thông tin từ người dùng"""
    print("🚀 AI ETL DATA - Xử lý dữ liệu text với AI")
    print("="*60)
    
    user_config = {}
    
    # 1. Chọn API Provider
    print("\n🔧 BƯỚC 1: Chọn API Provider")
    print("-" * 40)
    print("1. Gemini AI (Google)")
    
    # Kiểm tra lại động
    openai_available = check_openai_available()
    available_options = ["1"]
    if openai_available:
        print("2. OpenAI (ChatGPT)")
        available_options.append("2")
    else:
        print("2. OpenAI (ChatGPT) - ❌ Không có sẵn (cần cài đặt: pip install openai>=1.0.0)")
    
    max_option = len(available_options)
    prompt_text = f"\nChọn API provider ({'/'.join(available_options)}): " if max_option > 1 else "\nNhấn Enter để tiếp tục với Gemini AI: "
    
    while True:
        if max_option == 1:
            input(prompt_text)
            user_config['api_provider'] = 'gemini'
            print("✅ Đã chọn Gemini AI")
            break
        else:
            provider_choice = input(prompt_text).strip()
            if provider_choice == "1":
                user_config['api_provider'] = 'gemini'
                print("✅ Đã chọn Gemini AI")
                break
            elif provider_choice == "2" and openai_available:
                user_config['api_provider'] = 'openai'
                print("✅ Đã chọn OpenAI")
                break
            else:
                if not openai_available and provider_choice == "2":
                    print("❌ OpenAI không có sẵn. Vui lòng cài đặt: pip install openai>=1.0.0")
                else:
                    print(f"❌ Vui lòng chọn {' hoặc '.join(available_options)}")
    
    # 2. Nhập API Key
    api_provider_name = "Gemini" if user_config['api_provider'] == 'gemini' else "OpenAI"
    print(f"\n📡 BƯỚC 2: Cấu hình API {api_provider_name}")
    print("-" * 40)
    
    while True:
        api_key = input(f"Nhập {api_provider_name} API Key: ").strip()
        if api_key:
            user_config['api_key'] = api_key
            print("✅ API Key đã được nhập")
            break
        print("❌ API Key không được để trống!")
    
    # 3. Chọn model
    print(f"\n🤖 BƯỚC 3: Chọn model AI")
    print("-" * 40)
    
    # Chọn models dựa trên provider
    if user_config['api_provider'] == 'gemini':
        # Load fine-tuned models cho Gemini
        fine_tuned_models = load_fine_tuned_models()
        all_models = []
        model_info = {}
        
        # Thêm standard Gemini models
        print("📋 GEMINI STANDARD MODELS:")
        for i, model in enumerate(AVAILABLE_GEMINI_MODELS, 1):
            print(f"  {i}. {model}")
            all_models.append(model)
            model_info[model] = None  # Standard model
        
        # Thêm fine-tuned models nếu có
        if fine_tuned_models:
            print("\n🎯 GEMINI FINE-TUNED MODELS:")
            for name, info in fine_tuned_models.items():
                index = len(all_models) + 1
                display_name = info.get('display_name', name)
                training_date = info.get('training_date', 'N/A')[:10]  # Only date part
                print(f"  {index}. {display_name} (Fine-tuned: {training_date})")
                all_models.append(name)
                model_info[name] = info
    else:
        # OpenAI models
        all_models = AVAILABLE_OPENAI_MODELS.copy()
        model_info = {model: None for model in all_models}
        fine_tuned_models = None
        
        print("📋 OPENAI MODELS:")
        for i, model in enumerate(AVAILABLE_OPENAI_MODELS, 1):
            print(f"  {i}. {model}")
    
    while True:
        try:
            choice = input(f"\nChọn model (1-{len(all_models)}) hoặc nhập tên model khác: ").strip()
            
            # Kiểm tra nếu là số
            if choice.isdigit():
                model_index = int(choice) - 1
                if 0 <= model_index < len(all_models):
                    selected_model = all_models[model_index]
                    user_config['model_name'] = selected_model
                    user_config['fine_tuned_model_info'] = model_info[selected_model]
                    
                    # Hiển thị info cho fine-tuned model
                    if user_config['api_provider'] == 'gemini' and model_info[selected_model]:
                        print(f"✅ Đã chọn fine-tuned Gemini model: {selected_model}")
                        print(f"📅 Training date: {model_info[selected_model].get('training_date', 'N/A')[:19]}")
                        if model_info[selected_model].get('requires_context'):
                            print(f"🎯 Model này sử dụng prompt context từ fine-tuning")
                    else:
                        api_name = "Gemini" if user_config['api_provider'] == 'gemini' else "OpenAI"
                        print(f"✅ Đã chọn {api_name} model: {selected_model}")
                    break
                else:
                    print(f"❌ Vui lòng chọn từ 1 đến {len(all_models)}")
            else:
                # Người dùng nhập tên model khác
                if choice:
                    user_config['model_name'] = choice
                    user_config['fine_tuned_model_info'] = None
                    api_name = "Gemini" if user_config['api_provider'] == 'gemini' else "OpenAI"
                    print(f"✅ Đã chọn custom {api_name} model: {choice}")
                    break
                print("❌ Tên model không được để trống!")
        except ValueError:
            print("❌ Vui lòng nhập số hợp lệ!")
    
    # 4. Nhập đường dẫn file input
    print("\n📁 BƯỚC 4: Chọn file dữ liệu")
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
    
    # 5. Chọn cột cần xử lý
    print("\n📊 BƯỚC 5: Chọn cột dữ liệu")
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
        
        print("\n🎯 Tùy chọn lựa chọn cột:")
        print("  1️⃣  Chọn 1 cột duy nhất")
        print("  2️⃣  Chọn nhiều cột để ghép lại")
        
        while True:
            mode_choice = input("\nChọn chế độ (1 hoặc 2): ").strip()
            
            if mode_choice == "1":
                # Chế độ chọn 1 cột duy nhất
                while True:
                    choice = input(f"\nChọn cột cần xử lý (1-{len(df.columns)}) hoặc nhập tên cột: ").strip()
                    
                    if choice.isdigit():
                        col_index = int(choice) - 1
                        if 0 <= col_index < len(df.columns):
                            user_config['message_column'] = df.columns[col_index]
                            user_config['selected_columns'] = [df.columns[col_index]]
                            user_config['multi_column_mode'] = False
                            break
                        else:
                            print(f"❌ Vui lòng chọn từ 1 đến {len(df.columns)}")
                    else:
                        if choice in df.columns:
                            user_config['message_column'] = choice
                            user_config['selected_columns'] = [choice]
                            user_config['multi_column_mode'] = False
                            break
                        print("❌ Tên cột không tồn tại!")
                
                print(f"✅ Đã chọn cột: '{user_config['message_column']}'")
                break
                
            elif mode_choice == "2":
                # Chế độ chọn nhiều cột
                print("\n📝 Nhập các số cột cần xử lý, cách nhau bằng dấu phẩy")
                print("   Ví dụ: 1,3,5 hoặc 2,4,6,8")
                
                while True:
                    choices = input(f"\nNhập các số cột (1-{len(df.columns)}): ").strip()
                    
                    try:
                        # Parse input
                        column_indices = [int(x.strip()) - 1 for x in choices.split(',')]
                        
                        # Validate indices
                        invalid_indices = [i+1 for i in column_indices if i < 0 or i >= len(df.columns)]
                        if invalid_indices:
                            print(f"❌ Số cột không hợp lệ: {invalid_indices}. Vui lòng chọn từ 1 đến {len(df.columns)}")
                            continue
                        
                        # Get column names
                        selected_columns = [df.columns[i] for i in column_indices]
                        
                        # Set config
                        user_config['selected_columns'] = selected_columns
                        user_config['message_column'] = selected_columns[0]  # First column as primary
                        user_config['multi_column_mode'] = True
                        
                        print(f"\n✅ Đã chọn {len(selected_columns)} cột:")
                        for i, col in enumerate(selected_columns, 1):
                            print(f"  {i}. {col}")
                        break
                        
                    except ValueError:
                        print("❌ Format không đúng. Vui lòng nhập các số cách nhau bằng dấu phẩy (VD: 1,3,5)")
                break
            else:
                print("❌ Vui lòng chọn 1 hoặc 2")
    else:
        print("❌ Không thể load file để phân tích cột")
        return None
    
    # 6. Cấu hình checkpoint
    print("\n💾 BƯỚC 6: Cấu hình checkpoint")
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
    
    # 7. Chọn output format  
    print("\n📄 BƯỚC 7: Chọn định dạng output")
    print("-" * 40)
    
    while True:
        print("Chọn định dạng output:")
        print("  1. Text format (truyền thống) - Compatible với tất cả prompts")
        print("  2. JSON format (structured) - Chính xác hơn, dễ parse")
        
        format_choice = input("\nChọn format (1 hoặc 2) [1]: ").strip()
        
        if format_choice in ['', '1']:
            user_config['use_json_output'] = False
            print("✅ Sẽ sử dụng text format")
            break
        elif format_choice == '2':
            user_config['use_json_output'] = True
            print("✅ Sẽ sử dụng JSON format") 
            print("💡 JSON format yêu cầu prompt phù hợp hoặc sử dụng template json_classify")
            break
        else:
            print("❌ Vui lòng chọn 1 hoặc 2")

    # 8. Nhập prompt
    print("\n✍️ BƯỚC 8: Cấu hình prompt AI")
    print("-" * 40)
    
    # Hiển thị các template có sẵn, highlight JSON template nếu chọn JSON
    print("Các template prompt có sẵn:")
    template_keys = list(DEFAULT_PROMPT_TEMPLATES.keys())
    for i, key in enumerate(template_keys, 1):
        if key == 'json_classify' and user_config['use_json_output']:
            print(f"  {i}. {key}: {DEFAULT_PROMPT_TEMPLATES[key][:50]}... 🌟 RECOMMENDED cho JSON")
        else:
            print(f"  {i}. {key}: {DEFAULT_PROMPT_TEMPLATES[key][:50]}...")
    
    print(f"  {len(template_keys) + 1}. Đọc prompt từ file (.txt)")
    print(f"  {len(template_keys) + 2}. Tự nhập prompt")
    
    while True:
        try:
            choice = input(f"\nChọn template (1-{len(template_keys) + 2}): ").strip()
            
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
                    # Đọc prompt từ file
                    prompt_from_file = load_prompt_from_file()
                    if prompt_from_file:
                        user_config['prompt'] = prompt_from_file
                        print(f"✅ Đã load prompt từ file")
                        print(f"Prompt preview: {prompt_from_file[:200]}...")
                        break
                    else:
                        print("❌ Không thể load prompt từ file, vui lòng chọn lại")
                elif template_index == len(template_keys) + 1:
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
                    print(f"❌ Vui lòng chọn từ 1 đến {len(template_keys) + 2}")
            else:
                print("❌ Vui lòng nhập số hợp lệ!")
        except ValueError:
            print("❌ Vui lòng nhập số hợp lệ!")
    
    # 9. Tổng kết
    print("\n📋 BƯỚC 9: Tổng kết cấu hình")
    print("="*60)
    api_name = "Gemini" if user_config['api_provider'] == 'gemini' else "OpenAI"
    print(f"🔧 API Provider: {api_name}")
    print(f"🤖 Model: {user_config['model_name']}")
    print(f"📁 File input: {Path(user_config['input_file']).name}")
    
    if user_config.get('multi_column_mode', False):
        print(f"📊 Chế độ: Nhiều cột ({len(user_config['selected_columns'])} cột)")
        for i, col in enumerate(user_config['selected_columns'], 1):
            print(f"     {i}. {col}")
    else:
        print(f"📊 Cột xử lý: {user_config['message_column']}")
    
    print(f"💾 Checkpoint: {'Có' if user_config['use_checkpoint'] else 'Không'}")
    output_format = "JSON" if user_config.get('use_json_output', False) else "Text"
    print(f"📄 Output format: {output_format}")
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
Công cụ xử lý dữ liệu text bằng AI (Gemini và OpenAI), hỗ trợ đa dạng các tác vụ:
- Tóm tắt văn bản
- Phân loại nội dung  
- Phân tích cảm xúc
- Trích xuất từ khóa
- Dịch thuật
- Và nhiều tác vụ khác tùy chỉnh

🤖 API HỖ TRỢ:
- Gemini AI (Google): gemma-3-27b-it, gemini-2.0-flash, gemini-2.5-flash
- OpenAI: gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini

📁 ĐỊNH DẠNG FILE HỖ TRỢ:
- Excel (.xlsx, .xls)
- CSV (.csv)

⚙️ TÍNH NĂNG:
✅ Hỗ trợ cả Gemini AI và OpenAI API
✅ Tự động phát hiện cột dữ liệu
✅ Checkpoint để tiếp tục khi bị dừng
✅ Prompt templates có sẵn + đọc từ file .txt
✅ Báo cáo tiến trình real-time
✅ Xử lý lỗi thông minh
✅ Xuất kết quả cùng thư mục

🚀 CÁCH SỬ DỤNG:
1. Chạy: python main.py
2. Chọn API Provider (Gemini/OpenAI)
3. Nhập API Key
4. Chọn model AI
5. Chọn file dữ liệu (.xlsx/.csv)
6. Chọn cột cần xử lý
7. Cấu hình checkpoint
8. Chọn/nhập prompt (template có sẵn, file .txt, hoặc tự nhập)
9. Xác nhận và bắt đầu

📊 KẾT QUẢ:
- File kết quả: <tên_file>_ai_result_<timestamp>.<định_dạng>
- File checkpoint: <tên_file>_checkpoint.<định_dạng>
- Log file: ai_etl_data.log

💡 MẸO:
- Sử dụng checkpoint để xử lý file lớn
- Prompt càng cụ thể kết quả càng tốt
- Kiểm tra API Key có quyền truy cập model
""")

def show_main_menu():
    """Hiển thị menu chính"""
    print("\n🚀 AI ETL DATA - CÔNG CỤ XỬ LÝ DỮ LIỆU VỚI AI")
    print("="*60)
    print("🤖 HỖ TRỢ MULTI-API:")
    print("   🔥 Gemini AI: gemma-3-27b-it, gemini-2.0-flash, gemini-2.5-flash")
    print("   🧠 OpenAI: gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini")
    print("📋 TÍNH NĂNG CHÍNH:")
    print("   ✅ Xử lý file Excel/CSV với checkpoint thông minh")
    print("   ✅ Template prompt đa dạng + đọc prompt từ file")
    print("   ✅ Xử lý đa cột (multi-column) với AI analysis")
    print("   ✅ Batch Processing: Tăng tốc 5-10x")
    print("   ✅ Parallel Processing: Tăng tốc 15-30x")
    print("   ✅ Monitoring real-time với progress bar")
    print("   ✅ Error handling và auto-recovery")
    print("="*60)
    print("🎯 Bắt đầu thôi!")
    print()

if __name__ == "__main__":
    display_help() 