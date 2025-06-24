

import sys
import os
from pathlib import Path

# Thêm thư mục hiện tại vào Python path
sys.path.append(str(Path(__file__).parent))

from user_interface import get_user_input, display_help
from processor import run_processor

def main():
    """Hàm main chính của chương trình"""
    
    
    # Kiểm tra arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help', 'help']:
            display_help()
            return
        elif sys.argv[1] in ['-v', '--version', 'version']:
            print("AI ETL DATA v1.0.0")
            return
    
    try:
        # Thu thập thông tin từ người dùng
        config = get_user_input()
        
        if config is None:
            print("❌ Đã hủy thao tác")
            return
        
        # Chạy processor
        success = run_processor(config)
        
        if success:
            print("\n🎊 CHÚC MỪNG! Xử lý dữ liệu hoàn tất thành công!")
            print("🔍 Kiểm tra file kết quả trong cùng thư mục với file input.")
        else:
            print("\n😞 Xử lý dữ liệu không hoàn tất!")
            print("💡 Kiểm tra log file để biết chi tiết lỗi.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Chương trình bị dừng bởi người dùng")
    except Exception as e:
        print(f"\n💥 Lỗi không mong muốn: {str(e)}")
        print("💡 Vui lòng kiểm tra lại thông tin đầu vào và thử lại")

def check_dependencies():
    """Kiểm tra các dependencies cần thiết"""
    required_packages = {
        'pandas': 'pandas',
        'google-generativeai': 'google.generativeai', 
        'tqdm': 'tqdm',
        'openpyxl': 'openpyxl'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Thiếu các package sau:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Cài đặt bằng lệnh:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

if __name__ == "__main__":
    # Kiểm tra dependencies trước khi chạy
    if not check_dependencies():
        sys.exit(1)
    
    # Chạy chương trình chính
    main() 