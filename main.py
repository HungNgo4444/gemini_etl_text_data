#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI ETL Data - Công cụ xử lý dữ liệu text với Gemini AI và OpenAI
Hỗ trợ Batch Processing và Parallel Processing cho tốc độ tối ưu
"""

import sys
import os
from pathlib import Path

# Thêm thư mục hiện tại vào Python path
sys.path.append(str(Path(__file__).parent))

from user_interface import get_user_input, show_main_menu
from processor import run_processor

def main():
    """Hàm chính của chương trình"""
    try:
        # Hiển thị menu chính
        show_main_menu()
        
        # Thu thập thông tin từ user
        config = get_user_input()
        
        if not config:
            print("❌ Không thể thu thập thông tin cấu hình!")
            return False
        
        # Chạy processor
        success = run_processor(config)
        
        if success:
            print("\n🎉 Hoàn thành thành công!")
            print("📧 Liên hệ: AI ETL Data Team")
            print("🔗 GitHub: https://github.com/HungNgo4444/gemini_etl_text_data")
        else:
            print("\n❌ Quá trình xử lý gặp lỗi!")
            
        return success
        
    except KeyboardInterrupt:
        print("\n⏹️ Người dùng dừng chương trình.")
        return False
    except Exception as e:
        print(f"\n💥 Lỗi không mong muốn: {str(e)}")
        return False

def check_dependencies():
    """Kiểm tra các dependencies cần thiết"""
    required_packages = {
        'pandas': 'pandas',
        'google-generativeai': 'google.generativeai', 
        'openai': 'openai',
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