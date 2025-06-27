#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script để test OpenAI integration
"""

import sys
import os
from pathlib import Path

# Thêm thư mục hiện tại vào Python path
sys.path.append(str(Path(__file__).parent))

from utils import initialize_ai_model, process_text_with_ai

def demo_openai():
    """Demo OpenAI integration"""
    print("🚀 Demo OpenAI Integration")
    print("="*50)
    
    # Test data
    test_prompt = "Hãy tóm tắt nội dung sau một cách ngắn gọn:"
    test_text = "Đây là một văn bản mẫu để test chức năng tóm tắt của AI. Văn bản này chứa nhiều thông tin quan trọng và cần được tóm tắt lại một cách ngắn gọn và rõ ràng."
    
    # Test API key (sẽ được input từ user)
    api_key = input("Nhập OpenAI API Key để test: ").strip()
    if not api_key:
        print("❌ Cần API Key để test!")
        return
    
    # Test model
    model_name = "gpt-3.5-turbo"
    
    print(f"🤖 Testing model: {model_name}")
    print(f"📝 Test prompt: {test_prompt}")
    print(f"📄 Test text: {test_text}")
    print("-" * 50)
    
    try:
        # Khởi tạo model
        print("🔧 Khởi tạo OpenAI model...")
        model = initialize_ai_model('openai', api_key, model_name)
        
        if not model:
            print("❌ Không thể khởi tạo OpenAI model!")
            return
        
        print("✅ Khởi tạo thành công!")
        
        # Test xử lý text
        print("🚀 Testing text processing...")
        result = process_text_with_ai(model, test_text, test_prompt)
        
        print("\n📊 KẾT QUẢ:")
        print("="*50)
        print(result)
        print("="*50)
        print("✅ Test thành công!")
        
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")

if __name__ == "__main__":
    demo_openai() 