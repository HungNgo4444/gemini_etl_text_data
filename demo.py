#!/usr/bin/env python3
"""
Demo script để tạo dữ liệu mẫu và test AI ETL Data
"""

import pandas as pd
import os
from pathlib import Path

def create_sample_data():
    """Tạo file dữ liệu mẫu để test"""
    
    sample_data = {
        'ID': range(1, 21),
        'MESSAGE': [
            "Khách hàng hỏi về giá sản phẩm và thời gian giao hàng",
            "Phản ánh về chất lượng dịch vụ không tốt, nhân viên thiếu nhiệt tình",
            "Yêu cầu tư vấn về các gói bảo hiểm phù hợp với gia đình",
            "Khiếu nại về việc giao hàng chậm và sản phẩm bị hỏng",
            "Cảm ơn công ty về dịch vụ tốt và sản phẩm chất lượng",
            "Hỏi về chính sách đổi trả hàng và bảo hành sản phẩm",
            "Đề xuất cải thiện giao diện website để dễ sử dụng hơn",
            "Phàn nàn về thái độ phục vụ của nhân viên bán hàng",
            "Yêu cầu hỗ trợ kỹ thuật để khắc phục lỗi ứng dụng",
            "Khen ngợi về chương trình khuyến mãi hấp dẫn",
            "Hỏi về thủ tục mở tài khoản và các giấy tờ cần thiết",
            "Báo cáo lỗi hệ thống không thể đăng nhập vào tài khoản",
            "Yêu cầu tư vấn đầu tư và các gói tiết kiệm",
            "Phản hồi tích cực về chất lượng sản phẩm mới",
            "Khiếu nại về phí dịch vụ cao so với đối thủ cạnh tranh",
            "Đề nghị thêm phương thức thanh toán online",
            "Hỏi về lịch làm việc và thông tin liên hệ chi nhánh",
            "Phản ánh về môi trường cửa hàng không thoải mái",
            "Yêu cầu hủy dịch vụ và hoàn tiền phí đã đóng",
            "Cảm ơn về sự hỗ trợ nhiệt tình của đội ngũ CSKH"
        ],
        'DATE': pd.date_range('2025-01-01', periods=20, freq='D'),
        'CATEGORY': ['INQUIRY', 'COMPLAINT', 'INQUIRY', 'COMPLAINT', 'PRAISE', 
                    'INQUIRY', 'SUGGESTION', 'COMPLAINT', 'SUPPORT', 'PRAISE',
                    'INQUIRY', 'ISSUE', 'INQUIRY', 'FEEDBACK', 'COMPLAINT',
                    'SUGGESTION', 'INQUIRY', 'COMPLAINT', 'REQUEST', 'PRAISE']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Lưu file Excel
    excel_file = Path(__file__).parent / "sample_data.xlsx"
    df.to_excel(excel_file, index=False)
    print(f"✅ Đã tạo file mẫu Excel: {excel_file}")
    
    # Lưu file CSV
    csv_file = Path(__file__).parent / "sample_data.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"✅ Đã tạo file mẫu CSV: {csv_file}")
    
    return excel_file, csv_file

def run_demo():
    """Chạy demo với dữ liệu mẫu"""
    print("🚀 AI ETL DATA - DEMO")
    print("="*50)
    
    # Tạo dữ liệu mẫu
    excel_file, csv_file = create_sample_data()
    
    print(f"\n📊 Dữ liệu mẫu đã được tạo:")
    print(f"   📁 Excel: {excel_file.name}")
    print(f"   📁 CSV: {csv_file.name}")
    
    print(f"\n💡 Để test chương trình:")
    print(f"   1. Chạy: python main.py")
    print(f"   2. Nhập API Key Gemini của bạn")
    print(f"   3. Chọn model (khuyến nghị: gemma-3-27b-it)")
    print(f"   4. Nhập đường dẫn file: {excel_file}")
    print(f"   5. Chọn cột: MESSAGE")
    print(f"   6. Chọn template prompt phù hợp")
    
    print(f"\n📋 Ví dụ prompt templates để test:")
    print(f"   • Tóm tắt: 'Hãy tóm tắt nội dung khách hàng trong 1-2 câu ngắn gọn'")
    print(f"   • Phân loại: 'Phân loại nội dung này thành: Khiếu nại, Hỏi đáp, Khen ngợi, Đề xuất'")
    print(f"   • Cảm xúc: 'Phân tích cảm xúc của khách hàng: Tích cực, Tiêu cực, hay Trung tính'")
    
    # Preview dữ liệu
    df = pd.read_excel(excel_file)
    print(f"\n🔍 Preview dữ liệu (5 dòng đầu):")
    print("-" * 80)
    for i in range(min(5, len(df))):
        print(f"ID {df.iloc[i]['ID']}: {df.iloc[i]['MESSAGE'][:80]}...")
    
    print(f"\n🎯 Kết quả mong đợi:")
    print(f"   - File output: sample_data_ai_result_<timestamp>.xlsx")
    print(f"   - Cột mới: AI_RESULT với kết quả xử lý AI")
    print(f"   - Thời gian ước tính: ~1-2 phút (20 records)")

if __name__ == "__main__":
    run_demo() 