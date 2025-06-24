#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Parallel Processing - Minh họa tốc độ xử lý song song
"""

import pandas as pd
import time
from datetime import datetime

def create_demo_data():
    """Tạo dữ liệu demo cho parallel processing test"""
    
    # Dữ liệu mẫu với 50 records để test parallel processing
    demo_data = []
    
    products = ['iPhone 15', 'Samsung Galaxy S24', 'Xiaomi 14', 'OPPO Find X7', 'Vivo V30']
    sentiments = ['positive', 'negative', 'neutral', 'mixed']
    sources = ['Facebook', 'Instagram', 'TikTok', 'YouTube', 'Website']
    
    messages = [
        "Sản phẩm này thực sự tuyệt vời! Chất lượng vượt ngoài mong đợi của tôi.",
        "Giao hàng nhanh, đóng gói cẩn thận. Sẽ mua lại lần sau.",
        "Giá cả hợp lý, phù hợp với chất lượng sản phẩm.",
        "Dịch vụ khách hàng rất tốt, nhân viên tư vấn nhiệt tình.",
        "Sản phẩm có một số khuyết điểm nhỏ nhưng nhìn chung vẫn ổn.",
        "Không hài lòng với chất lượng, không giống như mô tả.",
        "Thời gian giao hàng hơi lâu so với cam kết ban đầu.",
        "Sản phẩm bình thường, không có gì đặc biệt để nói.",
        "Thiết kế đẹp mắt, màu sắc bắt mắt, phù hợp với xu hướng.",
        "Công nghệ hiện đại, tính năng đa dạng, dễ sử dụng.",
        "Bao bì sản phẩm rất đẹp, thích hợp làm quà tặng.",
        "Hiệu suất hoạt động ổn định, không gặp lỗi gì trong quá trình sử dụng.",
        "Hướng dẫn sử dụng chi tiết, dễ hiểu, phù hợp với mọi đối tượng.",
        "Giá thành có phần cao so với mặt bằng chung của thị trường.",
        "Chất liệu cao cấp, độ bền tốt, xứng đáng với số tiền bỏ ra.",
        "Màn hình hiển thị sắc nét, độ phân giải cao, trải nghiệm tốt.",
        "Pin có dung lượng lớn, thời gian sử dụng lâu dài.",
        "Camera chụp ảnh sắc nét, màu sắc tự nhiên, chất lượng cao.",
        "Âm thanh stereo sống động, bass mạnh mẽ, trung thực.",
        "Kết nối wifi ổn định, tốc độ internet nhanh chóng.",
    ]
    
    for i in range(50):
        demo_data.append({
            'ID': f'MSG_{i+1:03d}',
            'MESSAGE': messages[i % len(messages)],
            'PRODUCT': products[i % len(products)],
            'SOURCE': sources[i % len(sources)],
            'DATE': f'2025-01-{(i % 30) + 1:02d}',
            'PRIORITY': ['High', 'Medium', 'Low'][i % 3]
        })
    
    df = pd.DataFrame(demo_data)
    filename = 'parallel_demo_data.xlsx'
    df.to_excel(filename, index=False)
    
    print(f"📊 DEMO DATA CHO PARALLEL PROCESSING")
    print("="*50)
    print(f"✅ Đã tạo file: {filename}")
    print(f"📈 Số records: {len(df)}")
    print(f"📋 Các cột:")
    for i, col in enumerate(df.columns, 1):
        sample_value = df[col].iloc[0]
        print(f"  {i}. {col} (VD: {sample_value})")
    
    print(f"\n💡 Hướng dẫn test Parallel Processing:")
    print(f"   1. Chạy chương trình chính: python main.py")
    print(f"   2. Chọn file: {filename}")
    print(f"   3. Chọn cột MESSAGE (cột 2)")
    print(f"   4. Chương trình sẽ tự động detect Parallel Processing")
    print(f"   5. Quan sát tốc độ xử lý với progress bar và logs")
    
    print(f"\n⚡ Performance Expected:")
    print(f"   - Single: ~2.5 phút (50 records × 3s)")
    print(f"   - Batch: ~30 giây (10 batches × 3s)")  
    print(f"   - Parallel: ~15 giây (5 batches song song)")
    
    print(f"\n🛠️ Test Configuration:")
    print(f"   - MAX_CONCURRENT_THREADS = 2")
    print(f"   - THREAD_BATCH_SIZE = 5")
    print(f"   - Sẽ có 2 threads, mỗi thread xử lý 5 records/batch")
    print(f"   - Tổng: 5 batches song song thay vì 10 batches tuần tự")

if __name__ == "__main__":
    create_demo_data() 