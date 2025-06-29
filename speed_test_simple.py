#!/usr/bin/env python3
"""
Speed test đơn giản cho async vs batch processing
"""

import time
import asyncio
from utils import process_data_with_async, process_data_batch_only, initialize_ai_model
from config import *

def create_test_data(size=20):
    """Tạo test data"""
    data = []
    for i in range(size):
        data.append({
            'MESSAGE': f'Test message {i+1}: Đây là nội dung test để đánh giá tốc độ xử lý.',
            'ID': i+1
        })
    return data

async def test_async_processing():
    """Test async processing"""
    print("🔧 ASYNC PROCESSING TEST")
    print("=" * 40)
    
    # Setup
    api_provider = "openai"  # Thay đổi theo nhu cầu
    api_key = "your-api-key"  # Thay đổi
    model_name = "gpt-4o-mini"
    test_size = 10
    
    prompt = "Phân tích cảm xúc: TÍCH CỰC, TIÊU CỰC hoặc TRUNG TÍNH"
    
    # Tạo data và model
    test_data = create_test_data(test_size)
    model_obj = initialize_ai_model(api_provider, api_key, model_name)
    
    print(f"📊 Test size: {test_size} items")
    print(f"📊 ASYNC_BATCH_SIZE: {ASYNC_BATCH_SIZE}")
    print(f"📊 MAX_CONCURRENT_REQUESTS: {MAX_CONCURRENT_REQUESTS}")
    
    # Test async
    print(f"\n🚀 Testing ASYNC processing...")
    start_time = time.time()
    
    try:
        async_results = await process_data_with_async(
            model=model_obj,
            data=test_data,
            column_names=['MESSAGE'],
            prompt=prompt,
            is_multicolumn=False
        )
        
        async_time = time.time() - start_time
        async_throughput = test_size / async_time
        
        print(f"✅ ASYNC: {async_time:.2f}s ({async_throughput:.2f} items/s)")
        
    except Exception as e:
        print(f"❌ ASYNC FAILED: {str(e)}")
        async_time = None
        async_results = None
    
    # Delay
    print(f"\n⏳ Waiting 5s...")
    await asyncio.sleep(5)
    
    # Test batch
    print(f"\n📦 Testing BATCH processing...")
    start_time = time.time()
    
    try:
        batch_results = process_data_batch_only(
            model=model_obj,
            data=test_data,
            column_names=['MESSAGE'],
            prompt=prompt,
            is_multicolumn=False
        )
        
        batch_time = time.time() - start_time
        batch_throughput = test_size / batch_time
        
        print(f"✅ BATCH: {batch_time:.2f}s ({batch_throughput:.2f} items/s)")
        
    except Exception as e:
        print(f"❌ BATCH FAILED: {str(e)}")
        batch_time = None
        batch_results = None
    
    # Comparison
    print(f"\n🏆 COMPARISON")
    print("=" * 40)
    
    if async_time and batch_time:
        if async_time < batch_time:
            improvement = batch_time / async_time
            print(f"🚀 ASYNC is {improvement:.2f}x FASTER!")
        else:
            improvement = async_time / batch_time
            print(f"📦 BATCH is {improvement:.2f}x FASTER!")
        
        print(f"⚡ Async:  {async_time:.2f}s")
        print(f"📦 Batch:  {batch_time:.2f}s")
    else:
        print("❌ Could not compare due to errors")

if __name__ == "__main__":
    print("🎯 SIMPLE SPEED TEST")
    print("=" * 50)
    print("⚠️  Please edit the script to add your API key!")
    print()
    
    # Uncomment để chạy test:
    # asyncio.run(test_async_processing())
    
    print("✅ Script loaded successfully. Edit API key and uncomment the last line to run.") 