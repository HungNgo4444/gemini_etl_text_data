import pandas as pd
from datetime import datetime, timedelta
import random

def create_multicolumn_demo_data():
    """Tạo dữ liệu demo với nhiều cột để test tính năng multi-column"""
    
    # Dữ liệu mẫu
    products = [
        "ChocoPie Truyền Thống", "Bánh quy deMarie", "Nước khoáng núi lửa Jeju",
        "Sữa chua uống ProYo Cam", "Custas nhân Kem Trứng", "Tok vị Phô mai",
        "An Vị Tự Nhiên", "Jungle Boy Lắk phô mai", "Masita vị bắp bơ"
    ]
    
    channels = ["Facebook", "Instagram", "TikTok", "YouTube", "Website"]
    
    authors = [
        "Nguyễn Văn A", "Trần Thị B", "Lê Văn C", "Phạm Thị D", "Hoàng Văn E",
        "Vũ Thị F", "Đỗ Văn G", "Bùi Thị H", "Ngô Văn I", "Lý Thị K"
    ]
    
    messages = [
        "Sản phẩm này rất ngon, tôi đã mua nhiều lần và rất hài lòng với chất lượng.",
        "Giá cả hợp lý, đóng gói đẹp, giao hàng nhanh. Sẽ ủng hộ tiếp!",
        "Chất lượng tuyệt vời, vị ngon như mong đợi. Recommend cho mọi người.",
        "Tôi muốn hỏi về chương trình khuyến mãi tháng này có gì không?",
        "Sản phẩm bị hỏng khi nhận hàng, cần hỗ trợ đổi trả.",
        "Bánh ngon nhưng hơi ngọt, có thể giảm đường một chút không?",
        "Shop có bán ở khu vực Hà Nội không? Tôi muốn mua trực tiếp.",
        "Rất thích packaging mới, trông sang trọng và hiện đại.",
        "Đã thử nhiều vị khác nhau, đều ngon cả. Cảm ơn shop!",
        "Có thể tư vấn sản phẩm phù hợp cho trẻ em không?",
        "Mình mua làm quà tặng sinh nhật, bạn rất thích!",
        "Chương trình tích điểm thế nào? Làm sao để đổi quà?",
        "Bánh hết hạn sử dụng rồi nhưng vẫn còn trong bao bì.",
        "Cảm ơn shop đã giao hàng nhanh trong dịp Tết.",
        "Có khuyến mãi gì cho khách hàng thân thiết không?"
    ]
    
    sentiments = ["Positive", "Negative", "Neutral"]
    
    # Tạo 30 records demo
    data = []
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(30):
        record = {
            "ID": f"MSG_{i+1:03d}",
            "MESSAGE": random.choice(messages),
            "AUTHOR": random.choice(authors),
            "PRODUCT": random.choice(products),
            "CHANNEL": random.choice(channels),
            "DATE": (base_date + timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d"),
            "SENTIMENT": random.choice(sentiments),
            "RATING": random.randint(1, 5),
            "CATEGORY": random.choice(["Review", "Question", "Complaint", "Compliment", "Inquiry"])
        }
        data.append(record)
    
    # Tạo DataFrame
    df = pd.DataFrame(data)
    
    # Lưu file
    output_file = "demo_multicolumn_data.xlsx"
    df.to_excel(output_file, index=False)
    
    print("📊 DEMO DATA MULTI-COLUMN")
    print("="*50)
    print(f"✅ Đã tạo file: {output_file}")
    print(f"📈 Số records: {len(df)}")
    print(f"📋 Các cột:")
    
    for i, col in enumerate(df.columns, 1):
        sample_data = df[col].iloc[0]
        print(f"  {i}. {col} (VD: {sample_data})")
    
    print(f"\n💡 Gợi ý sử dụng:")
    print(f"   - Chọn cột: 2,3,4 (MESSAGE, AUTHOR, PRODUCT)")
    print(f"   - Prompt: Phân tích sentiment và tóm tắt nội dung")
    print(f"   - Kết quả sẽ bao gồm cả 3 cột thông tin")
    
    return output_file

if __name__ == "__main__":
    create_multicolumn_demo_data() 