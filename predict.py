import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# =======================================================
# 1. KHAI BÁO KIẾN TRÚC MÔ HÌNH
# =======================================================
class WeatherTransformer(nn.Module):
    def __init__(self, num_features=5, hidden_dim=64, num_layers=2, nhead=4, seq_len=14, pred_len=3):
        super(WeatherTransformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        
        self.embedding = nn.Linear(num_features, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(seq_len * hidden_dim, pred_len * num_features)
        
    def forward(self, x):
        embedded = self.embedding(x)                         
        encoded = self.transformer_encoder(embedded)         
        encoded_flat = encoded.reshape(encoded.size(0), -1)  
        out = self.fc_out(encoded_flat)                      
        return out.view(-1, self.pred_len, self.num_features)

# =======================================================
# 2. HÀM LUẬT: PHÂN LOẠI NGUY CƠ BÃO (ĐÃ NÂNG CẤP)
# =======================================================
def classify_storm_risk(slp, mxspd, prcp):
    """
    Hàm đánh giá rủi ro đa biến: Gió (knots), Áp suất (hPa) và Lượng mưa (inches)
    """
    # 🔴 Nguy cơ CAO: Bão mạnh hoặc Siêu bão
    if mxspd >= 64 or slp <= 980:
        return "🔴 NGUY CƠ CAO (Bão mạnh/Siêu bão - Gió giật cấp 12+)"
    
    # 🟡 Nguy cơ TRUNG BÌNH: Áp thấp / Bão yếu HOẶC Mưa lớn cực đoan
    # Mưa > 3.0 inches (~76mm) kèm áp suất chùng xuống là dấu hiệu xoáy thuận nhiệt đới
    elif (34 <= mxspd < 64) or (980 < slp <= 1000) or (prcp > 3.0 and slp <= 1005):
        if prcp > 3.0:
            return "🟡 NGUY CƠ TRUNG BÌNH (Cảnh báo mưa lũ lớn & Áp thấp)"
        else:
            return "🟡 NGUY CƠ TRUNG BÌNH (Áp thấp nhiệt đới / Bão yếu)"
    
    # 🟢 Nguy cơ THẤP
    else:
        return "🟢 NGUY CƠ THẤP (Thời tiết biển bình thường)"

# =======================================================
# 3. TÁI TẠO BỘ CHUẨN HÓA (SCALER) TỪ DỮ LIỆU GỐC
# =======================================================
# LƯU Ý: Đảm bảo đường dẫn này trỏ đúng tới file Data sạch của bạn
csv_file = r'C:\Users\Thich Minh Duc\Code\Preprocessing Data\East_Sea_Weather_Final.csv'
features = ['TEMP', 'PRCP', 'WDSP', 'MXSPD', 'SLP']

print("Đang khởi tạo môi trường dự báo và bộ Scaler...")
df = pd.read_csv(csv_file)

# Khởi tạo scaler để khớp tỷ lệ gốc
scaler = StandardScaler()
scaler.fit(df[features]) 

# =======================================================
# 4. LOAD TRỌNG SỐ MÔ HÌNH ĐÃ TRAIN
# =======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WeatherTransformer().to(device)

try:
    model.load_state_dict(torch.load('transformer_storm_model.pth', map_location=device, weights_only=True))
    model.eval()
    print("✅ Đã load thành công AI Model 'transformer_storm_model.pth'")
except FileNotFoundError:
    print("❌ LỖI: Không tìm thấy file 'transformer_storm_model.pth'. Hãy chắc chắn bạn đã chạy xong file train!")
    exit()

# =======================================================
# 5. LẤY DỮ LIỆU THỰC TẾ & CHẠY DỰ BÁO
# =======================================================
# Đổi mã trạm ở đây nếu bạn muốn xem một trạm cụ thể trên Biển Đông
sample_station = df['STATION'].iloc[0] 

# Trích xuất đúng 14 ngày cuối cùng của trạm đó
station_data = df[df['STATION'] == sample_station].sort_values('DATE')
if len(station_data) < 14:
    print("❌ Trạm này không có đủ 14 ngày dữ liệu để dự báo!")
    exit()

last_14_days = station_data.tail(14)

print(f"\n=> 📡 Đang phân tích dữ liệu Trạm: {sample_station}")
print(f"=> 📅 Dữ liệu đầu vào từ: {last_14_days['DATE'].iloc[0]} đến {last_14_days['DATE'].iloc[-1]}")

# Scale 14 ngày này và đẩy vào Tensor
input_features = last_14_days[features].values
input_scaled = scaler.transform(input_features)
x_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(device)

# Chạy Inference (Dự báo)
with torch.no_grad():
    y_pred_scaled = model(x_tensor)
    y_pred_numpy = y_pred_scaled.cpu().squeeze(0).numpy()
    
    # Giải chuẩn hóa trả về số đo vật lý
    y_pred_real = scaler.inverse_transform(y_pred_numpy)

# =======================================================
# 6. IN KẾT QUẢ ĐẦU RA (TÍCH HỢP LƯỢNG MƯA)
# =======================================================
print("\n" + "="*65)
print("📊 KẾT QUẢ DỰ BÁO 3 NGÀY TỚI TỪ AI TRANSFORMER")
print("="*65)

for day in range(3):
    temp_pred = y_pred_real[day, 0]
    prcp_pred = y_pred_real[day, 1]  # Lấy lượng mưa
    wdsp_pred = y_pred_real[day, 2]
    mxspd_pred = y_pred_real[day, 3] # Lấy sức gió tối đa
    slp_pred = y_pred_real[day, 4]   # Lấy áp suất
    
    # Gọi hàm phân loại mới (Truyền 3 tham số)
    risk_label = classify_storm_risk(slp=slp_pred, mxspd=mxspd_pred, prcp=prcp_pred)
    
    print(f"▶️ NGÀY DỰ BÁO +{day + 1}:")
    print(f"  🌡 Nhiệt độ:    {temp_pred:.1f} °F")
    print(f"  🌧 Lượng mưa:   {prcp_pred:.2f} inches")
    print(f"  🌪 Áp suất SLP: {slp_pred:.1f} hPa")
    print(f"  🌬 Gió tối đa:  {mxspd_pred:.1f} knots")
    print(f"  => {risk_label}\n")