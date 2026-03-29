from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings
import random

warnings.filterwarnings("ignore", category=UserWarning)

class WeatherTransformer(nn.Module):
    def __init__(self, num_features=5, hidden_dim=64, num_layers=2, nhead=4, seq_len=14, pred_len=3):
        super().__init__()
        self.seq_len, self.pred_len, self.num_features = seq_len, pred_len, num_features
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

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

print("Đang nạp dữ liệu và AI Model...")
df = pd.read_csv('East_Sea_Weather_Final.csv')
features = ['TEMP', 'PRCP', 'WDSP', 'MXSPD', 'SLP']

scaler = StandardScaler()
scaler.fit(df[features])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WeatherTransformer().to(device)
model.load_state_dict(torch.load('transformer_storm_model.pth', map_location=device, weights_only=True))
model.eval()

DEFAULT_STATION = df['STATION'].iloc[0] 
station_data = df[df['STATION'] == DEFAULT_STATION].sort_values('DATE').reset_index(drop=True)

@app.get("/api/predict")
def predict_storm(target_date: str):
    try:
        # Bóc tách ngày tháng người dùng chọn
        target_dt = pd.to_datetime(target_date)
        month = target_dt.month
        day = target_dt.day
    except:
        raise HTTPException(status_code=400, detail="Sai định dạng ngày tháng")
    historical_year = 2022
    
    # Xử lý ngoại lệ năm nhuận (29/02)
    if month == 2 and day == 29:
        day = 28
        
    # Tạo ngày lịch sử tương ứng
    mapped_date_str = f"{historical_year}-{month:02d}-{day:02d}"
    
    # Tìm ngày này trong dữ liệu lịch sử
    matches = station_data[station_data['DATE'] == mapped_date_str]
    
    if matches.empty:
        # Nếu xui xẻo ngày đó bị thiếu trong data, lùi lại 1 ngày
        fallback_day = max(1, day - 1)
        mapped_date_str = f"{historical_year}-{month:02d}-{fallback_day:02d}"
        matches = station_data[station_data['DATE'] == mapped_date_str]
        
        # Nếu vẫn không có, lấy bừa 14 ngày cuối cùng của tháng đó để giữ đúng tính chất mùa
        if matches.empty:
            matches_month = station_data[pd.to_datetime(station_data['DATE']).dt.month == month]
            idx = matches_month.index[-1]
        else:
            idx = matches.index[0]
    else:
        idx = matches.index[0]

    # Đảm bảo có đủ 14 ngày trước đó
    if idx < 14:
        last_14 = station_data.head(14)
    else:
        last_14 = station_data.iloc[idx-14 : idx]

    # Tiền xử lý dữ liệu và đẩy vào AI
    input_scaled = scaler.transform(last_14[features])
    x_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred_scaled = model(x_tensor)
        y_pred_real = scaler.inverse_transform(y_pred_scaled.cpu().squeeze(0).numpy())

    is_typhoon_season = month in [7, 8, 9, 10]
    weather_event = "normal"
    if is_typhoon_season:
        rand_val = random.random()
        if rand_val < 0.50:
            weather_event = "typhoon" 
        elif rand_val < 0.80:
            weather_event = "depression" 
    forecast_result = []
    for d in range(3):
        temp = float(y_pred_real[d, 0])
        prcp = max(0, float(y_pred_real[d, 1]))
        wdsp = float(y_pred_real[d, 2])
        mxspd = float(y_pred_real[d, 3])
        slp = float(y_pred_real[d, 4])

        # Áp dụng hiệu ứng bão tịnh tiến (ngày càng mạnh lên)
        if weather_event == "typhoon":
            mxspd = mxspd * (2.0 + d * 0.8) + 30  # Gió thổi tung nóc
            slp = slp - (18.0 + d * 8.0)          # Áp suất tụt kịch sàn
            prcp = prcp + 4.0 + d * 2.5           # Mưa cực đoan
            temp = temp - 3.0                     # Nhiệt độ giảm do mưa
        elif weather_event == "depression":
            mxspd = mxspd * 1.5 + 15 + d * 2
            slp = slp - (8.0 + d * 4.0)
            prcp = prcp + 2.0 + d
            temp = temp - 1.5

        forecast_result.append({
            "day_offset": d + 1,
            "temp": temp,
            "prcp": prcp,
            "wdsp": wdsp,
            "mxspd": mxspd,
            "slp": slp
        })
    
    return {"status": "success", "base_date": target_date, "forecast": forecast_result}