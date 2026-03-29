import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class EastSeaStormDataset(Dataset):
    def __init__(self, csv_path, seq_len=14, pred_len=3, features=['TEMP', 'PRCP', 'WDSP', 'MXSPD', 'SLP']):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        
        print(f"Đang đọc dữ liệu từ {csv_path}...")
        df = pd.read_csv(csv_path)
      
        self.scaler = StandardScaler()
        df[self.features] = self.scaler.fit_transform(df[self.features])
        
        self.x_data = []
        self.y_data = []
        
        print("Đang tiến hành cắt Sliding Windows theo từng trạm...")
        grouped = df.groupby('STATION')
        
        for station, group in grouped:
            data_values = group[self.features].values
            
            # Bỏ qua nếu trạm này có quá ít dữ liệu, không đủ cắt 1 cửa sổ
            if len(data_values) < (self.seq_len + self.pred_len):
                continue
                
            # Trượt từ đầu đến cuối dữ liệu của trạm này
            for i in range(len(data_values) - self.seq_len - self.pred_len + 1):
                # Quá khứ (Input X)
                x = data_values[i : i + self.seq_len]
                # Tương lai (Target Y)
                y = data_values[i + self.seq_len : i + self.seq_len + self.pred_len]
                
                self.x_data.append(x)
                self.y_data.append(y)
                
        # 3. CHUYỂN ĐỔI SANG PYTORCH TENSOR
        self.x_data = torch.tensor(np.array(self.x_data), dtype=torch.float32)
        self.y_data = torch.tensor(np.array(self.y_data), dtype=torch.float32)
        
        print(f"🎉 Hoàn tất! Đã tạo ra {len(self.x_data)} mẫu (samples).")
        print(f"Kích thước X (Input): {self.x_data.shape} -> (Batch, Seq_len, Features)")
        print(f"Kích thước Y (Target): {self.y_data.shape} -> (Batch, Pred_len, Features)")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

csv_file = r'C:\Users\Thich Minh Duc\Code\Preprocessing Data\East_Sea_Weather_Final.csv'

# Khởi tạo Dataset (Nhìn 14 ngày quá khứ, đoán 3 ngày tương lai)
dataset = EastSeaStormDataset(csv_path=csv_file, seq_len=14, pred_len=3)

# Đóng gói vào DataLoader (Mỗi lần nhét 64 mẫu vào mô hình để Train)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

# Lấy thử 1 Batch ra xem hình thù thế nào
x_batch, y_batch = next(iter(dataloader))
print(f"\nMột batch X lấy ra để đưa vào Informer có shape: {x_batch.shape}")

import torch.nn as nn
import torch.optim as optim

# ==========================================
# BƯỚC 2: XÂY DỰNG KIẾN TRÚC TRANSFORMER
# ==========================================
class WeatherTransformer(nn.Module):
    def __init__(self, num_features=5, hidden_dim=64, num_layers=2, nhead=4, seq_len=14, pred_len=3):
        super(WeatherTransformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features
        
        # 1. Lớp nhúng (Embedding): Nâng 5 đặc trưng lên không gian đa chiều (64 chiều) để Transformer dễ phân tích
        self.embedding = nn.Linear(num_features, hidden_dim)
        
        # 2. Lõi Transformer Encoder: Học cơ chế "Sự chú ý" (Attention) giữa các ngày trong quá khứ
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Lớp đầu ra (Output Layer): Chuyển thông tin đã học thành dự báo 3 ngày tới
        self.fc_out = nn.Linear(seq_len * hidden_dim, pred_len * num_features)
        
    def forward(self, x):
        # x shape: [Batch, 14, 5]
        embedded = self.embedding(x)                         # -> [Batch, 14, 64]
        encoded = self.transformer_encoder(embedded)         # -> [Batch, 14, 64]
        
        # Ép phẳng dữ liệu để đưa qua lớp suy luận cuối cùng
        encoded_flat = encoded.reshape(encoded.size(0), -1)  # -> [Batch, 14 * 64]
        out = self.fc_out(encoded_flat)                      # -> [Batch, 3 * 5]
        
        # Định hình lại cho khớp với Y_Target: [Batch, 3, 5]
        return out.view(-1, self.pred_len, self.num_features)

# Khởi tạo mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n=> Đang sử dụng thiết bị: {device}")

model = WeatherTransformer().to(device)

# ==========================================
# BƯỚC 3: VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP)
# ==========================================
# Hàm mất mát (MSE) để tính sai số dự báo
criterion = nn.MSELoss()
# Thuật toán tối ưu hóa (Adam) để cập nhật trọng số
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5  # Chạy thử 5 vòng trước để xem mô hình có hội tụ không

print("\n BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    # Duyệt qua từng batch dữ liệu (64 mẫu mỗi lần)
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        # Xóa gradient cũ
        optimizer.zero_grad()
        
        # Đưa X (14 ngày) vào mô hình để dự đoán Y_hat (3 ngày)
        y_pred = model(x_batch)
        
        # Tính sai số giữa Dự đoán (y_pred) và Thực tế (y_batch)
        loss = criterion(y_pred, y_batch)
        
    
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        
        if (batch_idx + 1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
            
    avg_loss = total_loss / len(dataloader)
    print(f"Hết Epoch {epoch+1} | Lỗi trung bình (Loss): {avg_loss:.4f}")

# Lưu trọng số mô hình sau khi học xong
torch.save(model.state_dict(), 'transformer_storm_model.pth')
print("\n Đã lưu mô hình thành công vào file: transformer_storm_model.pth")