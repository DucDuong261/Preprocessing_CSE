import pandas as pd
import requests
import os
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

years_to_download = range(2019, 2024) 
output_folder = 'Global_Weather_Data'
history_file = 'isd-history.csv' # File metadata gốc của NOAA

# cấu hình kết nối với retry
session = requests.Session()
retry = Retry(connect=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# lấy danh sách 12000 trạm toàn cầu từ file history
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print(">> Đang đọc danh sách trạm toàn cầu...")
try:
   
    df = pd.read_csv(history_file)
    
    #lọc các tram còn hoạt động để tránh  tải các trạm dừng hoạt động
    active_stations = df[df['END'] >= 20230101].copy()
    
    # Tạo tên file chuẩn (USAF + WBAN không gạch ngang)
    def make_filename(row):
        usaf = str(row['USAF']).zfill(6)
        wban = str(row['WBAN']).zfill(5)
        return f"{usaf}{wban}"
        
    # Lấy danh sách ID
    target_ids = active_stations.apply(make_filename, axis=1).tolist()
    
    print(f">> Tìm thấy {len(target_ids)} trạm đang hoạt động trên thế giới.")
    print(">> Chuẩn bị tải dữ liệu Big Data...")
    
except Exception as e:
    print(f"[Lỗi] Không đọc được file {history_file}. Hãy tải nó về trước!")
    print(e)
    exit()

# URL gốc để tải dữ liệu
base_url = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access"

count_total = 0
start_time = time.time()

for year in years_to_download:
    print(f"\n==========================================")
    print(f"  ĐANG XỬ LÝ NĂM {year}")
    print(f"==========================================")
    
    year_folder = os.path.join(output_folder, str(year))
    if not os.path.exists(year_folder):
        os.makedirs(year_folder)
        
    for i, station_id in enumerate(target_ids):
        file_name = f"{station_id}.csv"
        save_path = os.path.join(year_folder, file_name)
        
        # Check nếu tải rồi thì bỏ qua
        if os.path.exists(save_path):
            continue
            
        download_url = f"{base_url}/{year}/{file_name}"
        
        try:
            # Timeout 20s
            response = session.get(download_url, headers=headers, timeout=20)
            
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                # In ra tiến trình mỗi 100 file
                if i % 100 == 0:
                    print(f"  [OK] Đã tải gói {i}/{len(target_ids)} - File: {file_name}")
                count_total += 1
            elif response.status_code == 404:
                # Trạm này năm này không có số liệu -> Bỏ qua
                pass
            
            # Nghỉ một khoảng thời gian để tránh bị chặn
            time.sleep(0.05) 

        except Exception as e:
            print(f"  [Err] {file_name}: {e}")

print("\n--- HOÀN TẤT CHIẾN DỊCH ---")
print(f"Tổng số file tải được: {count_total}")
print(f"Thời gian chạy: {(time.time() - start_time)/60:.1f} phút")