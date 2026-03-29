import pandas as pd
import requests
import os
import time

# --- CẤU HÌNH ---
station_file = 'east_sea_station_list.csv' 
# Chọn năm gần nhất để test trước (2022 hoặc 2023)
years_to_download = range(2023, 2024) 
output_folder = 'EastSea_Data'

# --- XỬ LÝ ---
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Header giả lập trình duyệt (QUAN TRỌNG)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

try:
    df_stations = pd.read_csv(station_file)
    target_stations = df_stations['FileName'].tolist()
    print(f"Đã đọc được {len(target_stations)} trạm từ danh sách.")
except Exception as e:
    print(f"Lỗi đọc file danh sách: {e}")
    exit()

base_url = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access"

print("--- BẮT ĐẦU TẢI DỮ LIỆU ---")

count_success = 0
count_fail = 0

for year in years_to_download:
    print(f"\n>> Đang xử lý năm {year}...")
    
    year_folder = os.path.join(output_folder, str(year))
    if not os.path.exists(year_folder):
        os.makedirs(year_folder)

    for station_id in target_stations:
        file_name = f"{station_id.replace('-', '')}.csv"
        download_url = f"{base_url}/{year}/{file_name}"
        save_path = os.path.join(year_folder, file_name)

        if os.path.exists(save_path):
            print(f"  [Skip] {file_name} đã tồn tại.")
            continue

        try:
            # Thêm headers vào request
            response = requests.get(download_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"  [OK] Đã tải {file_name}")
                count_success += 1
            else:
                # Hiện lỗi cụ thể
                print(f"  [Miss] Code {response.status_code} - URL: {download_url}")
                count_fail += 1
        except Exception as e:
            print(f"  [Error] {e}")
            count_fail += 1

print("\n--- HOÀN TẤT ---")
print(f"Thành công: {count_success}")
print(f"Thất bại: {count_fail}")