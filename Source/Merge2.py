import os
import glob
import time

# ==========================================
# CÀI ĐẶT ĐƯỜNG DẪN Ở ĐÂY
# ==========================================
# Trỏ đến thư mục cha chứa các thư mục năm (2019, 2020, 2021...)
BASE_DIR = r"C:\Users\Thich Minh Duc\Code\Preprocessing Data\Dataset\Global_Weather_Data" 

# Danh sách các năm bạn muốn gộp (thêm các năm khác nếu cần)
YEARS = ['2019', '2020', '2021', '2022', '2023'] 

# File đầu ra tổng
OUTPUT_FILE = r"C:\Users\Thich Minh Duc\Code\Preprocessing Data\Dataset\all_5_years_clean.csv"

def merge_clean_weather_data():
    start_time = time.time()
    valid_files_count = 0
    skipped_files_count = 0
    is_first_file = True
    
    print(f"=== BẮT ĐẦU GỘP DỮ LIỆU THÔNG MINH ===")
    
    # Mở file output để ghi (chế độ 'w' sẽ ghi đè nếu file đã tồn tại)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for year in YEARS:
            search_path = os.path.join(BASE_DIR, year, "*.csv")
            files = glob.glob(search_path)
            
            if not files:
                print(f"--> Cảnh báo: Không tìm thấy file nào trong thư mục năm {year}")
                continue
                
            print(f"--> Đang xử lý năm {year}: Tìm thấy {len(files)} file CSV...")
            
            for file_path in files:
                # 1. Lọc nhanh qua tên file (nếu tên file chứa chữ history, station...)
                filename = os.path.basename(file_path).lower()
                if "history" in filename or "station" in filename or "readme" in filename:
                    skipped_files_count += 1
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        # Đọc thử dòng đầu tiên (Header)
                        first_line = infile.readline()
                        
                        # 2. Kiểm tra Header: Dữ liệu thời tiết chuẩn NOAA GSOD thường phải có cột STATION và TEMP
                        if not first_line or "STATION" not in first_line or "TEMP" not in first_line:
                            skipped_files_count += 1
                            continue
                            
                        # Nếu đây là file hợp lệ đầu tiên, ghi luôn Header ra file tổng
                        if is_first_file:
                            outfile.write(first_line)
                            is_first_file = False
                        
                        # Ghi các dòng dữ liệu còn lại (bỏ qua dòng Header của file con này)
                        last_line = ""
                        for line in infile:
                            outfile.write(line)
                            last_line = line
                            
                        # Đảm bảo file con kết thúc bằng dấu xuống dòng để không bị dính vào file tiếp theo
                        if last_line and not last_line.endswith('\n'):
                            outfile.write('\n')
                            
                        valid_files_count += 1
                        
                        # In tiến độ cho đỡ chán
                        if valid_files_count % 2000 == 0:
                            print(f"   ...Đã gộp thành công {valid_files_count} file...")
                            
                except Exception as e:
                    print(f"Lỗi không thể đọc file {file_path}: {e}")

    duration = time.time() - start_time
    print("\n" + "="*50)
    print(f"🎉 HOÀN TẤT QUÁ TRÌNH GỘP FILE!")
    print(f"- Tổng số file hợp lệ đã gộp: {valid_files_count}")
    print(f"- Số lượng file rác bị loại bỏ: {skipped_files_count}")
    print(f"- Thời gian chạy: {duration:.2f} giây")
    print(f"- File đầu ra đã lưu tại: {OUTPUT_FILE}")
    print("="*50)

if __name__ == "__main__":
    merge_clean_weather_data()