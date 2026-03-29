import os
import glob
import time

BASE_DIR = r"C:\Users\Thich Minh Duc\Code\bigdata\BTL_CuoiKy_dataset\Global_Weather_Data" 

YEARS = ['2019', '2020', '2021', '2022', '2023']
OUTPUT_FILE = "all_5_years.csv"

def merge_all_years():
    start_time = time.time()
    total_files = 0
    
    print(f"=== BẮT ĐẦU GỘP DỮ LIỆU {YEARS} ===")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        is_first_file = True # Biến này để giữ lại header của file đầu tiên
        
        for year in YEARS:
            search_path = os.path.join(BASE_DIR, year, "*.csv")
            files = glob.glob(search_path)
            
            print(f"--> Đang xử lý năm {year}: Tìm thấy {len(files)} file.")
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        # Đọc nội dung
                        lines = infile.readlines()
                        
                        if not lines: continue # Bỏ qua file rỗng

                        # XỬ LÝ HEADER (Tiêu đề cột)
                        if is_first_file:
                            # File đầu tiên: Ghi tất cả (bao gồm cả dòng tiêu đề)
                            outfile.writelines(lines)
                            is_first_file = False
                        else:
                            # Các file sau: Bỏ dòng đầu tiên (Header) đi, chỉ lấy dữ liệu
                            # Để file tổng không bị lặp lại chữ "STATION,DATE..." hàng nghìn lần
                            outfile.writelines(lines[1:]) 
                        
                        # Thêm ký tự xuống dòng ở cuối mỗi file con cho chắc ăn
                        if lines and not lines[-1].endswith('\n'):
                            outfile.write('\n')

                    total_files += 1
                    if total_files % 5000 == 0:
                        print(f"   ...Đã gộp được {total_files} file...")
                        
                except Exception as e:
                    print(f"Lỗi file {file_path}: {e}")

    duration = time.time() - start_time
    print("="*40)
    print(f"HOÀN TẤT! Tổng cộng {total_files} file đã được gộp.")
    print(f"File đầu ra: {os.path.abspath(OUTPUT_FILE)}")
    print(f"Thời gian xử lý: {duration:.2f} giây")

if __name__ == "__main__":
    merge_all_years()