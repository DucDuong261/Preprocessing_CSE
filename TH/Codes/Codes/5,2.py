import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập style cho biểu đồ
sns.set_style("darkgrid")

# 1. Đọc bộ dữ liệu tips.csv
tips_data = pd.read_csv('C:\Users\Thich Minh Duc\Downloads\tips.csv')

# 2. Sử dụng qcut để tìm các điểm ranh giới (bins) chia dữ liệu thành 5 khoảng 
# sao cho số lượng hóa đơn (frequency) trong mỗi khoảng là bằng nhau
discretised_bill, bins = pd.qcut(tips_data['total_bill'], 5, labels=None, retbins=True, precision=3, duplicates='raise')

# 3. Tạo nhãn tương ứng cho 5 Bins (từ Bin_no_1 đến Bin_no_5)
bin_labels = ['Bin_no_' + str(i) for i in range(1, 6)]

# 4. Áp dụng các điểm ranh giới đã tìm được vào hàm pd.cut để gán nhãn cho từng hàng dữ liệu
tips_data['bill_bins'] = pd.cut(x=tips_data['total_bill'], bins=bins, labels=bin_labels, include_lowest=True)

# 5. Đếm số lượng hóa đơn trong mỗi Bin để vẽ biểu đồ
plt.figure(figsize=(8, 6))
tips_data.groupby('bill_bins')['total_bill'].count().plot.bar(color='skyblue', edgecolor='black')

# 6. Trang trí biểu đồ
plt.title('Frequency of Total Bills in 5 Equal Frequency Bins')
plt.xlabel('Bill Bins')
plt.ylabel('Frequency (Count)')
plt.xticks(rotation=0)
plt.tight_layout()

# Lưu lại biểu đồ thành file hình ảnh
plt.savefig('tips_bins_plot.png')

# In kết quả tần suất trong mỗi Bin ra màn hình
print(tips_data['bill_bins'].value_counts().sort_index())