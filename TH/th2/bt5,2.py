import matplotlib.pyplot as plt

# Dữ liệu Family Size của 2 nhóm
group_le_6 = [0, 0, 2, 2, 5, 9, 10, 11, 13, 13, 14]
group_gt_6 = [0, 0, 0, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 15]

# Thiết lập biểu đồ
plt.figure(figsize=(10, 5))

# Vẽ Box plot theo chiều ngang (vert=False)
plt.boxplot([group_le_6, group_gt_6], labels=['<= 6 years', '> 6 years'], vert=False)

# Trang trí trục và tiêu đề
plt.title('Box Plot: Family Size by Mother\'s Education Level')
plt.xlabel('Family Size')
plt.ylabel('Education Level')
plt.xticks(range(0, 17))
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Hiển thị
plt.show()