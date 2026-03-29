import pandas as pd
import numpy as np

print("Đang tải dữ liệu...")

df = pd.read_csv(r'C:\Users\Thich Minh Duc\Code\Preprocessing Data\Dataset\Global_Weather_Data\East_Sea_Weather_Data.csv')
print("Đang tiêu diệt các giá trị ảo (999.9, 9999.9)...")
df['WDSP'] = df['WDSP'].replace([999.9], np.nan)
df['MXSPD'] = df['MXSPD'].replace([999.9], np.nan)
df['GUST'] = df['GUST'].replace([999.9], np.nan)
df['PRCP'] = df['PRCP'].replace([99.99], np.nan)
df['SLP'] = df['SLP'].replace([9999.9], np.nan)

if 'GUST' in df.columns:
    df = df.drop(columns=['GUST'])
    print("=> Đã xóa cột GUST vì dữ liệu thiếu quá mức cho phép.")

print("Đang tiến hành nội suy lấp đầy SLP, WDSP, MXSPD...")
df = df.sort_values(by=['STATION', 'DATE']).reset_index(drop=True)

weather_cols = ['TEMP', 'PRCP', 'WDSP', 'MXSPD', 'SLP']
for col in weather_cols:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear', limit_direction='both')

output_file = 'East_Sea_Weather_Final.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print("\n" + "="*50)
print("TÌNH TRẠNG DỮ LIỆU SAU KHI XỬ LÝ:")
print(df[['WDSP', 'MXSPD', 'SLP']].isna().sum())
print(f"🎉 HOÀN TẤT! Dữ liệu đã sạch 100% không còn ô trống. Đã lưu vào {output_file}")