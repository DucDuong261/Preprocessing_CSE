import pandas as pd

df = pd.read_csv(r'C:\Users\Thich Minh Duc\Code\bigdata\BTL_CuoiKy_dataset\isd-history.csv')


lat_min, lat_max = 5.0, 25.0
lon_min, lon_max = 100.0, 125.0


east_sea_stations = df[
    (df['LAT'] >= lat_min) & (df['LAT'] <= lat_max) &
    (df['LON'] >= lon_min) & (df['LON'] <= lon_max)
]


def make_filename(row):
    usaf = str(row['USAF']).zfill(6) # Đảm bảo đủ 6 số
    wban = str(row['WBAN']).zfill(5) # Đảm bảo đủ 5 số
    return f"{usaf}-{wban}"

east_sea_stations['FileName'] = east_sea_stations.apply(make_filename, axis=1)

recent_stations = east_sea_stations[east_sea_stations['END'] >= 20230101]

print(f"Tìm thấy {len(recent_stations)} trạm khí tượng tại khu vực Biển Đông.")
print(recent_stations[['STATION NAME', 'CTRY', 'FileName']].head(10))


recent_stations[['FileName', 'CTRY', 'LAT', 'LON']].to_csv('east_sea_station_list.csv', index=False)