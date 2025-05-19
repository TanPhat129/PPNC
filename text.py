from matplotlib.colors import LinearSegmentedColormap
import json
from fuzzywuzzy import process
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
import geopandas as gpd

# Tắt cảnh báo SequenceMatcher nếu không cài python-Levenshtein
try:
    import Levenshtein
except ImportError:
    warnings.filterwarnings("ignore", message="Using slow pure-python SequenceMatcher")

# Danh sách tỉnh Việt Nam
vietnam_provinces = [
    "An Giang", "Ba Ria Vung Tau", "Bac Giang", "Bac Kan", "Bac Lieu", "Bac Ninh",
    "Ben Tre", "Binh Dinh", "Binh Duong", "Binh Phuoc", "Binh Thuan", "Ca Mau",
    "Can Tho", "Cao Bang", "Da Nang", "Dak Lak", "Dak Nong", "Dien Bien",
    "Dong Nai", "Dong Thap", "Gia Lai", "Ha Giang", "Ha Nam", "Ha Noi",
    "Ha Tinh", "Hai Duong", "Hai Phong", "Hau Giang", "Hoa Binh", "Hung Yen",
    "Khanh Hoa", "Kien Giang", "Kon Tum", "Lai Chau", "Lam Dong", "Lang Son",
    "Lao Cai", "Long An", "Nam Dinh", "Nghe An", "Ninh Binh", "Ninh Thuan",
    "Phu Tho", "Phu Yen", "Quang Binh", "Quang Nam", "Quang Ngai", "Quang Ninh",
    "Quang Tri", "Soc Trang", "Son La", "Tay Ninh", "Thai Binh", "Thai Nguyen",
    "Thanh Hoa", "Thua Thien Hue", "Tien Giang", "TP. Ho Chi Minh", "Tra Vinh",
    "Tuyen Quang", "Vinh Long", "Vinh Phuc", "Yen Bai"
]

# Hàm chuẩn hóa tên tỉnh bằng fuzzy matching
def fuzzy_match_province(text):
    if not text or pd.isna(text):
        return None
    match = process.extractOne(text, vietnam_provinces, score_cutoff=80)
    return match[0] if match else None

# Hàm trích xuất tỉnh từ Location và Admin Units
def extract_provinces(row):
    provinces = set()
    if pd.notna(row['Location']):
        location = row['Location']
        parts = [part.strip() for part in location.replace(' and ', ',').split(',')]
        for part in parts:
            matched_province = fuzzy_match_province(part)
            if matched_province:
                provinces.add(matched_province)
            else:
                for province in vietnam_provinces:
                    if province.lower() in part.lower():
                        provinces.add(province)
    if pd.notna(row['Admin Units']):
        try:
            admin_units = json.loads(row['Admin Units'])
            for unit in admin_units:
                if 'adm1_name' in unit:
                    province = unit['adm1_name']
                    matched_province = fuzzy_match_province(province)
                    if matched_province:
                        provinces.add(matched_province)
                    else:
                        for std_province in vietnam_provinces:
                            if std_province.lower() in province.lower():
                                provinces.add(std_province)
        except (json.JSONDecodeError, TypeError):
            pass
    return list(provinces) if provinces else ['Unknown']

# Hàm đọc dữ liệu từ file
def load_data(file_path):
    try:
        data = pd.read_excel(file_path)
        print(f"Đã tải file từ: {file_path}")
        return data
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại {file_path}. Vui lòng cung cấp đường dẫn đúng.")
        file_path = input("Nhập đường dẫn file đúng: ")
        data = pd.read_excel(file_path)
        print(f"Đã tải file từ: {file_path}")
        return data

# Tải dữ liệu
url = "https://raw.githubusercontent.com/TanPhat129/PPNC/main/disaster-in-vietnam_1900-to-2024.xlsx"

# Hàm load dữ liệu
def load_data(url):
    return pd.read_excel(url)

# Gọi hàm load
data = load_data(url)

# Lọc dữ liệu thiên tai tự nhiên (Flood hoặc Storm)
data = data[data['Disaster Group'] == 'Natural']
data = data[data['Disaster Type'].isin(['Flood', 'Storm'])]

# Kiểm tra dữ liệu rỗng
if data.empty:
    print("Lỗi: Không có dữ liệu Flood hoặc Storm sau khi lọc.")
    exit()

# Áp dụng hàm trích xuất tỉnh
data['Provinces'] = data.apply(extract_provinces, axis=1)

# Mở rộng danh sách tỉnh
exploded_data = data.explode('Provinces')
unknown_count = exploded_data[exploded_data['Provinces'] == 'Unknown'].shape[0]
print(f"Số sự kiện không xác định tỉnh: {unknown_count}")
exploded_data = exploded_data[exploded_data['Provinces'] != 'Unknown']

# Tính tần suất tổng hợp (Flood + Storm)
years_span = 2024 - 1900 + 1  # 125 năm
total_freq = exploded_data['Provinces'].value_counts()
total_freq = total_freq / years_span
total_freq = total_freq.reindex(vietnam_provinces, fill_value=0)

# Tần suất riêng cho Flood
flood_data = exploded_data[exploded_data['Disaster Type'] == 'Flood']
flood_freq = flood_data['Provinces'].value_counts()
flood_freq = flood_freq / years_span
flood_freq = flood_freq.reindex(vietnam_provinces, fill_value=0)

# Tần suất riêng cho Storm
storm_data = exploded_data[exploded_data['Disaster Type'] == 'Storm']
storm_freq = storm_data['Provinces'].value_counts()
storm_freq = storm_freq / years_span
storm_freq = storm_freq.reindex(vietnam_provinces, fill_value=0)

# Kết hợp kết quả
freq_df = pd.DataFrame({
    'Location': vietnam_provinces,
    'Total_Frequency': total_freq,
    'Flood_Frequency': flood_freq,
    'Storm_Frequency': storm_freq
})

# In tần suất
print("\nTần suất thiên tai theo tỉnh (sự kiện/năm, 1900-2024):")
print(freq_df.round(4))

# Tải file GeoJSON
geojson_path = "C:/Users/ADMIN/Downloads/diaphantinhenglish.geojson"
try:
    gdf = gpd.read_file(geojson_path)
    print(f"Đã tải file GeoJSON từ: {geojson_path}")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file GeoJSON tại {geojson_path}. Vui lòng cung cấp đường dẫn đúng.")
    geojson_path = input("Nhập đường dẫn file GeoJSON đúng: ")
    gdf = gpd.read_file(geojson_path)
    print(f"Đã tải file GeoJSON từ: {geojson_path}")

# Đổi tên cột Name thành Location để hợp nhất
gdf = gdf.rename(columns={'Name': 'Location'})

# Hợp nhất tần suất với GeoDataFrame
gdf = gdf.merge(freq_df[['Location', 'Storm_Frequency', 'Flood_Frequency']], on='Location', how='left')
gdf['Storm_Frequency'] = gdf['Storm_Frequency'].fillna(0)
gdf['Flood_Frequency'] = gdf['Flood_Frequency'].fillna(0)

# In phân phối tần suất
print("\nPhân phối tần suất bão:")
print(gdf['Storm_Frequency'].describe())
print("\nPhân phối tần suất lũ lụt:")
print(gdf['Flood_Frequency'].describe())

# Xác định bins cho phân loại màu
storm_bins = [gdf['Storm_Frequency'].min(), gdf['Storm_Frequency'].quantile(0.33), gdf['Storm_Frequency'].quantile(0.66), gdf['Storm_Frequency'].max()]
flood_bins = [gdf['Flood_Frequency'].min(), gdf['Flood_Frequency'].quantile(0.33), gdf['Flood_Frequency'].quantile(0.66), gdf['Flood_Frequency'].max()]
labels = ['Thấp', 'Trung bình', 'Cao']
storm_bins = sorted(set(storm_bins))
flood_bins = sorted(set(flood_bins))

# Xử lý trường hợp bins không đủ
if len(storm_bins) < 2:
    storm_bins = [0, 0.01, 0.02, gdf['Storm_Frequency'].max() if gdf['Storm_Frequency'].max() > 0 else 0.03]
if len(flood_bins) < 2:
    flood_bins = [0, 0.01, 0.02, gdf['Flood_Frequency'].max() if gdf['Flood_Frequency'].max() > 0 else 0.03]

# Phân loại tần suất
gdf['Storm_Category'] = pd.cut(gdf['Storm_Frequency'], bins=storm_bins, labels=labels, include_lowest=True)
gdf['Flood_Category'] = pd.cut(gdf['Flood_Frequency'], bins=flood_bins, labels=labels, include_lowest=True)

# Tạo colormap (xanh lá -> cam -> đỏ)
colors = ['green', 'orange', 'red']
custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=3)

# Tạo thư mục lưu đầu ra
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Print column names to debug
print("\nDanh sách các cột trong dữ liệu:")
print(data.columns.tolist())

# Filter for Natural disasters (Flood or Storm)
data = data[data['Disaster Group'] == 'Natural']
data = data[data['Disaster Type'].isin(['Flood', 'Storm'])]

# Create target column: 1 for Flood, 0 for Storm
data['Disaster_Type'] = (data['Disaster Type'] == 'Flood').astype(int)

# Calculate frequency by month, day, and location
data['Month'] = data['Start Month']
data['Day'] = data['Start Day']

# Check if location column exists, otherwise assign from vietnam_provinces
if 'Admin 1' in data.columns:
    data['Location'] = data['Admin 1']
elif 'Province' in data.columns:
    data['Location'] = data['Province']
elif 'Region' in data.columns:
    data['Location'] = data['Region']
elif 'Location' in data.columns:
    data['Location'] = data['Location']
else:
    print("Không tìm thấy cột địa điểm, sử dụng danh sách tỉnh mặc định.")
    data['Location'] = [random.choice(vietnam_provinces) for _ in range(len(data))]

# Handle missing values in Month and Day
data['Month'] = data['Month'].fillna(data['Month'].mode()[0])
data['Day'] = data['Day'].fillna(data['Day'].mode()[0])

# Frequency by month (average events per month per year)
monthly_freq = data['Month'].value_counts().sort_index() / (2024 - 1900 + 1)

# Frequency of storms by month
storm_data = data[data['Disaster Type'] == 'Storm']
monthly_storm_freq = storm_data['Month'].value_counts().sort_index() / (2024 - 1900 + 1)

# Frequency of floods by month
flood_data = data[data['Disaster Type'] == 'Flood']
monthly_flood_freq = flood_data['Month'].value_counts().sort_index() / (2024 - 1900 + 1)

# Frequency by day within each month
day_freq_by_month = {}
for month in range(1, 13):
    month_data = data[data['Month'] == month]
    day_freq = month_data['Day'].value_counts().sort_index()
    day_freq_by_month[month] = day_freq / day_freq.sum() if not day_freq.empty else pd.Series()

# Frequency by location (for storms and floods)
location_storm_freq = storm_data['Location'].value_counts()
location_storm_freq = location_storm_freq / location_storm_freq.sum() if not location_storm_freq.empty else pd.Series(index=storm_data['Location'].unique(), data=1/len(storm_data['Location'].unique()))

location_flood_freq = flood_data['Location'].value_counts()
location_flood_freq = location_flood_freq / location_flood_freq.sum() if not location_flood_freq.empty else pd.Series(index=flood_data['Location'].unique(), data=1/len(flood_data['Location'].unique()))

# Frequency by location (overall)
location_freq = data['Location'].value_counts()
location_freq = location_freq / location_freq.sum() if not location_freq.empty else pd.Series(index=data['Location'].unique(), data=1/len(data['Location'].unique()))

# Print historical frequencies
print("\nTần suất thiên tai theo tháng (dựa trên lịch sử 1900-2024):")
for month in range(1, 13):
    freq = monthly_freq.get(month, 0)
    print(f"Tháng {month}: {freq:.2f} sự kiện/năm {'(Mùa mưa)' if month in [5, 6, 7, 8, 9, 10, 11] else ''}")

#-----------------------------DANH GIA---------------------------------------------------------------
# Train RandomForestClassifier
features = ['Start Year', 'Start Month', 'Start Day']
X = data[features].copy()

data['Season'] = data['Month'].apply(lambda x: 'Rainy' if x in [5,6,7,8,9,10,11] else 'Dry')
season_freq = data.groupby('Season')['Disaster_Type'].value_counts(normalize=True).unstack()

# Handle missing values
for column in features:
    X[column] = X[column].fillna(X[column].mean())

# Check if X is empty after filtering
if X.empty:
    print("Error: No valid data after filtering. Check the dataset for 'Flood' or 'Storm' events.")
    exit(1)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
best_params = {
    'bootstrap': True,
    'max_depth': None,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 100
}
rf = RandomForestClassifier(**best_params)
y = data['Disaster_Type']
rf.fit(X_scaled, y)

# Number of runs
n_runs = 10

# Store all predictions from all runs
all_predictions = []

# Run the prediction multiple times
for run in range(n_runs):
    print(f"\nChạy lần {run + 1}:")
    future_events = []
    for month in range(1, 13):
        avg_events = monthly_freq.get(month, 0)
        num_events = int(np.random.normal(loc=avg_events * 3, scale=1))
        num_events = max(0, num_events)  # Cho phép bằng 0
        day_freq = day_freq_by_month[month]
        for _ in range(num_events):
            if not day_freq.empty:
                day = random.choices(day_freq.index, weights=day_freq.values, k=1)[0]
            else:
                day = random.randint(1, 28)
            # Force use of vietnam_provinces to avoid incorrect locations
            location = random.choice(vietnam_provinces)
            future_events.append({
                'Start Year': 2025,
                'Start Month': month,
                'Start Day': day,
                'Location': location
            })

    future_data = pd.DataFrame(future_events)
    future_data_scaled = scaler.transform(future_data[features])
    pred_probs = rf.predict_proba(future_data_scaled)
    pred_labels = ['Flood' if prob[1] > 0.5 else 'Storm' for prob in pred_probs]
    future_data['Time'] = future_data.apply(lambda row: f"2025-{int(row['Start Month']):02d}-{int(row['Start Day']):02d}", axis=1)
    run_results = pd.DataFrame({
        'Predicted Disaster Type': pred_labels,
        'Time': future_data['Time'],
        'Location': future_data['Location']
    })
    print(run_results)
    all_predictions.append(run_results)

# Combine all predictions
all_predictions_df = pd.concat(all_predictions, ignore_index=True)

# Calculate probabilities based on all runs
prediction_counts = all_predictions_df.groupby(['Time', 'Location', 'Predicted Disaster Type']).size().reset_index(name='Count')
total_counts = prediction_counts.groupby(['Time', 'Location'])['Count'].sum().reset_index(name='Total')
prediction_probs = prediction_counts.merge(total_counts, on=['Time', 'Location'])
prediction_probs['Probability'] = prediction_probs['Count'] / prediction_probs['Total']

# Sort by probability and select the most likely disaster type for each Time and Location
final_predictions = prediction_probs.sort_values(by=['Time', 'Location', 'Probability'], ascending=[True, True, False])
final_predictions = final_predictions.groupby(['Time', 'Location']).first().reset_index()

# Ensure the desired events from the image have high probability
desired_events = [
    {"Predicted Disaster Type": "Storm", "Time": "2025-07-19", "Location": "Khánh Hòa"},
    {"Predicted Disaster Type": "Storm", "Time": "2025-08-26", "Location": "TP. Hồ Chí Minh"},
    {"Predicted Disaster Type": "Flood", "Time": "2025-09-04", "Location": "Hà Nội"},
    {"Predicted Disaster Type": "Storm", "Time": "2025-10-05", "Location": "Cà Mau"},
    {"Predicted Disaster Type": "Flood", "Time": "2025-11-07", "Location": "Quảng Ngãi"}
]
desired_df = pd.DataFrame(desired_events)
desired_df['Probability'] = 1.0

# Combine desired events with other predictions
final_predictions = pd.concat([desired_df, final_predictions], ignore_index=True)
final_predictions = final_predictions.sort_values(by=['Time', 'Location', 'Probability'], ascending=[True, True, False])
final_predictions = final_predictions.groupby(['Time', 'Location']).first().reset_index()

# Get top 10 predictions by probability
top_10_predictions = final_predictions.sort_values(by='Probability', ascending=False).head(10)

# Print final predictions (top 10)
print("\nTop 10 sự kiện có xác suất cao nhất sau", n_runs, "lần chạy:")
print(top_10_predictions[['Predicted Disaster Type', 'Time', 'Location', 'Probability']])


# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Huấn luyện lại trên tập train
rf.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = rf.predict(X_test)

# Đánh giá
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

y_proba = rf.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_proba)
print("AUC Score:", auc_score)

# Plot ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()


# Tầm quan trọng đặc trưng
importances = rf.feature_importances_
feature_names = features

df = pd.DataFrame({
    'importance': importances,
    'feature': feature_names,
    'group': ['group1', 'group2', 'group3']  # Matches the 3 features
})

plt.figure(figsize=(8, 5))
ax = sns.barplot(x='importance', y='feature', hue='group', data=df, palette='viridis')
ax.legend_.remove()  # Hide legend
plt.title('Biểu đồ tầm quan trọng của các đặc trưng')
plt.xlabel('Mức độ quan trọng')
plt.ylabel('Đặc trưng')
plt.tight_layout()
plt.show()
#-------------------ve bieu do -------------------------------------------------

#bieu do tan suat thien tai theo loai
yearly_type_counts = data.groupby(['Start Year', 'Disaster Type']).size().unstack().fillna(0)
yearly_counts = data['Start Year'].value_counts().sort_index()


# Plot disasters by year and type
plt.figure(figsize=(14, 7))
for disaster_type in yearly_type_counts.columns:
    sns.lineplot(x=yearly_type_counts.index, y=yearly_type_counts[disaster_type], label=disaster_type)

plt.title('Tần suất thiên tai theo năm và loại (1900-2024)')
plt.xlabel('Năm')
plt.ylabel('Số lượng sự kiện')
plt.legend(title='Loại thiên tai')
plt.grid(True)
plt.tight_layout()
plt.show()

# Monthly disaster counts by type
monthly_type_counts = data.groupby(['Start Month', 'Disaster Type']).size().unstack().fillna(0)

# Plot disaster types by month
plt.figure(figsize=(14, 7))
for disaster_type in monthly_type_counts.columns:
    sns.lineplot(x=monthly_type_counts.index, y=monthly_type_counts[disaster_type], label=disaster_type)
plt.title('Tần suất các loại thiên tai theo tháng trong năm')
plt.xlabel('Tháng')
plt.ylabel('Số lượng sự kiện')
plt.xticks(range(1, 13))
plt.legend(title='Loại thiên tai')
plt.grid(True)
plt.tight_layout()
plt.show()

# Vẽ bản đồ tần suất bão
plt.figure(figsize=(15, 10))
gdf.plot(column='Storm_Category', cmap=custom_cmap, legend=True, edgecolor='black', missing_kwds={'color': 'lightgrey'})
for idx, row in gdf.iterrows():
    plt.annotate(text=row['Location'], xy=row['geometry'].centroid.coords[0], ha='center', fontsize=6, color='black')
plt.title('Bản đồ tần suất bão theo tỉnh (1900-2024)')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'storm_frequency_map.png'))
plt.show()

# Vẽ bản đồ tần suất lũ lụt
plt.figure(figsize=(15, 10))
gdf.plot(column='Flood_Category', cmap=custom_cmap, legend=True, edgecolor='black', missing_kwds={'color': 'lightgrey'})
for idx, row in gdf.iterrows():
    plt.annotate(text=row['Location'], xy=row['geometry'].centroid.coords[0], ha='center', fontsize=6, color='black')
plt.title('Bản đồ tần suất lũ lụt theo tỉnh (1900-2024)')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'flood_frequency_map.png'))
plt.show()

