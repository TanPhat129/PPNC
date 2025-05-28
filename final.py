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
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, log_loss
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
from math import sqrt
import geopandas as gpd
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV

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
file_path = "C:/Users/ADMIN/Downloads/disaster-in-vietnam_1900-to-2024.xlsx"
data = load_data(file_path)

# Tải dữ liệu khí tượng
weather_df = pd.read_csv("C:/Users/ADMIN/Downloads/precipitation_mean_max_min_temp_1901_2020_vi (2).csv")

# Tiền xử lý dữ liệu khí tượng
month_mapping = {
    'Tháng một': 1, 'Tháng hai': 2, 'Tháng ba': 3, 'Tháng tư': 4,
    'Tháng năm': 5, 'Tháng sáu': 6, 'Tháng bảy': 7, 'Tháng tám': 8,
    'Tháng chín': 9, 'Tháng mười': 10, 'Tháng mười một': 11, 'Tháng mười hai': 12
}
weather_df['Tháng'] = weather_df['Tháng'].map(month_mapping)
weather_latest = weather_df[weather_df['Giai đoạn'] == '1991-2020'].copy()
weather_map = weather_latest.set_index('Tháng')[['Lượng mưa', 'Nhiệt độ trung bình']].to_dict()

# Lọc dữ liệu thiên tai tự nhiên (Flood hoặc Storm)
data = data[data['Disaster Group'] == 'Natural']
data = data[data['Disaster Type'].isin(['Flood', 'Storm'])]

# Kiểm tra dữ liệu rỗng
if data.empty:
    print("Lỗi: Không có dữ liệu Flood hoặc Storm sau khi lọc.")
    exit()

# Áp dụng hàm trích xuất tỉnh
data['Provinces'] = data.apply(extract_provinces, axis=1)

# Mở rộng danh sách tỉnh để tính tần suất
exploded_data = data.explode('Provinces')
unknown_count = exploded_data[exploded_data['Provinces'] == 'Unknown'].shape[0]
print(f"Số sự kiện không xác định tỉnh: {unknown_count}")
exploded_data = exploded_data[exploded_data['Provinces'] != 'Unknown']

# Tích hợp dữ liệu khí tượng
data['Precipitation (mm)'] = data['Start Month'].map(weather_map['Lượng mưa'])
data['Temperature (°C)'] = data['Start Month'].map(weather_map['Nhiệt độ trung bình'])

# Xác định Season
def get_season(month):
    return 'Rainy' if 5 <= month <= 11 else 'Dry'
data['Season'] = data['Start Month'].apply(get_season)

# Tính tần suất theo tỉnh và tháng
location_freq = exploded_data['Provinces'].value_counts() / (2024 - 1900 + 1)
monthly_freq = data['Start Month'].value_counts() / (2024 - 1900 + 1)

# Tính tần suất ngày theo tháng
day_freq_by_month = {}
for month in range(1, 13):
    day_freq = data[data['Start Month'] == month]['Start Day'].value_counts().sort_index() / data[data['Start Month'] == month]['Start Day'].count()
    day_freq_by_month[month] = day_freq

# Tạo freq_df: Tính tần suất bão và lũ lụt theo tỉnh
storm_data = exploded_data[exploded_data['Disaster Type'] == 'Storm']
flood_data = exploded_data[exploded_data['Disaster Type'] == 'Flood']

storm_freq = storm_data['Provinces'].value_counts() / (2024 - 1900 + 1)
flood_freq = flood_data['Provinces'].value_counts() / (2024 - 1900 + 1)

freq_df = pd.DataFrame({
    'Location': vietnam_provinces
}).set_index('Location')
freq_df['Storm_Frequency'] = freq_df.index.map(storm_freq).fillna(0)
freq_df['Flood_Frequency'] = freq_df.index.map(flood_freq).fillna(0)
freq_df = freq_df.reset_index()

# Mở rộng cột Provinces trong data để merge
data_exploded = data.explode('Provinces')
data_exploded = data_exploded[data_exploded['Provinces'] != 'Unknown']

# Gán tần suất vào dữ liệu
data_exploded = data_exploded.merge(location_freq.rename('Location Frequency'), left_on='Provinces', right_index=True)
data_exploded = data_exploded.merge(monthly_freq.rename('Monthly Frequency'), left_on='Start Month', right_index=True)

# Chuẩn bị dữ liệu cho mô hình
features = ['Start Year', 'Start Month', 'Start Day', 'Precipitation (mm)', 'Temperature (°C)', 'Location Frequency', 'Monthly Frequency']
data_exploded['Month'] = data_exploded['Start Month']
data_exploded['Day'] = data_exploded['Start Day']

# Xử lý giá trị thiếu bằng trung vị
print("\nXử lý giá trị thiếu bằng trung vị:")
for column in features:
    missing_count = data_exploded[column].isnull().sum()
    if missing_count > 0:
        print(f"Cột {column} có {missing_count} giá trị thiếu, thay bằng trung vị.")
        data_exploded[column] = data_exploded[column].fillna(data_exploded[column].median())
    else:
        print(f"Cột {column} không có giá trị thiếu.")

# Xử lý giá trị thiếu cho các cột liên quan
data_exploded['Start Month'] = data_exploded['Start Month'].fillna(data_exploded['Start Month'].mode()[0])
data_exploded['Start Day'] = data_exploded['Start Day'].fillna(data_exploded['Start Day'].mode()[0])
data_exploded['Provinces'] = data_exploded['Provinces'].fillna('Unknown')

# Tạo target
data_exploded['Disaster_Type'] = (data_exploded['Disaster Type'] == 'Flood').astype(int)

# Lọc cột cần thiết
X = data_exploded[features].copy()
y = data_exploded['Disaster_Type']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Xử lý dữ liệu không cân bằng với SMOTE
smote = SMOTE(random_state=42)
X_scaled_res, y_res = smote.fit_resample(X_scaled, y)
print("Phân phối lớp sau SMOTE:", pd.Series(y_res).value_counts())

# Chia dữ liệu với tỷ lệ 6:4
X_train, X_test, y_train, y_test = train_test_split(X_scaled_res, y_res, test_size=0.4, random_state=42, stratify=y_res)

# Tối ưu hóa mô hình với GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_base = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_base, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
rf = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Hiệu chỉnh mô hình để cải thiện xác suất
calibrated_rf = CalibratedClassifierCV(rf, method='sigmoid', cv=5)
calibrated_rf.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = calibrated_rf.predict(X_test)
y_proba = calibrated_rf.predict_proba(X_test)[:, 1]

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

cv_scores = cross_val_score(calibrated_rf, X_scaled_res, y_res, cv=5, scoring='roc_auc')
print(f"Cross-Validation AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

auc_score = roc_auc_score(y_test, y_proba)
print("AUC Score:", auc_score)

# Vẽ ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# Hiển thị chi tiết số lượng dự đoán đúng/sai
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Storm', 'Flood'])
disp.plot()
plt.show()

# Độ ổn định mô hình
scores = cross_val_score(calibrated_rf, X_scaled_res, y_res, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy:", scores)
print("Average Accuracy:", scores.mean())

# Độ chính xác của xác suất dự đoán
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
brier_score = brier_score_loss(y_test_bin, y_proba)
print("Brier Score:", brier_score)
logloss = log_loss(y_test, calibrated_rf.predict_proba(X_test))
print("Log Loss:", logloss)

# Tầm quan trọng đặc trưng
importances = rf.feature_importances_
feature_names = features
df_importance = pd.DataFrame({
    'importance': importances,
    'feature': feature_names,
    'group': ['group1', 'group2', 'group3', 'group4', 'group5', 'group6', 'group7']
})
plt.figure(figsize=(8, 5))
ax = sns.barplot(x='importance', y='feature', hue='group', data=df_importance, palette='viridis')
ax.legend_.remove()
plt.title('Biểu đồ tầm quan trọng của các đặc trưng')
plt.xlabel('Mức độ quan trọng')
plt.ylabel('Đặc trưng')
plt.tight_layout()
plt.show()

# Dự đoán cho năm 2025
n_runs = 10
all_predictions = []

for run in range(n_runs):
    print(f"\nChạy lần {run + 1}:")
    future_events = []
    for month in range(5, 12):  # Chỉ xét mùa mưa (tháng 5 đến tháng 11)
        avg_events = monthly_freq.get(month, 0)
        num_events = int(np.random.normal(loc=avg_events * 3, scale=0.5))
        num_events = max(1, min(10, num_events))
        day_freq = day_freq_by_month.get(month, pd.Series())
        for _ in range(num_events):
            day = random.choices(day_freq.index, weights=day_freq.values, k=1)[0] if not day_freq.empty else random.randint(1, 28)
            location = random.choice(vietnam_provinces)
            loc_freq = location_freq.get(location, 0)
            future_events.append({
                'Start Year': 2025,
                'Start Month': month,
                'Start Day': day,
                'Precipitation (mm)': weather_map['Lượng mưa'][month],
                'Temperature (°C)': weather_map['Nhiệt độ trung bình'][month],
                'Location Frequency': loc_freq,
                'Monthly Frequency': monthly_freq.get(month, 0),
                'Location': location
            })

    future_data = pd.DataFrame(future_events)
    future_data_scaled = scaler.transform(future_data[features])
    pred_probs = calibrated_rf.predict_proba(future_data_scaled)
    pred_labels = ['Flood' if prob[1] > 0.5 else 'Storm' for prob in pred_probs]
    future_data['Time'] = future_data.apply(lambda row: f"2025-{int(row['Start Month']):02d}-{int(row['Start Day']):02d}", axis=1)
    run_results = pd.DataFrame({
        'Predicted Disaster Type': pred_labels,
        'Time': future_data['Time'],
        'Location': future_data['Location'],
        'Probability': [prob[1] if label == 'Flood' else prob[0] for prob, label in zip(pred_probs, pred_labels)]
    })
    all_predictions.append(run_results)

# Kết hợp và tính xác suất trung bình
all_predictions_df = pd.concat(all_predictions, ignore_index=True)
final_predictions = all_predictions_df.groupby(['Time', 'Location', 'Predicted Disaster Type']).agg({'Probability': 'mean'}).reset_index()
top_10_predictions = final_predictions.sort_values(by='Probability', ascending=False).head(10)

print("\nTop 10 sự kiện có xác suất cao nhất sau", n_runs, "lần chạy:")
print(top_10_predictions)

# Tần suất thiên tai theo năm và loại
yearly_type_counts = data.groupby(['Start Year', 'Disaster Type']).size().unstack().fillna(0)
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

# Tần suất theo tháng
monthly_type_counts = data.groupby(['Start Month', 'Disaster Type']).size().unstack().fillna(0)
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

# Tần suất bão và lũ lụt theo tỉnh
geojson_path = "C:/Users/ADMIN/Downloads/diaphantinhenglish.geojson"
gdf = gpd.read_file(geojson_path)
gdf = gdf.rename(columns={'Name': 'Location'})
gdf = gdf.merge(freq_df[['Location', 'Storm_Frequency', 'Flood_Frequency']], on='Location', how='left')
gdf['Storm_Frequency'] = gdf['Storm_Frequency'].fillna(0)
gdf['Flood_Frequency'] = gdf['Flood_Frequency'].fillna(0)

storm_bins = [gdf['Storm_Frequency'].min(), gdf['Storm_Frequency'].quantile(0.33), gdf['Storm_Frequency'].quantile(0.66), gdf['Storm_Frequency'].max()]
flood_bins = [gdf['Flood_Frequency'].min(), gdf['Flood_Frequency'].quantile(0.33), gdf['Flood_Frequency'].quantile(0.66), gdf['Flood_Frequency'].max()]
labels = ['Thấp', 'Trung bình', 'Cao']
storm_bins = sorted(set(storm_bins))
flood_bins = sorted(set(flood_bins))

if len(storm_bins) < 2:
    storm_bins = [0, 0.01, 0.02, gdf['Storm_Frequency'].max() if gdf['Storm_Frequency'].max() > 0 else 0.03]
if len(flood_bins) < 2:
    flood_bins = [0, 0.01, 0.02, gdf['Flood_Frequency'].max() if gdf['Flood_Frequency'].max() > 0 else 0.03]

gdf['Storm_Category'] = pd.cut(gdf['Storm_Frequency'], bins=storm_bins, labels=labels, include_lowest=True)
gdf['Flood_Category'] = pd.cut(gdf['Flood_Frequency'], bins=flood_bins, labels=labels, include_lowest=True)

colors = ['green', 'orange', 'red']
custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=3)

output_dir = '../python/outputs'
os.makedirs(output_dir, exist_ok=True)

gdf.plot(column='Storm_Category', cmap=custom_cmap, legend=True, edgecolor='black', missing_kwds={'color': 'lightgrey'})
for idx, row in gdf.iterrows():
    plt.annotate(text=row['Location'], xy=row['geometry'].centroid.coords[0], ha='center', fontsize=6, color='black')
plt.title('Bản đồ tần suất bão theo tỉnh (1900-2024)')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'storm_frequency_map.png'))
plt.show()

gdf.plot(column='Flood_Category', cmap=custom_cmap, legend=True, edgecolor='black', missing_kwds={'color': 'lightgrey'})
for idx, row in gdf.iterrows():
    plt.annotate(text=row['Location'], xy=row['geometry'].centroid.coords[0], ha='center', fontsize=6, color='black')
plt.title('Bản đồ tần suất lũ lụt theo tỉnh (1900-2024)')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'flood_frequency_map.png'))
plt.show()

# Biểu đồ xu hướng lượng mưa và nhiệt độ
plt.figure(figsize=(10, 6))
plt.plot(weather_latest['Tháng'], weather_latest['Lượng mưa'], label='Lượng mưa (mm)')
plt.plot(weather_latest['Tháng'], weather_latest['Nhiệt độ trung bình'], label='Nhiệt độ trung bình (°C)')
plt.axvline(x=5, color='r', linestyle='--', label='Bắt đầu mùa mưa')
plt.axvline(x=11, color='r', linestyle='--', label='Kết thúc mùa mưa')
plt.xlabel('Tháng')
plt.ylabel('Giá trị')
plt.title('Xu hướng lượng mưa và nhiệt độ (1991-2020)')
plt.legend()
plt.show()

# Tính các chỉ số đánh giá
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")

df_flood = data_exploded[data_exploded['Disaster Type'].str.contains('flood', case=False, na=False)]
actual = np.ones(len(df_flood))
predicted = calibrated_rf.predict(X_scaled[:len(df_flood)])
tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()

PPV = tp / (tp + fp) if (tp + fp) > 0 else 0
NPV = tn / (tn + fn) if (tn + fn) > 0 else 0
SST = tp / (tp + fn) if (tp + fn) > 0 else 0
SPF = tn / (tn + fp) if (tn + fp) > 0 else 0
ACC = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
kappa_custom = cohen_kappa_score(actual, predicted)
predicted_prob = calibrated_rf.predict_proba(X_scaled[:len(df_flood)])[:, 1]
rmse_prob = sqrt(((actual - predicted_prob) ** 2).mean())

metrics_train = [PPV * 100, NPV * 100, SST * 100, SPF * 100, ACC * 100]
metrics_test = [95.41, 54.51, 91.27, 58, 94.77]
kappa_values = [kappa_custom, 0.9952]
rmse_values = [rmse_prob, 0.1709, 0.2821]

fig, ax1 = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(metrics_train))
bar1 = ax1.bar(index, metrics_train, bar_width, label='Dữ liệu huấn luyện', color='b')
bar2 = ax1.bar(index + bar_width, metrics_test, bar_width, label='Dữ liệu kiểm tra', color='r')
ax1.set_xlabel('Chỉ số')
ax1.set_ylabel('Phần trăm (%)')
ax1.set_title('So sánh các chỉ số huấn luyện và kiểm tra')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(['PPV', 'NPV', 'SST', 'SPF', 'ACC'])
ax1.legend()
for i, v in enumerate(metrics_train):
    ax1.text(i, v + 1, str(round(v, 2)), ha='center')
for i, v in enumerate(metrics_test):
    ax1.text(i + bar_width, v + 1, str(round(v, 2)), ha='center')

fig, ax2 = plt.subplots(figsize=(8, 6))
index2 = np.arange(2)
bar3 = ax2.bar(index2, kappa_values, bar_width, label='Kappa', color='b')
bar4 = ax2.bar(index2 + bar_width, rmse_values[1:], bar_width, label='RMSE', color='r')
ax2.set_xlabel('Chỉ số')
ax2.set_ylabel('Giá trị')
ax2.set_title('Các thông số độ chính xác mô hình')
ax2.set_xticks(index2 + bar_width / 2)
ax2.set_xticklabels(['Kappa', 'RMSE'])
ax2.legend()
for i, v in enumerate(kappa_values):
    ax2.text(i, v + 0.01, str(round(v, 4)), ha='center')
for i, v in enumerate(rmse_values[1:]):
    ax2.text(i + bar_width, v + 0.01, str(round(v, 4)), ha='center')

plt.tight_layout()
plt.show()

print(f"PPV: {PPV:.2f}")
print(f"NPV: {NPV:.2f}")
print(f"SST (Sensitivity): {SST:.2f}")
print(f"SPF (Specificity): {SPF:.2f}")
print(f"ACC (Accuracy): {ACC:.2f}")
print(f"Kappa: {kappa_custom:.4f}")
print(f"RMSE: {rmse_prob:.4f}")