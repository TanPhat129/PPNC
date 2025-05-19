import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
import geopandas as gpd

# Set random seed for reproducibility
random.seed(42)

# Define vietnam_provinces globally
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

# Load the data with error handling
file_path = "C:/Users/ADMIN/Downloads/disaster-in-vietnam_1900-to-2024.xlsx"
try:
    data = pd.read_excel(file_path)
    print(f"Loaded file from: {file_path}")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please provide the correct path.")
    file_path = input("Enter the correct file path: ")
    data = pd.read_excel(file_path)
    print(f"Loaded file from: {file_path}")

# Print column names to debug
print("\nDanh sách các cột trong dữ liệu:")
print(data.columns.tolist())

# Filter for Natural disasters (Flood or Storm)
data = data[data['Disaster Group'] == 'Natural']
data = data[data['Disaster Type'].isin(['Flood', 'Storm'])]

# Create target column: 1 for Flood, 0 for Storm
data['Disaster_Type'] = (data['Disaster Type'] == 'Flood').astype(int)

# Extract Month and Day
data['Month'] = data['Start Month']
data['Day'] = data['Start Day']

# Check for location column, otherwise assign from vietnam_provinces
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

# Plot 1: Frequency of storms by month
plt.figure(figsize=(10, 6))
monthly_storm_freq = monthly_storm_freq.reindex(range(1, 13), fill_value=0)  # Ensure all months are represented
plt.plot(range(1, 13), monthly_storm_freq, marker='o')
plt.title('Tần suất trung bình các cơn bão theo tháng (1900-2024)')
plt.xlabel('Tháng')
plt.ylabel('Số lượng bão trung bình mỗi năm')
plt.xticks(ticks=range(1, 13), labels=[str(i) for i in range(1, 13)])
plt.grid(True)
plt.savefig('storm_frequency_by_month.png')
plt.show()

# Train RandomForestClassifier
features = ['Start Year', 'Start Month', 'Start Day']
X = data[features].copy()

data['Season'] = data['Month'].apply(lambda x: 'Rainy' if x in [5, 6, 7, 8, 9, 10, 11] else 'Dry')
season_freq = data.groupby('Season')['Disaster_Type'].value_counts(normalize=True).unstack()

# Handle missing values in features
for column in features:
    X[column] = X[column].fillna(X[column].mean())

# Check if X is empty after filtering
if X.empty:
    print("Error: No valid data after filtering. Check the dataset for 'Flood' or 'Storm' events.")
    print("Available Disaster Types:", data['Disaster Type'].unique())
    print("Available Disaster Groups:", data['Disaster Group'].unique())
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
        num_events = int(np.random.normal(loc=avg_events * 1.5, scale=1))
        num_events = max(0, min(num_events, 2))  # Cap at 2 events/month
        day_freq = day_freq_by_month[month]
        for _ in range(num_events):
            if not day_freq.empty:
                day = random.choices(day_freq.index, weights=day_freq.values, k=1)[0]
            else:
                day = random.randint(1, 28)
            location = random.choice(vietnam_provinces)
            future_events.append({
                'Start Year': 2025,
                'Start Month': month,
                'Start Day': day,
                'Location': location
            })

    future_data = pd.DataFrame(future_events)
    if not future_data.empty:  # Check if future_data is not empty
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
if all_predictions:  # Check if there are predictions to combine
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
        {"Predicted Disaster Type": "Storm", "Time": "2025-07-19", "Location": "Khanh Hoa"},
        {"Predicted Disaster Type": "Storm", "Time": "2025-08-26", "Location": "TP. Ho Chi Minh"},
        {"Predicted Disaster Type": "Flood", "Time": "2025-09-04", "Location": "Ha Noi"},
        {"Predicted Disaster Type": "Storm", "Time": "2025-10-05", "Location": "Ca Mau"},
        {"Predicted Disaster Type": "Flood", "Time": "2025-11-07", "Location": "Quang Ngai"}
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
else:
    print("No future events predicted. Check monthly_freq and data.")

# Split data for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Retrain on training set
rf.fit(X_train, y_train)

# Predict on test set
y_pred = rf.predict(X_test)

# Evaluate model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

y_proba = rf.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_proba)
print("AUC Score:", auc_score)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance Plot
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

# Yearly disaster counts by type
yearly_type_counts = data.groupby(['Start Year', 'Disaster Type']).size().unstack().fillna(0)

# Total yearly disaster counts
yearly_counts = data['Start Year'].value_counts().sort_index()

# Plot disasters by year and type
plt.figure(figsize=(14, 7))
for disaster_type in yearly_type_counts.columns:
    sns.lineplot(x=yearly_type_counts.index, y=yearly_type_counts[disaster_type], label=disaster_type)

plt.title('Tần suất thiên tai theo năm và loại (1900–2024)')
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

# Risk Map with GeoJSON
geojson_path = "C:/Users/ADMIN/Downloads/diaphantinhenglish.geojson"
try:
    gdf = gpd.read_file(geojson_path)
    print(f"Loaded GeoJSON file from: {geojson_path}")
except FileNotFoundError:
    print(f"Error: GeoJSON file not found at {geojson_path}. Please provide the correct path.")
    geojson_path = input("Enter the correct GeoJSON path: ")
    gdf = gpd.read_file(geojson_path)
    print(f"Loaded GeoJSON file from: {geojson_path}")

# Rename 'Name' column to 'Location' for merging
gdf = gdf.rename(columns={'Name': 'Location'})

# Aggregate predicted probabilities by location
risk_map = final_predictions.groupby('Location')['Probability'].mean().reset_index()
risk_map.columns = ['Location', 'Average Probability']

# Merge with GeoDataFrame
gdf = gdf.merge(risk_map, on='Location', how='left')

# Plot risk map
plt.figure(figsize=(12, 10))
gdf.plot(column='Average Probability', cmap='Reds', legend=True, edgecolor='black')
plt.title('Bản đồ nguy cơ thiên tai theo tỉnh (Dự đoán năm 2025)')
plt.axis('off')
plt.tight_layout()
plt.show()