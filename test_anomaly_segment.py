import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# === CONFIGURATION ===
csv_path = "results/prediction_patchtst_3.21.csv"             # 📝 Replace with your CSV file
timestamp_col = "ds"                   # Name of the timestamp column
segment_length = 12

# Column indexes (after removing timestamp):
y_col = 0
y_pred_col = 5
lo_95_col = 7
hi_95_col = 8

output_dir = "results/anomaly_segemnts_interval90"
os.makedirs(output_dir, exist_ok=True)

# === 1. Load CSV safely ===
df = pd.read_csv(csv_path)

# ✅ Convert timestamp column to datetime
df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')

# ✅ Drop rows with invalid timestamps or any missing values
df = df.dropna()

# === 2. Extract features and timestamps ===
timestamps = df[timestamp_col].values
features = df.drop(columns=[timestamp_col]).values  # Drop timestamp for now

# === 3. Create non-overlapping segments ===
num_segments = len(features) // segment_length
features = features[:num_segments * segment_length]
timestamps = timestamps[:num_segments * segment_length]

feature_segments = features.reshape(num_segments, segment_length, -1)
time_segments = timestamps.reshape(num_segments, segment_length)

# === 4. Re-insert timestamps into segments (as column 1) ===
segments = []
for feat_seg, time_seg in zip(feature_segments, time_segments):
    seg_with_time = np.insert(feat_seg, 1, time_seg, axis=1)  # Insert timestamp at column index 1
    segments.append(seg_with_time)

# === 5. Group segments by date ===
segments_by_date = defaultdict(list)
for segment in segments:
    ts = pd.to_datetime(segment[0, 1])
    date_key = ts.date()
    segments_by_date[date_key].append(segment)

# === 6. Detect anomaly days ===
dates_with_anomalies = set()
for segment in segments:
    for row in segment:
        y, lo, hi = row[y_col], row[lo_95_col], row[hi_95_col]
        if y < lo or y > hi:
            ts = pd.to_datetime(row[1])
            dates_with_anomalies.add(ts.date())
            break  # Only need to detect one anomaly per segment

# === 7. Plot all segments in each anomaly day ===
# === 7. Plot all segments in each anomaly day ===
for date_key in sorted(dates_with_anomalies):
    day_segments = segments_by_date[date_key]

    plt.figure(figsize=(14, 5))

    # Dùng cờ để chỉ label 1 lần
    has_labeled_true = False
    has_labeled_pred = False
    has_labeled_band = False
    has_labeled_anomaly = False

    for segment in day_segments:
        x = pd.to_datetime(segment[:, 1])              # Timestamp
        y_true = segment[:, y_col].astype(np.float32)
        y_pred = segment[:, y_pred_col].astype(np.float32)
        y_lo = segment[:, lo_95_col].astype(np.float32)
        y_hi = segment[:, hi_95_col].astype(np.float32)

        # Vẽ đường giá trị thực
        plt.plot(
            x, y_true,
            color='blue',
            label='Giá trị thực' if not has_labeled_true else None
        )
        has_labeled_true = True

        # Vẽ đường dự đoán
        plt.plot(
            x, y_pred,
            color='orange',
            label='Dự đoán' if not has_labeled_pred else None
        )
        has_labeled_pred = True

        # Tô vùng dự đoán
        plt.fill_between(
            x, y_lo, y_hi,
            color='gray', alpha=0.2,
            label='Khoảng dự đoán' if not has_labeled_band else None
        )
        has_labeled_band = True

        # Tô vùng bất thường
        anomalies = (y_true < y_lo) | (y_true > y_hi)
        if np.any(anomalies):
            plt.fill_between(
                x, y_true, y_pred,
                where=anomalies,
                color='red', alpha=0.5,
                label='Bất thường' if not has_labeled_anomaly else None
            )
            has_labeled_anomaly = True
        anomaly_times = pd.to_datetime(segment[anomalies, 1])
        for t in anomaly_times:
            plt.axvline(x=t, color='red', linestyle='--', alpha=0.5)

    plt.title(f"Ngày có bất thường: {date_key}")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá trị")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(output_dir, f"day_{date_key}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved anomaly plot for {date_key} → {save_path}")
