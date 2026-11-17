import pandas as pd
import numpy as np

# --------------------------------------------------
# Configuration
# --------------------------------------------------

file_path = "data/data/patient1/impaired_arm/01.parquet"

channels = [f"channel_{i:02d}" for i in range(1, 65)]

WINDOW_MS = 250
OVERLAP = 0.5

# Threshold for zero crossings and slope sign changes.
# We will initially use zero; later we can tune this.
ZC_THRESHOLD = 0.0
SSC_THRESHOLD = 0.0

def extract_features(signal, sampling_rate):
    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]

    if len(signal) < 2:
        return {}
    abs_signal = np.abs(signal)

    # Mean Absolute Value
    mav = np.mean(abs_signal)
    # Root Mean Square
    rms = np.sqrt(np.mean(signal ** 2))
    # Variance
    var = np.var(signal)
    # Standard Deviation
    std = np.std(signal)
    # Waveform Length
    wl = np.sum(np.abs(np.diff(signal)))
    # Zero Crossings
    zc = np.sum(
        ((signal[:-1] * signal[1:]) < 0) &
        (np.abs(signal[:-1] - signal[1:]) >= ZC_THRESHOLD)
    )
    # Slope Sign Changes
    diff_signal = np.diff(signal)
    ssc = np.sum(
        ((diff_signal[:-1] * diff_signal[1:]) < 0) &
        (np.abs(diff_signal[:-1]) >= SSC_THRESHOLD) &
        (np.abs(diff_signal[1:]) >= SSC_THRESHOLD)
    )
    # Integrated EMG
    iemg = np.sum(abs_signal)
    # Log Detector
    # Small epsilon avoids log(0).
    log_detector = np.exp(
        np.mean(np.log(abs_signal + 1e-10))
    )
    return {
        "MAV": mav,
        "RMS": rms,
        "VAR": var,
        "STD": std,
        "WL": wl,
        "ZC": zc,
        "SSC": ssc,
        "IEMG": iemg,
        "LOG": log_detector
    }

df = pd.read_parquet(file_path)
time_diffs = df["time"].diff().dropna()
sampling_rate = 1 / time_diffs.median()
print(f"Estimated sampling rate: {sampling_rate:.2f} Hz")

window_size = int(sampling_rate * WINDOW_MS / 1000)
step_size = int(window_size * (1 - OVERLAP))

rows = []

for gesture, gesture_df in df.groupby("movement_type"):
    gesture_df = gesture_df.sort_values("time")
    signal_data = gesture_df[channels].to_numpy()
    for start in range(
        0,
        len(signal_data) - window_size + 1,
        step_size
    ):
        window = signal_data[start:start + window_size]
        row = {}
        # Extract features independently for each channel
        for channel_idx, channel in enumerate(channels):
            channel_signal = window[:, channel_idx]
            features = extract_features(
                channel_signal,
                sampling_rate
            )
            for feature_name, value in features.items():
                row[f"{channel}_{feature_name}"] = value
        row["movement_type"] = gesture
        # Keep window metadata for later analysis
        row["window_start"] = gesture_df["time"].iloc[start]
        row["window_end"] = gesture_df["time"].iloc[start + window_size - 1]
        rows.append(row)


# Create tabular dataset

features_df = pd.DataFrame(rows)

print("\nExtracted dataset:")
print(features_df.shape)

print("\nClass distribution:")
print(features_df["movement_type"].value_counts())

print("\nPreview:")
print(features_df.iloc[:, :12].head())

# Save dataset
features_df.to_parquet(
    "data/patient1_impaired_arm_features.parquet",
    index=False
)

features_df.to_csv(
    "data/patient1_impaired_arm_features.csv",
    index=False
)
