import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "data/data/patient1/impaired_arm/01.parquet"

df = pd.read_parquet(file_path)

print(df.head())
print(df.info())
print(df["movement_type"].value_counts())

print("Available gestures:")
print(df["movement_type"].unique())

print("\nNumber of samples per gesture:")
print(df.groupby("movement_type").size())

channel = "channel_01"
gesture = df["movement_type"].dropna().iloc[0]
segment = df[df["movement_type"] == gesture]
plt.figure(figsize=(14, 4))
plt.plot(segment["time"],segment[channel])
plt.title(f"{gesture} - {channel}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.grid()
plt.show()

channels = [
    f"channel_{i:02d}"
    for i in range(1, 65)
]
fig, axes = plt.subplots(
    8, 8,
    figsize=(18, 14)
)
for i, ax in enumerate(axes.flat):
    ax.plot(segment["time"], segment[channels[i]])
    ax.set_title(channels[i], fontsize=8)
    ax.tick_params(labelsize=6)
plt.tight_layout()
plt.show()
