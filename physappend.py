import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import pandas as pd
import math
import os

destination_path = '/Users/stevenhu/Desktop/vibratophysdepth/Cello/CelloPhysDepth.csv'

audio_path = '/Users/stevenhu/Desktop/vibratophysdepth/Cello/Audio Recordings/mrSchumann/mrCSharp4.wav'
y, sr = librosa.load(audio_path, sr=44100)

f_string = 165.4

def analyze_f0_yin(
    y,
    sr,
    hop_length=16,
    fmin=275,
    fmax=290,
    savgol_window_max=201,
    savgol_polyorder=3,
    peak_distance=290,
    peak_prominence=1.5,
    plot=False
):
    f0_yin = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    times = librosa.times_like(f0_yin, sr=sr, hop_length=hop_length)
    window_length = min(savgol_window_max, len(f0_yin))
    if window_length % 2 == 0:
        window_length -= 1
    if window_length <= savgol_polyorder:
        window_length = savgol_polyorder + 2 if (savgol_polyorder + 2) % 2 == 1 else savgol_polyorder + 3
    f0_yin_interp = np.copy(f0_yin)
    nans = np.isnan(f0_yin_interp)
    if np.any(~nans) and np.any(nans):
        f0_yin_interp[nans] = np.interp(
            np.flatnonzero(nans), np.flatnonzero(~nans), f0_yin_interp[~nans]
        )
    f0_smooth = savgol_filter(f0_yin_interp, window_length=window_length, polyorder=savgol_polyorder)
    peaks_indices, _ = find_peaks(f0_smooth, distance=peak_distance, prominence=peak_prominence)
    troughs_indices, _ = find_peaks(-f0_smooth, distance=peak_distance, prominence=peak_prominence)
    def get_midpoints(events):
        midpoints = []
        for i in range(len(events)-1):
            if events[i,2] != events[i+1,2]:
                t_mid = 0.5 * events[i,0] + 0.5 * events[i+1,0]
                v_mid = 0.5 * events[i,1] + 0.5 * events[i+1,1]
                midpoints.append([t_mid, v_mid])
        return np.array(midpoints)
    peak_times = times[peaks_indices]
    trough_times = times[troughs_indices]
    peak_vals = f0_smooth[peaks_indices]
    trough_vals = f0_smooth[troughs_indices]
    events = np.concatenate([
        np.stack([peak_times, peak_vals, np.ones_like(peak_times)], axis=1),
        np.stack([trough_times, trough_vals, np.zeros_like(trough_times)], axis=1)
    ])
    events = events[events[:,0].argsort()]
    midpoints = get_midpoints(events)
    all_indices = np.concatenate([peaks_indices, troughs_indices])
    all_times = times[all_indices]
    all_f0s = f0_smooth[all_indices]
    if midpoints.size == 0:
        average_distance = np.nan
    else:
        trend_start = midpoints[:,0].min()
        trend_end = midpoints[:,0].max()
        mask = (all_times >= trend_start) & (all_times <= trend_end)
        relevant_times = all_times[mask]
        relevant_f0s = all_f0s[mask]
        trend_f0s = np.interp(relevant_times, midpoints[:,0], midpoints[:,1])
        vertical_distances = np.abs(relevant_f0s - trend_f0s)
        average_distance = vertical_distances.mean()
    return {
        "f0_yin": f0_yin,
        "f0_smooth": f0_smooth,
        "times": times,
        "peaks_indices": peaks_indices,
        "troughs_indices": troughs_indices,
        "midpoints": midpoints,
        "average_distance": average_distance,
        "vibrato_center_numerical": np.nanmean(midpoints[:,1]) if midpoints.size > 0 else np.nan
    }

f0 = analyze_f0_yin(y, sr, plot=False)

average_hz = float(f0['average_distance'])
average_cents = float(1200 * np.log2((441 + average_hz) / 441))
vibrato_center_numerical = float(f0['vibrato_center_numerical'])
physical_depth = float(f_string/(2*average_hz)*(math.sqrt(1+4*average_hz**2/vibrato_center_numerical**2)-1))
physical_position = float(f_string/vibrato_center_numerical)

fig, ax = plt.subplots(figsize=(12, 12))

# 1. F0 (YIN)
f0_cents = 1200 * np.log2(f0['f0_yin'] / 441)
f0_smooth_cents = 1200 * np.log2(f0['f0_smooth'] / 441)
midpoints_cents = None
if f0['midpoints'].size > 0:
    midpoints_cents = np.copy(f0['midpoints'])
    midpoints_cents[:,1] = 1200 * np.log2(midpoints_cents[:,1] / 441)

ax.plot(f0['times'], f0_cents, label='Raw YIN F0', alpha=0.5)
ax.plot(f0['times'], f0_smooth_cents, label='Smoothed', linewidth=2)
ax.plot(f0['times'][f0['peaks_indices']], f0_smooth_cents[f0['peaks_indices']], 'ro', label='Peaks')
ax.plot(f0['times'][f0['troughs_indices']], f0_smooth_cents[f0['troughs_indices']], 'go', label='Troughs')
if f0['midpoints'].size > 0:
    ax.scatter(midpoints_cents[:,0], midpoints_cents[:,1], color='purple', marker='x', label='Midpoints')
    sorted_indices = np.argsort(midpoints_cents[:,0])
    ax.plot(midpoints_cents[sorted_indices,0], midpoints_cents[sorted_indices,1], color='purple', linestyle='-', label='Midpoint Trend')
ax.set_ylabel("F0 (cents rel. A=441 Hz)")
ax.legend()
ax.set_title("Pitch (YIN, cents rel. 441 Hz)")

audio_filename = audio_path.split('/')[-1]
data = {
    'Audio File': [audio_filename],
    'Average Depth of Pitch Variation (cents)': [average_cents],
    'Average Depth of Pitch Variation (Hz)': [average_hz],
    'Tonal Center of Vibrato (Hz)': [vibrato_center_numerical],
    'Physical Depth of Vibrato': [physical_depth],
    'Physical Center of Vibrato': [physical_position],
    'String Frequency': [f_string]
}

df = pd.DataFrame(data)
numeric_cols = [
    'Average Depth of Pitch Variation (cents)',
    'Average Depth of Pitch Variation (Hz)',
    'Tonal Center of Vibrato (Hz)',
    'Physical Depth of Vibrato',
    'Physical Center of Vibrato',
    'String Frequency'
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

csv_path = destination_path

if os.path.isfile(csv_path):
    try:
        existing_df = pd.read_csv(csv_path)
        for col in numeric_cols:
            existing_df[col] = pd.to_numeric(existing_df[col], errors='coerce')
        df = pd.concat([existing_df, df], ignore_index=True)
    except pd.errors.EmptyDataError:
        pass

df.to_csv(csv_path, index=False, float_format='%.10g')