#%%
import os, h5py
import matplotlib.pyplot as plt
import numpy as np
dir_h5 = "Y:\\python\\leapmotion_handtracking\\data"
fname_h5 = "leap_recording_trigger_20260203_133117_corrected.h5"
path_h5 = os.path.join(dir_h5, fname_h5)

f= h5py.File(path_h5, 'r')
keys = f.keys()
print(keys)
type(f['leap_timestamp'])
arduino_sync = f['arduino_sync']
arduino_trigger_times_us = f['arduino_trigger_times_us']
leap_timestamp = f['leap_timestamp'][:]
leap_timestamp_corrected = f['leap_timestamp_corrected'][:]
trigger_status = f['trigger_status'][:]
system_timestamp = f['system_timestamp'][:]
trigger_onset_times = f['trigger_onset_times'][:]
trigger_onset_times_corrected = f['trigger_onset_times_corrected'][:]
idxs_onset = np.where(np.diff(trigger_status) == 1)
plt.ion()
#%% leap_timestampとsystem_timestampの差
system_time_elapsed = system_timestamp - system_timestamp[0]
leap_time_elapsed = leap_timestamp*10**-3 - leap_timestamp[0]*10**-3
diff_timestamp = leap_time_elapsed - system_time_elapsed*10**3
diff_timestamp_corrected = leap_timestamp_corrected - system_timestamp
#%% leap_timestamp_corrected と system_timestampの差
plt.figure()
plt.plot(system_time_elapsed, diff_timestamp_corrected)
#%% leap_timestamp_correctedとtrigger_onset_times_correctedの比較
#% Trigger vs Leap Timestamp Comparison

# 各トリガーに最も近いフレームを探す
diffs = []
for i, trig_time in enumerate(trigger_onset_times_corrected):
    idx = np.argmin(np.abs(leap_timestamp_corrected - trig_time))
    diff_ms = (leap_timestamp_corrected[idx] - trig_time) * 1000
    diffs.append(diff_ms)
    
    elapsed = trig_time - leap_timestamp_corrected[0]
    print(f'Trigger {i+1}: elapsed={elapsed:.2f}s, idx={idx}, diff={diff_ms:.3f} ms')

print()
print(f'Mean diff: {np.mean(diffs):.3f} ms')
print(f'Std diff: {np.std(diffs):.3f} ms')
print(f'Range: {np.min(diffs):.3f} ~ {np.max(diffs):.3f} ms')

#%% leap_timestampとsystem_timestampの比較
plt.figure()
plt.plot(system_time_elapsed, diff_timestamp)
#%% フレーム間の時間差のジッターの評価
plt.figure()
for idx in idxs_onset[0]:
    # print(*trigger_status[idx - 3:idx+7])
    diff = np.diff(leap_timestamp[idx - 3:idx+7])
    plt.plot(diff - diff[0])
f.close()

#%% フレーム近いインデックスと遅延を評価：TTLパルスはleap motionのフレームとスレッドが違っているのでばらつくはず
with h5py.File('data/leap_recording_trigger_20260203_133117.h5', 'r') as f:
    system_timestamp = f['system_timestamp'][:]
    trigger_onset_times = f['trigger_onset_times'][:]
    
    # 各TTLパルスに最も近いフレームのインデックス
    frame_indices = np.searchsorted(system_timestamp, trigger_onset_times)
    
    # TTL検知からフレーム記録までの遅延
    for i, onset in enumerate(trigger_onset_times):
        idx = min(frame_indices[i], len(system_timestamp) - 1)
        delay_ms = (system_timestamp[idx] - onset) * 1000
        print(f"Pulse {i+1}: delay = {delay_ms:.3f} ms")
        delay_ms2 = system_timestamp[system_timestamp >= onset][0] - onset
        # print(f"Alternative delay calculation: {delay_ms2 * 1000:.3f} ms")
#%% データ解析：TTLパルス前後のフレームindexを取得
with h5py.File('data/leap_recording_trigger_20260203_133117.h5', 'r') as f:
    system_timestamp = f['system_timestamp'][:]
    trigger_onset_times = f['trigger_onset_times'][:]
    
    # 各TTLパルス後のフレームを取得
    for onset in trigger_onset_times:
        # TTLパルス後20-50msのフレームを抽出
        mask = (system_timestamp >= onset - 0.050) & (system_timestamp <= onset + 0.100)
        frames_in_window = np.where(mask)[0]
        print(frames_in_window)

#%% TTLパルス前後の角度を切り出し保存
import sys
sys.path.append(r'Y:\python\leapmotion_handtracking\tools')
from analyze_finger_angles import extract_trigger_aligned_angles, save_trigger_aligned_data
import matplotlib.pyplot as plt

# データ抽出
data = extract_trigger_aligned_angles(
    'data/leap_recording_trigger_20260203_133117.h5',
    hand_type='right',
    finger_idx=1,  # Index finger
    pre_time=0.1,   # -100ms
    post_time=0.2   # +200ms
)

# プロット例：全試行の平均MCP屈曲角度
time_ms = data['time_axis'] * 1000  # msに変換
mean_flex = np.nanmean(data['mcp_flexion'], axis=0)
std_flex = np.nanstd(data['mcp_flexion'], axis=0)

plt.figure()
plt.fill_between(time_ms, mean_flex - std_flex, mean_flex + std_flex, alpha=0.3)
plt.plot(time_ms, mean_flex, 'b-', linewidth=2)
plt.axvline(0, color='r', linestyle='--', label='TTL trigger')
plt.xlabel('Time from trigger (ms)')
plt.ylabel('MCP Flexion (deg)')
plt.title('Index finger MCP flexion aligned to TTL')
plt.legend()
plt.show()

# 保存
save_trigger_aligned_data(data, 'data/trigger_aligned_angles.h5')
#%% データ解析：
import sys
import matplotlib.pyplot as plt
plt.ion()
sys.path.append(r'Y:\python\leapmotion_handtracking\tools')
import importlib
import analyze_finger_angles

# モジュールをリロード
importlib.reload(analyze_finger_angles)

from analyze_finger_angles import (
    extract_trigger_aligned_angles,
    plot_trigger_aligned_angles,
    plot_all_angles
)


# データ抽出
data = extract_trigger_aligned_angles(
    'data/leap_recording_trigger_20260203_133117_corrected.h5',
    hand_type='right',
    finger_idx=1,
    pre_time=0.1,
    post_time=0.2
)

# 単一プロット（MCP屈曲）
fig, ax = plot_trigger_aligned_angles(
    data,
    angle_type='mcp_flexion',
    show_trials=True,
    mep_window=(20, 50),
    save_path='mcp_flexion_plot.png'
)

# 4つの角度を一度にプロット
fig, axes = plot_all_angles(
    data,
    show_trials=True,
    mep_window=(20, 50),
    save_path='all_angles_plot.png'
)

# plt.show()
