#%%
import os, h5py
import matplotlib.pyplot as plt
import numpy as np
dir_h5 = "Y:\\python\\leapmotion_with_trigger\\data"
fname_h5 = "leap_recording_trigger_20260216_183851.h5" #ogata_test2
# fname_h5 = "leap_recording_trigger_20260224_134713.h5" #harada_test1
# fname_h5 = "leap_recording_trigger_20260302_150712_120rmt.h5" #shimanouchi 120%rmt
# fname_h5 = "leap_recording_trigger_20260302_151131_105rmt.h5" #shimanouchi 105%rmt
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
left_hand = f['left_hand']
right_hand = f['right_hand']
left_hand_valid = f['left_hand/valid'][:]
right_hand_valid = f['right_hand/valid'][:]
f.close()
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
with h5py.File(path_h5, 'r') as f:
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
with h5py.File(path_h5, 'r') as f:
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
sys.path.append(r'Y:\python\leapmotion_with_trigger\tools')
from analyze_finger_angles import extract_trigger_aligned_angles, save_trigger_aligned_data
import matplotlib.pyplot as plt

# データ抽出
data = extract_trigger_aligned_angles(
    path_h5,
    hand_type='right',
    finger_idx=1,  # Index finger
    pre_time=0.1,   # -100ms
    post_time=0.5  # +200ms
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
h5_suffix = fname_h5.split('_trigger_')[1]
save_trigger_aligned_data(data, f'data/trigger_aligned_angles_{h5_suffix}')
#%% データ解析：
import sys
import matplotlib.pyplot as plt
plt.ion()
sys.path.append(r'Y:\python\leapmotion_with_trigger\tools')
import importlib
import analyze_finger_angles
from mep_measure import measure_mep_amplitudes

# モジュールをリロード
importlib.reload(analyze_finger_angles)

from analyze_finger_angles import (
    extract_trigger_aligned_angles,
    plot_trigger_aligned_angles,
    plot_all_angles
)

# データ抽出
data = extract_trigger_aligned_angles(
    path_h5,
    hand_type='right',
    finger_idx=1,
    pre_time=0.1,
    post_time=0.5,
    baseline_correction=True
)


def get_peak_response(data):
    """
    刺激後(time_axis >= 0)の各angleについて絶対値最大のピーク値（符号付き）を返す。

    Args:
        data: extract_trigger_aligned_anglesの返り値 dict

    Returns:
        dict: angle名をキーとし、各試行のピーク値 array shape (n_triggers,) を値とする
    """
    time_axis = data['time_axis']
    post_mask = (time_axis >= 0.05) & (time_axis <= 0.2)  # 刺激後0-500msの範囲
    angle_keys = ['mcp_flexion', 'mcp_abduction', 'overall_flexion', 'overall_abduction']

    result = {}
    for key in angle_keys:
        arr = data[key][:, post_mask]  # (n_triggers, n_post_timepoints)
        abs_max_idx = np.nanargmax(np.abs(arr), axis=1)  # 各試行の絶対値最大インデックス
        result[key] = arr[np.arange(arr.shape[0]), abs_max_idx]  # 符号付きの値

    return result

peak_response = get_peak_response(data)

# mepデータの取得と振幅測定
dir_mep = r'E:\NeuroPhysiology\TMS\leapmotion_mep\20260216ogata_test2' # ogata_test2
# dir_mep = r'E:\NeuroPhysiology\TMS\leapmotion_mep\20260224_harada' # harada
# dir_mep = r'E:\NeuroPhysiology\TMS\leapmotion_mep\20260302_shimanouchi_120rmt30' # shimanouchi 120%rmt
# dir_mep = r'E:\NeuroPhysiology\TMS\leapmotion_mep\20260302_shimanouchi_105rmt10' # shimanouchi 105%rmt
mep_amplitudes = measure_mep_amplitudes(dir_mep)

png_suffix = fname_h5.split('_trigger_')[1].replace('.h5', '.png')
# 単一プロット（MCP屈曲）
fig, ax = plot_trigger_aligned_angles(
    data,
    angle_type='mcp_flexion', # 'mcp_flexion', 'mcp_abduction', 'overall_flexion', 'overall_abduction'
    show_trials=True,
    mep_window=(20, 50),
    save_path=os.path.join('data', f'mcp_flexion_plot_{png_suffix}')
)

# 4つの角度を一度にプロット
fig, axes = plot_all_angles(
    data,
    show_trials=True,
    mep_window=(20, 50),
    save_path=os.path.join('data', f'all_angles_plot_{png_suffix}')
)

# plt.show()
#%% MEP波形とLeap Motion波形の正規化重ね書き
from mep_measure import get_mep_data

# MEPデータ取得（mep_time: ms軸, mep_data: shape=(n_trials, n_timepoints)）
mep_time, mep_data = get_mep_data(dir_mep)
mean_mep = np.mean(mep_data, axis=0)

# Leap Motion 時間軸をmsに変換
leap_time_ms = data['time_axis'] * 1000

angle_types = ['mcp_flexion', 'mcp_abduction', 'overall_flexion', 'overall_abduction']
angle_labels = ['MCP Flexion', 'MCP Abduction', 'Overall Flexion', 'Overall Abduction']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (angle_type, angle_label) in enumerate(zip(angle_types, angle_labels)):
    ax = axes[idx]
    angle_data = data[angle_type]
    mean_angle = np.nanmean(angle_data, axis=0)
    std_angle = np.nanstd(angle_data, axis=0)

    # 各平均波形の絶対値最大で正規化（ピークを±1に揃える）
    scale_angle = np.nanmax(np.abs(mean_angle))
    scale_mep = np.max(np.abs(mean_mep))

    norm_angle = mean_angle / (scale_angle + 1e-10)
    norm_std = std_angle / (scale_angle + 1e-10)
    norm_mep = mean_mep / (scale_mep + 1e-10)

    # Leap Motion: 個別試行（薄く）
    for trial in angle_data:
        # ax.plot(leap_time_ms, trial / (scale_angle + 1e-10),
        #         color='steelblue', alpha=1.0, linewidth=1.5)
        ax.plot(leap_time_ms, trial,
                color='steelblue', alpha=1.0, linewidth=1.5)

    # # Leap Motion: 平均±SD
    # ax.fill_between(leap_time_ms, norm_angle - norm_std, norm_angle + norm_std,
    #                 alpha=0.8, color='steelblue')
    # ax.plot(leap_time_ms, norm_angle, color='steelblue', linewidth=2,
    #         label=f'{angle_label} (max={scale_angle:.1f}°)')

    # MEP: 個別試行（薄く）
    for trial in mep_data:
        # ax.plot(mep_time, trial / (scale_mep + 1e-10),
        #         color='tomato', alpha=0.8, linewidth=0.8)
        ax.plot(mep_time, trial / 1e3,
                color='tomato', alpha=0.8, linewidth=0.8)

    # MEP: 平均
    # ax.plot(mep_time, norm_mep, color='tomato', linewidth=2,
    #         label=f'MEP (max={scale_mep:.2f}mV)')

    ax.axvline(0, color='k', linestyle='--', linewidth=1.5, label='TMS trigger')
    ax.set_xlabel('Time from trigger (ms)', fontsize=10)
    ax.set_ylabel('Normalized amplitude (a.u.)', fontsize=20)
    ax.set_title(angle_label, fontsize=20)
    # ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(leap_time_ms[0], leap_time_ms[-1])
    ax.tick_params(axis='both', labelsize=18)

n_leap = data['mcp_flexion'].shape[0]
n_mep = mep_data.shape[0]
fig.suptitle(f"MEP vs Leap Motion - Normalized overlay\n"
             f"(Leap n={n_leap}, MEP n={n_mep})", fontsize=14)
plt.tight_layout()
plt.show()

#%% 角度ピーク値とMEP振幅の散布図
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
angle_key = 'mcp_abduction'  # mcp_abduction, mcp_flexion, overall_abduction, overall_flexion
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.figure()
plt.scatter(mep_amplitudes, peak_response[angle_key], color='blue')
plt.ylabel(f'Peak {angle_key} (deg)', fontsize=20)
plt.xlabel('MEP Amplitude (mV)', fontsize=20)
plt.title(f'Peak {angle_key} vs MEP Amplitude', fontsize=20)
plt.grid()
r, p = pearsonr(peak_response[angle_key], mep_amplitudes)
print(f'Pearson r: {r:.3f}, p-value: {p:.3f}')