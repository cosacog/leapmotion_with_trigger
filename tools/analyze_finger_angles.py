import h5py
import numpy as np
import sys
import os

def quaternion_to_palm_vectors(quat):
    """
    クォータニオンからpalm_normalとpalm_directionを計算

    Args:
        quat: [x, y, z, w] のクォータニオン

    Returns:
        palm_normal: 手のひら法線ベクトル（手のひらから下向き）
        palm_direction: 手のひら方向ベクトル（手首から指先方向）
    """
    x, y, z, w = quat

    # 回転行列の列ベクトルを抽出（標準的なクォータニオン→回転行列の公式）
    # Column 2: +Y軸の行き先を反転 = palm_normal（手のひらが向く方向）
    # Leap Motionの手のひら座標系では+Yが手の甲方向なので反転して外向きにする
    palm_normal = -np.array([
        2*(x*y - w*z),
        1 - 2*(x*x + z*z),
        2*(y*z + w*x)
    ])

    # Column 3: +Z軸の行き先を反転 = palm_direction（手首→指先方向）
    # Leap Motionの手のひら座標系では-Zが指先方向なので反転
    palm_direction = -np.array([
        2*(x*z + w*y),
        2*(y*z - w*x),
        1 - 2*(x*x + y*y)
    ])

    return palm_normal, palm_direction


def bone_to_vector(bone_data):
    """
    骨データ（prev_joint, next_joint）から方向ベクトルを計算

    Args:
        bone_data: [prev_joint[3], next_joint[3]] shape=(2, 3)

    Returns:
        direction_vector: next - prev
    """
    prev_joint = bone_data[0]  # [x, y, z]
    next_joint = bone_data[1]  # [x, y, z]
    return next_joint - prev_joint


def calculate_signed_angle(v1, v2, reference_normal):
    """
    2つのベクトル間の符号付き角度を計算（atan2使用で安定）

    Args:
        v1, v2: 角度を計算する2つのベクトル
        reference_normal: 符号判定用の法線ベクトル（回転軸）

    Returns:
        angle_deg: 符号付き角度（度、-180° ~ +180°の範囲）
               正: reference_normalの右手系でv1からv2への回転
               負: 逆回転
    """
    # 正規化
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
    n_norm = reference_normal / (np.linalg.norm(reference_normal) + 1e-10)

    # v1を基準軸、n_norm × v1を第2軸とする平面内座標系を構築
    axis_x = v1_norm
    axis_y = np.cross(n_norm, v1_norm)
    axis_y_len = np.linalg.norm(axis_y)

    if axis_y_len < 1e-10:
        # v1とnormalが平行な場合（特殊ケース）
        return 0.0

    axis_y = axis_y / axis_y_len

    # v2をこの2D座標系に投影
    x = np.dot(v2_norm, axis_x)
    y = np.dot(v2_norm, axis_y)

    # atan2で-180° ~ +180°の範囲で角度計算
    angle_rad = np.arctan2(y, x)

    return np.degrees(angle_rad)


def calculate_finger_angles(vec1, vec2, palm_ori):
    """
    2つのベクトル間の屈曲角度と外転角度を計算

    Args:
        vec1: 基準ベクトル（例: Metacarpal方向）
        vec2: 比較ベクトル（例: Proximal方向、または全体方向）
        palm_ori: [x, y, z, w] - palm orientation quaternion

    Returns:
        flexion_angle: 屈曲/伸展角度（正=屈曲、負=伸展）
        abduction_angle: 外転/内転角度（正=外転（橈側）、負=内転（尺側））
    """
    palm_normal, palm_direction = quaternion_to_palm_vectors(palm_ori)

    # 手の座標系を定義
    # lateral_axis: 横軸（親指→小指方向）= palm_normal × palm_direction
    lateral_axis = np.cross(palm_normal, palm_direction)
    lateral_axis = lateral_axis / (np.linalg.norm(lateral_axis) + 1e-10)

    # 正規化（ゼロベクトル検出）
    vec1_len = np.linalg.norm(vec1)
    vec2_len = np.linalg.norm(vec2)
    if vec1_len < 1e-10 or vec2_len < 1e-10:
        return np.nan, np.nan
    vec1_norm = vec1 / vec1_len
    vec2_norm = vec2 / vec2_len

    # === 屈曲/伸展角度 ===
    # lateral_axisに垂直な平面（矢状面）への投影で評価
    vec1_proj_flex = vec1_norm - np.dot(vec1_norm, lateral_axis) * lateral_axis
    vec2_proj_flex = vec2_norm - np.dot(vec2_norm, lateral_axis) * lateral_axis

    vec1_proj_flex = vec1_proj_flex / (np.linalg.norm(vec1_proj_flex) + 1e-10)
    vec2_proj_flex = vec2_proj_flex / (np.linalg.norm(vec2_proj_flex) + 1e-10)

    flexion_angle = calculate_signed_angle(vec1_proj_flex, vec2_proj_flex, lateral_axis)

    # === 外転/内転角度 ===
    # palm_normalに垂直な平面（手のひら平面）への投影で評価
    vec1_proj_abd = vec1_norm - np.dot(vec1_norm, palm_normal) * palm_normal
    vec2_proj_abd = vec2_norm - np.dot(vec2_norm, palm_normal) * palm_normal

    vec1_proj_abd = vec1_proj_abd / (np.linalg.norm(vec1_proj_abd) + 1e-10)
    vec2_proj_abd = vec2_proj_abd / (np.linalg.norm(vec2_proj_abd) + 1e-10)

    abduction_angle = calculate_signed_angle(vec1_proj_abd, vec2_proj_abd, palm_normal)

    return flexion_angle, abduction_angle


def calculate_mcp_angles(finger_data, palm_ori):
    """
    MCP関節の角度を計算（Metacarpal vs Proximal）

    Args:
        finger_data: shape=(4, 2, 3) - [bone_idx, joint_idx, xyz]
                     bone_idx: 0=Metacarpal, 1=Proximal, 2=Intermediate, 3=Distal
        palm_ori: [x, y, z, w] - palm orientation quaternion

    Returns:
        flexion_angle: 屈曲/伸展角度（正=屈曲、負=伸展）
        abduction_angle: 外転/内転角度（正=外転（橈側）、負=内転（尺側））
    """
    metacarpal_vec = bone_to_vector(finger_data[0])  # bones[0]
    proximal_vec = bone_to_vector(finger_data[1])    # bones[1]
    return calculate_finger_angles(metacarpal_vec, proximal_vec, palm_ori)


def calculate_overall_bend_angle(finger_data, palm_ori):
    """
    全体的な屈曲角度を計算（Metacarpal基部 → 指先）

    Args:
        finger_data: shape=(4, 2, 3) - [bone_idx, joint_idx, xyz]
        palm_ori: [x, y, z, w] - palm orientation quaternion

    Returns:
        flexion_angle: 全体屈曲角度（正=屈曲、負=伸展）
        abduction_angle: 全体外転角度（正=外転、負=内転）
    """
    # Metacarpal方向ベクトル
    metacarpal_vec = bone_to_vector(finger_data[0])

    # Metacarpal遠位端から指先（Distal遠位端）へのベクトル
    metacarpal_distal_end = finger_data[0][1]  # Metacarpal next_joint
    fingertip = finger_data[3][1]  # Distal next_joint (fingertip)
    overall_vec = fingertip - metacarpal_distal_end

    return calculate_finger_angles(metacarpal_vec, overall_vec, palm_ori)


def analyze_index_finger(h5_filepath, hand_type='left', save_output=True, output_format='csv'):
    """
    人差し指の角度を全フレームで解析

    Args:
        h5_filepath: HDF5ファイルパス
        hand_type: 'left' or 'right'
        save_output: データをファイルに保存するか
        output_format: 出力形式 ('csv', 'h5', 'both')
    """
    print(f"Opening {h5_filepath}...")
    print(f"Analyzing {hand_type} hand index finger\n")

    with h5py.File(h5_filepath, 'r') as f:
        n_frames = f['leap_timestamp'].shape[0]
        print(f"Total frames: {n_frames}\n")

        hand_prefix = f'{hand_type}_hand'

        # データセット読み込み
        valid = f[f'{hand_prefix}/valid'][:]
        palm_ori = f[f'{hand_prefix}/palm_ori'][:]  # shape: (n_frames, 4)
        fingers = f[f'{hand_prefix}/fingers'][:]     # shape: (n_frames, 5, 4, 2, 3)
        timestamps = f['leap_timestamp'][:]

        # 人差し指は fingers[:, 1, :, :, :] (digit index = 1)
        index_finger = fingers[:, 1, :, :, :]  # shape: (n_frames, 4, 2, 3)

        # 全フレームのデータを保存するリスト
        results = []

        print("Frame | Valid | MCP Flex | MCP Abd | Overall Flex | Overall Abd")
        print("-" * 70)

        for frame_idx in range(n_frames):
            if not valid[frame_idx]:
                # 無効なフレームもデータに含める（NaNとして）
                results.append({
                    'frame': frame_idx,
                    'timestamp': timestamps[frame_idx],
                    'valid': False,
                    'mcp_flexion': np.nan,
                    'mcp_abduction': np.nan,
                    'overall_flexion': np.nan,
                    'overall_abduction': np.nan
                })
                if frame_idx < 10:
                    print(f"{frame_idx:5d} | False | -        | -       | -            | -")
                continue

            # MCP関節角度
            mcp_flex, mcp_abd = calculate_mcp_angles(
                index_finger[frame_idx],
                palm_ori[frame_idx]
            )

            # 全体屈曲角度
            overall_flex, overall_abd = calculate_overall_bend_angle(
                index_finger[frame_idx],
                palm_ori[frame_idx]
            )

            # 結果を保存
            results.append({
                'frame': frame_idx,
                'timestamp': timestamps[frame_idx],
                'valid': True,
                'mcp_flexion': mcp_flex,
                'mcp_abduction': mcp_abd,
                'overall_flexion': overall_flex,
                'overall_abduction': overall_abd
            })

            # サンプル表示（最初の10フレームのみ詳細表示）
            if frame_idx < 10:
                print(f"{frame_idx:5d} | True  | {mcp_flex:+7.2f}° | {mcp_abd:+7.2f}° | {overall_flex:+11.2f}° | {overall_abd:+10.2f}°")

        if n_frames > 10:
            print(f"... ({n_frames - 10} more frames)")

        print("\n=== Angle Definitions ===")
        print("Flexion/Extension (palm_normal reference):")
        print("  Positive (+) = Flexion (bending toward palm)")
        print("  Negative (-) = Extension (bending away from palm)")
        print("\nAbduction/Adduction (palm_direction reference):")
        print("  Positive (+) = Abduction (toward thumb/radial side)")
        print("  Negative (-) = Adduction (toward pinky/ulnar side)")

        # データ保存
        if save_output:
            _save_results(results, h5_filepath, hand_type, output_format)

        return results


def _save_results(results, source_filepath, hand_type, output_format):
    """
    解析結果をファイルに保存

    Args:
        results: 解析結果のリスト
        source_filepath: 元のHDF5ファイルパス
        hand_type: 'left' or 'right'
        output_format: 'csv', 'h5', or 'both'
    """
    import pandas as pd

    # DataFrameに変換
    df = pd.DataFrame(results)

    # 出力ファイル名の生成（元のファイルと同じディレクトリに保存）
    source_dir = os.path.dirname(source_filepath)
    base_name = os.path.splitext(os.path.basename(source_filepath))[0]
    output_base = os.path.join(source_dir, f"{base_name}_{hand_type}_index_angles")

    print(f"\n=== Saving Results ===")

    # CSV形式で保存
    if output_format in ['csv', 'both']:
        csv_path = f"{output_base}.csv"
        df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"CSV saved to: {csv_path}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df)} ({df['valid'].sum()} valid)")

    # HDF5形式で保存
    if output_format in ['h5', 'both']:
        h5_path = f"{output_base}.h5"
        with h5py.File(h5_path, 'w') as hf:
            # メタデータ
            hf.attrs['source_file'] = source_filepath
            hf.attrs['hand_type'] = hand_type
            hf.attrs['finger'] = 'index'
            hf.attrs['total_frames'] = len(results)
            hf.attrs['valid_frames'] = int(df['valid'].sum())

            # データセット作成
            hf.create_dataset('frame', data=df['frame'].values, dtype='i4')
            hf.create_dataset('timestamp', data=df['timestamp'].values, dtype='i8')
            hf.create_dataset('valid', data=df['valid'].values, dtype='bool')

            # 角度データ（NaNを含む可能性があるのでfloat64）
            hf.create_dataset('mcp_flexion', data=df['mcp_flexion'].values, dtype='f8')
            hf.create_dataset('mcp_abduction', data=df['mcp_abduction'].values, dtype='f8')
            hf.create_dataset('overall_flexion', data=df['overall_flexion'].values, dtype='f8')
            hf.create_dataset('overall_abduction', data=df['overall_abduction'].values, dtype='f8')

        print(f"HDF5 saved to: {h5_path}")
        print(f"  Datasets: frame, timestamp, valid, mcp_flexion, mcp_abduction, overall_flexion, overall_abduction")

    print(f"\nData export complete!")


def extract_trigger_aligned_angles(h5_filepath, hand_type='left', finger_idx=1,
                                    pre_time=0.1, post_time=0.2):
    """
    trigger_onset_timesを基準に角度データを切り出す

    Args:
        h5_filepath: HDF5ファイルパス
        hand_type: 'left' or 'right'
        finger_idx: 指のインデックス (0=Thumb, 1=Index, 2=Middle, 3=Ring, 4=Pinky)
        pre_time: トリガー前の時間（秒、正の値）デフォルト0.1秒
        post_time: トリガー後の時間（秒）デフォルト0.2秒

    Returns:
        dict: {
            'trigger_times': array of trigger onset times,
            'time_axis': array of time relative to trigger (seconds),
            'mcp_flexion': array shape (n_triggers, n_timepoints),
            'mcp_abduction': array shape (n_triggers, n_timepoints),
            'overall_flexion': array shape (n_triggers, n_timepoints),
            'overall_abduction': array shape (n_triggers, n_timepoints),
            'valid': array shape (n_triggers, n_timepoints),
            'frame_indices': array shape (n_triggers, n_timepoints),
        }
    """
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    print(f"Opening {h5_filepath}...")
    print(f"Extracting {hand_type} hand {finger_names[finger_idx]} finger angles")
    print(f"Time window: -{pre_time*1000:.0f}ms to +{post_time*1000:.0f}ms relative to trigger\n")

    with h5py.File(h5_filepath, 'r') as f:
        # Check if trigger_onset_times exists
        if 'trigger_onset_times' not in f:
            raise ValueError("trigger_onset_times not found in HDF5 file. "
                           "Recording may not have trigger data.")

        # Load data
        system_timestamp = f['system_timestamp'][:]
        # trigger_onset_times = f['trigger_onset_times'][:]
        # trigger_onset_times = f['trigger_onset_times_corrected'][:]
        trigger_onset_times = f['trigger_onset_times_corrected'][:]
        n_triggers = len(trigger_onset_times)

        print(f"Found {n_triggers} triggers")
        print(f"Total frames: {len(system_timestamp)}")

        if n_triggers == 0:
            raise ValueError("No triggers found in the recording.")

        hand_prefix = f'{hand_type}_hand'
        valid = f[f'{hand_prefix}/valid'][:]
        palm_ori = f[f'{hand_prefix}/palm_ori'][:]
        fingers = f[f'{hand_prefix}/fingers'][:]

        # Target finger data
        finger_data = fingers[:, finger_idx, :, :, :]  # shape: (n_frames, 4, 2, 3)

        # Estimate sampling rate from data
        dt = np.median(np.diff(system_timestamp))
        sampling_rate = 1.0 / dt
        print(f"Estimated sampling rate: {sampling_rate:.1f} Hz")

        # Calculate number of samples for the time window
        n_pre_samples = int(np.ceil(pre_time / dt))
        n_post_samples = int(np.ceil(post_time / dt))
        n_timepoints = n_pre_samples + n_post_samples + 1

        # Create time axis relative to trigger
        time_axis = np.arange(-n_pre_samples, n_post_samples + 1) * dt

        # Initialize output arrays
        mcp_flexion = np.full((n_triggers, n_timepoints), np.nan)
        mcp_abduction = np.full((n_triggers, n_timepoints), np.nan)
        overall_flexion = np.full((n_triggers, n_timepoints), np.nan)
        overall_abduction = np.full((n_triggers, n_timepoints), np.nan)
        valid_mask = np.zeros((n_triggers, n_timepoints), dtype=bool)
        frame_indices = np.full((n_triggers, n_timepoints), -1, dtype=int)

        # Extract epochs for each trigger
        for trig_idx, trig_time in enumerate(trigger_onset_times):
            # Find the frame closest to trigger time
            center_idx = np.searchsorted(system_timestamp, trig_time)

            # Calculate start and end frame indices
            start_idx = center_idx - n_pre_samples
            end_idx = center_idx + n_post_samples + 1

            # Check bounds
            if start_idx < 0 or end_idx > len(system_timestamp):
                print(f"  Trigger {trig_idx + 1}: Out of bounds (skipping)")
                continue

            # Extract frames for this epoch
            for i, frame_idx in enumerate(range(start_idx, end_idx)):
                frame_indices[trig_idx, i] = frame_idx

                if not valid[frame_idx]:
                    continue

                valid_mask[trig_idx, i] = True

                # Calculate MCP angles
                mcp_flex, mcp_abd = calculate_mcp_angles(
                    finger_data[frame_idx],
                    palm_ori[frame_idx]
                )
                mcp_flexion[trig_idx, i] = mcp_flex
                mcp_abduction[trig_idx, i] = mcp_abd

                # Calculate overall angles
                overall_flex, overall_abd = calculate_overall_bend_angle(
                    finger_data[frame_idx],
                    palm_ori[frame_idx]
                )
                overall_flexion[trig_idx, i] = overall_flex
                overall_abduction[trig_idx, i] = overall_abd

            # Print progress
            if (trig_idx + 1) % 10 == 0 or trig_idx == 0:
                valid_count = np.sum(valid_mask[trig_idx])
                print(f"  Trigger {trig_idx + 1}/{n_triggers}: {valid_count}/{n_timepoints} valid frames")

        print(f"\nExtraction complete!")
        print(f"  Valid epochs: {np.sum(np.any(valid_mask, axis=1))}/{n_triggers}")

        return {
            'trigger_times': trigger_onset_times,
            'time_axis': time_axis,
            'mcp_flexion': mcp_flexion,
            'mcp_abduction': mcp_abduction,
            'overall_flexion': overall_flexion,
            'overall_abduction': overall_abduction,
            'valid': valid_mask,
            'frame_indices': frame_indices,
            'sampling_rate': sampling_rate,
            'hand_type': hand_type,
            'finger_name': finger_names[finger_idx],
        }


def save_trigger_aligned_data(data, output_filepath):
    """
    extract_trigger_aligned_anglesの結果をHDF5ファイルに保存

    Args:
        data: extract_trigger_aligned_anglesの戻り値
        output_filepath: 出力ファイルパス
    """
    with h5py.File(output_filepath, 'w') as f:
        # Metadata
        f.attrs['hand_type'] = data['hand_type']
        f.attrs['finger_name'] = data['finger_name']
        f.attrs['sampling_rate'] = data['sampling_rate']
        f.attrs['n_triggers'] = len(data['trigger_times'])
        f.attrs['n_timepoints'] = len(data['time_axis'])

        # Data
        f.create_dataset('trigger_times', data=data['trigger_times'], dtype='f8')
        f.create_dataset('time_axis', data=data['time_axis'], dtype='f8')
        f.create_dataset('mcp_flexion', data=data['mcp_flexion'], dtype='f8')
        f.create_dataset('mcp_abduction', data=data['mcp_abduction'], dtype='f8')
        f.create_dataset('overall_flexion', data=data['overall_flexion'], dtype='f8')
        f.create_dataset('overall_abduction', data=data['overall_abduction'], dtype='f8')
        f.create_dataset('valid', data=data['valid'], dtype='bool')
        f.create_dataset('frame_indices', data=data['frame_indices'], dtype='i4')

    print(f"Saved to: {output_filepath}")


def plot_trigger_aligned_angles(data, angle_type='mcp_flexion', show_trials=True,
                                  mep_window=(20, 50), figsize=(12, 8), save_path=None):
    """
    トリガー整列した角度データをプロット

    Args:
        data: extract_trigger_aligned_anglesの戻り値
        angle_type: プロットする角度タイプ
                   'mcp_flexion', 'mcp_abduction', 'overall_flexion', 'overall_abduction'
        show_trials: 各試行のデータを重ね書きするか
        mep_window: MEPウィンドウ (start_ms, end_ms) または None
        figsize: 図のサイズ
        save_path: 保存先パス（Noneの場合は保存しない）

    Returns:
        fig, ax: matplotlib figure and axes
    """
    import matplotlib.pyplot as plt

    # 時間軸をmsに変換
    time_ms = data['time_axis'] * 1000

    # 角度データを取得
    angle_data = data[angle_type]

    # 角度タイプの日本語/英語ラベル
    angle_labels = {
        'mcp_flexion': 'MCP Flexion',
        'mcp_abduction': 'MCP Abduction',
        'overall_flexion': 'Overall Flexion',
        'overall_abduction': 'Overall Abduction'
    }
    angle_label = angle_labels.get(angle_type, angle_type)

    # 統計量を計算
    mean_angle = np.nanmean(angle_data, axis=0)
    std_angle = np.nanstd(angle_data, axis=0)
    n_trials = angle_data.shape[0]

    # プロット作成
    fig, ax = plt.subplots(figsize=figsize)

    # MEPウィンドウをハイライト
    if mep_window is not None:
        ax.axvspan(mep_window[0], mep_window[1], alpha=0.15, color='yellow',
                   label=f'MEP window ({mep_window[0]}-{mep_window[1]}ms)')

    # 各試行のデータをプロット
    if show_trials:
        for i in range(n_trials):
            ax.plot(time_ms, angle_data[i], color='black', alpha=0.7, linewidth=1.5)

    # 平均±SDをプロット
    ax.fill_between(time_ms, mean_angle - std_angle, mean_angle + std_angle,
                    alpha=0.4, color='blue', label='Mean ± SD')
    ax.plot(time_ms, mean_angle, 'b-', linewidth=2, label='Mean')

    # トリガー時刻の縦線
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='TTL trigger')

    # ラベルとタイトル
    ax.set_xlabel('Time from trigger (ms)', fontsize=12)
    ax.set_ylabel(f'{angle_label} (deg)', fontsize=12)
    ax.set_title(f"{data['hand_type'].capitalize()} {data['finger_name']} finger - {angle_label}\n"
                 f"(n={n_trials} trials, {data['sampling_rate']:.1f} Hz)", fontsize=14)

    # グリッドと凡例
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # x軸の範囲を設定
    ax.set_xlim(time_ms[0], time_ms[-1])

    plt.tight_layout()

    # 保存
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, ax


def plot_all_angles(data, show_trials=True, mep_window=(20, 50), figsize=(14, 10), save_path=None):
    """
    4つの角度タイプを2x2のサブプロットで表示

    Args:
        data: extract_trigger_aligned_anglesの戻り値
        show_trials: 各試行のデータを重ね書きするか
        mep_window: MEPウィンドウ (start_ms, end_ms) または None
        figsize: 図のサイズ
        save_path: 保存先パス（Noneの場合は保存しない）

    Returns:
        fig, axes: matplotlib figure and axes
    """
    import matplotlib.pyplot as plt

    angle_types = ['mcp_flexion', 'mcp_abduction', 'overall_flexion', 'overall_abduction']
    angle_labels = ['MCP Flexion', 'MCP Abduction', 'Overall Flexion', 'Overall Abduction']

    time_ms = data['time_axis'] * 1000
    n_trials = data['mcp_flexion'].shape[0]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, (angle_type, angle_label) in enumerate(zip(angle_types, angle_labels)):
        ax = axes[idx]
        angle_data = data[angle_type]

        mean_angle = np.nanmean(angle_data, axis=0)
        std_angle = np.nanstd(angle_data, axis=0)

        # MEPウィンドウ
        if mep_window is not None:
            ax.axvspan(mep_window[0], mep_window[1], alpha=0.15, color='yellow')

        # 各試行
        if show_trials:
            for i in range(n_trials):
                ax.plot(time_ms, angle_data[i], color='black', alpha=0.7, linewidth=1.5)

        # 平均±SD
        ax.fill_between(time_ms, mean_angle - std_angle, mean_angle + std_angle,
                        alpha=0.4, color='blue')
        ax.plot(time_ms, mean_angle, 'b-', linewidth=2)

        # トリガー線
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5)

        ax.set_xlabel('Time (ms)', fontsize=10)
        ax.set_ylabel(f'{angle_label} (deg)', fontsize=10)
        ax.set_title(angle_label, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time_ms[0], time_ms[-1])

    fig.suptitle(f"{data['hand_type'].capitalize()} {data['finger_name']} finger - "
                 f"Trigger-aligned angles (n={n_trials})", fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, axes


def main():
    """
    Usage:
        python analyze_finger_angles.py <h5_file> [hand_type] [--no-save] [--format csv|h5|both]

    Examples:
        python analyze_finger_angles.py data/recording.h5
        python analyze_finger_angles.py data/recording.h5 right
        python analyze_finger_angles.py data/recording.h5 left --format both
        python analyze_finger_angles.py data/recording.h5 --no-save
    """
    # コマンドライン引数の解析
    save_output = True
    output_format = 'csv'
    hand_type = 'left'
    filepath = None

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == '--no-save':
            save_output = False
        elif arg == '--format':
            if i + 1 < len(args):
                output_format = args[i + 1]
                if output_format not in ['csv', 'h5', 'both']:
                    print("Format must be 'csv', 'h5', or 'both'")
                    return
        elif arg in ['left', 'right']:
            hand_type = arg
        elif not arg.startswith('--') and arg not in ['csv', 'h5', 'both']:
            if filepath is None:
                filepath = arg

    # ファイルパスが指定されていない場合、最新のファイルを探す
    if filepath is None:
        data_dir = "data"
        if os.path.exists(data_dir):
            files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".h5")]
            if files:
                filepath = max(files, key=os.path.getctime)
                print(f"No file specified. Using latest: {filepath}\n")
            else:
                print("Usage: python analyze_finger_angles.py <path_to_h5_file> [left|right] [--no-save] [--format csv|h5|both]")
                return
        else:
            print("Usage: python analyze_finger_angles.py <path_to_h5_file> [left|right] [--no-save] [--format csv|h5|both]")
            return

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    try:
        analyze_index_finger(filepath, hand_type, save_output, output_format)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
