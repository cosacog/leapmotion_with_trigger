import h5py
import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# analyze_finger_angles.pyから関数をインポート
sys.path.insert(0, os.path.dirname(__file__))
from analyze_finger_angles import (
    quaternion_to_palm_vectors,
    bone_to_vector,
    calculate_signed_angle,
    calculate_mcp_angles,
    calculate_overall_bend_angle
)


def validate_flexion_data(h5_filepath, hand_type='left'):
    """
    屈曲/伸展角度の妥当性を検証
    """
    print(f"Opening {h5_filepath}...")
    print(f"Validating {hand_type} hand index finger flexion angles\n")

    with h5py.File(h5_filepath, 'r') as f:
        n_frames = f['leap_timestamp'].shape[0]
        print(f"Total frames: {n_frames}\n")

        hand_prefix = f'{hand_type}_hand'

        # データ読み込み
        valid = f[f'{hand_prefix}/valid'][:]
        palm_ori = f[f'{hand_prefix}/palm_ori'][:]
        fingers = f[f'{hand_prefix}/fingers'][:]
        timestamps = f['leap_timestamp'][:]

        # 人差し指
        index_finger = fingers[:, 1, :, :, :]

        # 各フレームの角度を計算
        mcp_flex_angles = []
        overall_flex_angles = []
        valid_frames = []

        for frame_idx in range(n_frames):
            if not valid[frame_idx]:
                continue

            mcp_flex, _ = calculate_mcp_angles(
                index_finger[frame_idx],
                palm_ori[frame_idx]
            )

            overall_flex, _ = calculate_overall_bend_angle(
                index_finger[frame_idx],
                palm_ori[frame_idx]
            )

            mcp_flex_angles.append(mcp_flex)
            overall_flex_angles.append(overall_flex)
            valid_frames.append(frame_idx)

        mcp_flex_angles = np.array(mcp_flex_angles)
        overall_flex_angles = np.array(overall_flex_angles)

        # 統計情報
        print("=== MCP Joint Flexion/Extension ===")
        print(f"Mean:   {np.mean(mcp_flex_angles):+7.2f}°")
        print(f"Std:    {np.std(mcp_flex_angles):7.2f}°")
        print(f"Min:    {np.min(mcp_flex_angles):+7.2f}°")
        print(f"Max:    {np.max(mcp_flex_angles):+7.2f}°")
        print(f"Range:  {np.max(mcp_flex_angles) - np.min(mcp_flex_angles):7.2f}°")

        print("\n=== Overall Finger Flexion ===")
        print(f"Mean:   {np.mean(overall_flex_angles):+7.2f}°")
        print(f"Std:    {np.std(overall_flex_angles):7.2f}°")
        print(f"Min:    {np.min(overall_flex_angles):+7.2f}°")
        print(f"Max:    {np.max(overall_flex_angles):+7.2f}°")
        print(f"Range:  {np.max(overall_flex_angles) - np.min(overall_flex_angles):7.2f}°")

        # 時系列プロット
        print("\n=== Creating time-series plot ===")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # タイムスタンプを秒単位に変換
        time_seconds = (timestamps[valid_frames] - timestamps[valid_frames[0]]) / 1e6

        # MCP Flexion
        ax1.plot(time_seconds, mcp_flex_angles, 'b-', linewidth=1.5, label='MCP Flexion')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_ylabel('MCP Flexion Angle (degrees)', fontsize=12)
        ax1.set_title('Index Finger Flexion/Extension Angles Over Time', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Overall Flexion
        ax2.plot(time_seconds, overall_flex_angles, 'r-', linewidth=1.5, label='Overall Flexion')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Overall Flexion Angle (degrees)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        # 保存
        output_path = 'flexion_validation.png'
        plt.savefig(output_path, dpi=150)
        print(f"\nPlot saved to: {output_path}")

        # ヒストグラム
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

        ax3.hist(mcp_flex_angles, bins=30, color='blue', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('MCP Flexion Angle (degrees)', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('MCP Flexion Distribution', fontsize=14)
        ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)

        ax4.hist(overall_flex_angles, bins=30, color='red', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Overall Flexion Angle (degrees)', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Overall Flexion Distribution', fontsize=14)
        ax4.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path2 = 'flexion_histogram.png'
        plt.savefig(output_path2, dpi=150)
        print(f"Histogram saved to: {output_path2}")

        plt.close('all')

        # 動きの範囲の評価
        print("\n=== Movement Assessment ===")
        mcp_range = np.max(mcp_flex_angles) - np.min(mcp_flex_angles)
        overall_range = np.max(overall_flex_angles) - np.min(overall_flex_angles)

        print(f"MCP flexion range: {mcp_range:.2f}°")
        if mcp_range < 10:
            print("  → Very small movement (expected: 20-60° for grasping)")
        elif mcp_range < 30:
            print("  → Small to moderate movement")
        else:
            print("  → Good range of motion")

        print(f"\nOverall flexion range: {overall_range:.2f}°")
        if overall_range < 20:
            print("  → Very small movement (expected: 40-80° for grasping)")
        elif overall_range < 50:
            print("  → Moderate movement")
        else:
            print("  → Good range of motion")


def main():
    if len(sys.argv) < 2:
        data_dir = "data"
        if os.path.exists(data_dir):
            files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".h5")]
            if files:
                filepath = max(files, key=os.path.getctime)
                print(f"Using latest file: {filepath}\n")
            else:
                print("Usage: python validate_flexion_angles.py <path_to_h5_file> [left|right]")
                return
        else:
            print("Usage: python validate_flexion_angles.py <path_to_h5_file> [left|right]")
            return
    else:
        filepath = sys.argv[1]

    hand_type = 'left'
    if len(sys.argv) >= 3:
        hand_type = sys.argv[2].lower()
        if hand_type not in ['left', 'right']:
            print("Hand type must be 'left' or 'right'")
            return

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    try:
        validate_flexion_data(filepath, hand_type)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
