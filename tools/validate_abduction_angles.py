import h5py
import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from analyze_finger_angles import calculate_mcp_angles, calculate_overall_bend_angle


def validate_abduction_data(h5_filepath, hand_type='left'):
    """
    外転/内転角度の妥当性を検証
    """
    print(f"Opening {h5_filepath}...")
    print(f"Validating {hand_type} hand index finger abduction angles\n")

    with h5py.File(h5_filepath, 'r') as f:
        n_frames = f['leap_timestamp'].shape[0]
        print(f"Total frames: {n_frames}\n")

        hand_prefix = f'{hand_type}_hand'

        valid = f[f'{hand_prefix}/valid'][:]
        palm_ori = f[f'{hand_prefix}/palm_ori'][:]
        fingers = f[f'{hand_prefix}/fingers'][:]
        timestamps = f['leap_timestamp'][:]

        index_finger = fingers[:, 1, :, :, :]

        mcp_abd_angles = []
        overall_abd_angles = []
        mcp_flex_angles = []
        overall_flex_angles = []
        valid_frames = []

        for frame_idx in range(n_frames):
            if not valid[frame_idx]:
                continue

            mcp_flex, mcp_abd = calculate_mcp_angles(
                index_finger[frame_idx],
                palm_ori[frame_idx]
            )

            overall_flex, overall_abd = calculate_overall_bend_angle(
                index_finger[frame_idx],
                palm_ori[frame_idx]
            )

            mcp_abd_angles.append(mcp_abd)
            overall_abd_angles.append(overall_abd)
            mcp_flex_angles.append(mcp_flex)
            overall_flex_angles.append(overall_flex)
            valid_frames.append(frame_idx)

        mcp_abd_angles = np.array(mcp_abd_angles)
        overall_abd_angles = np.array(overall_abd_angles)
        mcp_flex_angles = np.array(mcp_flex_angles)
        overall_flex_angles = np.array(overall_flex_angles)

        # 統計
        print("=== MCP Joint Abduction/Adduction ===")
        print(f"Mean:   {np.mean(mcp_abd_angles):+7.2f}°")
        print(f"Std:    {np.std(mcp_abd_angles):7.2f}°")
        print(f"Min:    {np.min(mcp_abd_angles):+7.2f}°")
        print(f"Max:    {np.max(mcp_abd_angles):+7.2f}°")
        print(f"Range:  {np.max(mcp_abd_angles) - np.min(mcp_abd_angles):7.2f}°")

        print("\n=== Overall Finger Abduction ===")
        print(f"Mean:   {np.mean(overall_abd_angles):+7.2f}°")
        print(f"Std:    {np.std(overall_abd_angles):7.2f}°")
        print(f"Min:    {np.min(overall_abd_angles):+7.2f}°")
        print(f"Max:    {np.max(overall_abd_angles):+7.2f}°")
        print(f"Range:  {np.max(overall_abd_angles) - np.min(overall_abd_angles):7.2f}°")

        # 相関分析
        print("\n=== Correlation Analysis ===")
        corr_mcp = np.corrcoef(mcp_flex_angles, mcp_abd_angles)[0, 1]
        corr_overall = np.corrcoef(overall_flex_angles, overall_abd_angles)[0, 1]
        print(f"Correlation (MCP Flex vs Abd):     {corr_mcp:+.3f}")
        print(f"Correlation (Overall Flex vs Abd): {corr_overall:+.3f}")
        print("\nNote: High correlation suggests abduction is affected by hand rotation,")
        print("      not actual finger abduction movement.")

        # タイムスタンプ変換
        time_seconds = (timestamps[valid_frames] - timestamps[valid_frames[0]]) / 1e6

        # プロット1: 時系列
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

        ax1.plot(time_seconds, mcp_abd_angles, 'g-', linewidth=1.5)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_ylabel('MCP Abduction (degrees)', fontsize=11)
        ax1.set_title('MCP Abduction/Adduction Over Time', fontsize=12)
        ax1.grid(True, alpha=0.3)

        ax2.plot(time_seconds, overall_abd_angles, 'm-', linewidth=1.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_ylabel('Overall Abduction (degrees)', fontsize=11)
        ax2.set_title('Overall Abduction Over Time', fontsize=12)
        ax2.grid(True, alpha=0.3)

        ax3.plot(time_seconds, mcp_flex_angles, 'b-', linewidth=1.5)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.set_ylabel('MCP Flexion (degrees)', fontsize=11)
        ax3.set_title('MCP Flexion (for comparison)', fontsize=12)
        ax3.set_xlabel('Time (seconds)', fontsize=11)
        ax3.grid(True, alpha=0.3)

        ax4.plot(time_seconds, overall_flex_angles, 'r-', linewidth=1.5)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax4.set_ylabel('Overall Flexion (degrees)', fontsize=11)
        ax4.set_title('Overall Flexion (for comparison)', fontsize=12)
        ax4.set_xlabel('Time (seconds)', fontsize=11)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        output1 = 'abduction_validation_timeseries.png'
        plt.savefig(output1, dpi=150)
        print(f"\nTime-series plot saved to: {output1}")

        # プロット2: 散布図（Flex vs Abd）
        fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))

        ax5.scatter(mcp_flex_angles, mcp_abd_angles, alpha=0.5, s=10)
        ax5.set_xlabel('MCP Flexion (degrees)', fontsize=11)
        ax5.set_ylabel('MCP Abduction (degrees)', fontsize=11)
        ax5.set_title(f'MCP: Flexion vs Abduction (r={corr_mcp:+.3f})', fontsize=12)
        ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax5.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax5.grid(True, alpha=0.3)

        ax6.scatter(overall_flex_angles, overall_abd_angles, alpha=0.5, s=10)
        ax6.set_xlabel('Overall Flexion (degrees)', fontsize=11)
        ax6.set_ylabel('Overall Abduction (degrees)', fontsize=11)
        ax6.set_title(f'Overall: Flexion vs Abduction (r={corr_overall:+.3f})', fontsize=12)
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax6.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        output2 = 'abduction_validation_scatter.png'
        plt.savefig(output2, dpi=150)
        print(f"Scatter plot saved to: {output2}")

        plt.close('all')

        # 評価
        print("\n=== Assessment ===")
        mcp_range = np.max(mcp_abd_angles) - np.min(mcp_abd_angles)
        overall_range = np.max(overall_abd_angles) - np.min(overall_abd_angles)

        print(f"MCP abduction range: {mcp_range:.2f}°")
        if mcp_range < 5:
            print("  → Very small (expected for grasping without abduction)")
        elif mcp_range < 15:
            print("  → Small to moderate")
        else:
            print("  → Large (may indicate hand rotation artifacts)")

        print(f"\nOverall abduction range: {overall_range:.2f}°")

        print(f"\nDifference (Overall - MCP): {overall_range - mcp_range:.2f}°")
        if abs(overall_range - mcp_range) < 5:
            print("  → Similar ranges (good - MCP controls abduction)")
        else:
            print("  → Large difference (may indicate calculation issue)")


def main():
    if len(sys.argv) < 2:
        data_dir = "data"
        if os.path.exists(data_dir):
            files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".h5")]
            if files:
                filepath = max(files, key=os.path.getctime)
                print(f"Using latest file: {filepath}\n")
            else:
                print("Usage: python validate_abduction_angles.py <path_to_h5_file> [left|right]")
                return
        else:
            print("Usage: python validate_abduction_angles.py <path_to_h5_file> [left|right]")
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
        validate_abduction_data(filepath, hand_type)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
