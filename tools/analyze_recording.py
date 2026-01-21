# -*- coding: utf-8 -*-
"""
Leap Motion Recording Analyzer
Analyzes HDF5 recordings to detect frame drops and timing issues.
"""

import sys
import numpy as np
import h5py


def analyze_recording(filename):
    """
    Analyze a Leap Motion HDF5 recording file.

    Args:
        filename: Path to the HDF5 file
    """
    print(f"Analyzing: {filename}\n")

    with h5py.File(filename, 'r') as f:
        # Read metadata if available
        print("=== Recording Metadata ===")
        if 'total_frames_recorded' in f.attrs:
            print(f"Total frames recorded: {f.attrs['total_frames_recorded']}")
        if 'frames_dropped' in f.attrs:
            print(f"Frames dropped (during recording): {f.attrs['frames_dropped']}")
        if 'queue_size' in f.attrs:
            print(f"Queue size: {f.attrs['queue_size']}")
        if 'save_interval' in f.attrs:
            print(f"Save interval: {f.attrs['save_interval']} seconds")
        print()

        # Read timestamps
        timestamps = f['leap_timestamp'][:]
        task_status = f['task_status'][:]

        n_frames = len(timestamps)
        print(f"=== Frame Statistics ===")
        print(f"Total frames in file: {n_frames}")

        if n_frames < 2:
            print("Not enough frames to analyze timing.")
            return

        # Convert timestamps from microseconds to seconds
        timestamps_sec = timestamps / 1_000_000.0

        # Calculate time differences between consecutive frames
        time_diffs = np.diff(timestamps_sec)

        # Calculate statistics
        mean_interval = np.mean(time_diffs)
        std_interval = np.std(time_diffs)
        min_interval = np.min(time_diffs)
        max_interval = np.max(time_diffs)

        # Expected frame rate (assuming ~100 Hz)
        expected_interval = 0.01  # 10ms
        fps = 1.0 / mean_interval if mean_interval > 0 else 0

        print(f"Average frame rate: {fps:.2f} fps")
        print(f"Mean interval: {mean_interval*1000:.2f} ms")
        print(f"Std deviation: {std_interval*1000:.2f} ms")
        print(f"Min interval: {min_interval*1000:.2f} ms")
        print(f"Max interval: {max_interval*1000:.2f} ms")
        print()

        # Detect potential frame drops
        # A drop is suspected if interval is significantly larger than expected
        drop_threshold = expected_interval * 2.0  # 2x expected interval
        suspected_drops = time_diffs > drop_threshold
        n_suspected_drops = np.sum(suspected_drops)

        print(f"=== Frame Drop Detection ===")
        print(f"Drop threshold: {drop_threshold*1000:.2f} ms")
        print(f"Suspected drop events: {n_suspected_drops}")

        if n_suspected_drops > 0:
            drop_indices = np.where(suspected_drops)[0]

            # Calculate estimated frames lost
            estimated_frames_lost = 0
            print("\nDrop details (showing first 20):")
            print(f"{'Index':<8} {'Time (s)':<12} {'Interval (ms)':<15} {'Est. Frames Lost':<18}")
            print("-" * 60)

            for i, idx in enumerate(drop_indices[:20]):
                interval_ms = time_diffs[idx] * 1000
                frames_lost = int(time_diffs[idx] / expected_interval) - 1
                estimated_frames_lost += frames_lost
                time_sec = timestamps_sec[idx]

                print(f"{idx:<8} {time_sec:<12.3f} {interval_ms:<15.2f} {frames_lost:<18}")

            if len(drop_indices) > 20:
                print(f"... and {len(drop_indices) - 20} more drop events")

                # Calculate total estimated frames lost
                for idx in drop_indices[20:]:
                    frames_lost = int(time_diffs[idx] / expected_interval) - 1
                    estimated_frames_lost += frames_lost

            print()
            print(f"Total estimated frames lost: {estimated_frames_lost}")

            # Calculate drop rate
            total_expected_frames = n_frames + estimated_frames_lost
            drop_rate = 100.0 * estimated_frames_lost / total_expected_frames
            print(f"Estimated drop rate: {drop_rate:.2f}%")
        else:
            print("No significant frame drops detected.")

        print()

        # Task status analysis
        print("=== Task Status Analysis ===")
        task_on_frames = np.sum(task_status == 1)
        task_off_frames = np.sum(task_status == 0)

        print(f"Frames with task ON: {task_on_frames} ({100.0*task_on_frames/n_frames:.1f}%)")
        print(f"Frames with task OFF: {task_off_frames} ({100.0*task_off_frames/n_frames:.1f}%)")

        # Recording duration
        duration = timestamps_sec[-1] - timestamps_sec[0]
        print()
        print(f"=== Recording Duration ===")
        print(f"Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

        # File size estimate
        print()
        print(f"=== Data Completeness ===")
        left_valid = f['left_hand/valid'][:]
        right_valid = f['right_hand/valid'][:]

        left_valid_count = np.sum(left_valid)
        right_valid_count = np.sum(right_valid)

        print(f"Frames with left hand detected: {left_valid_count} ({100.0*left_valid_count/n_frames:.1f}%)")
        print(f"Frames with right hand detected: {right_valid_count} ({100.0*right_valid_count/n_frames:.1f}%)")
        print(f"Frames with both hands: {np.sum(left_valid & right_valid)} ({100.0*np.sum(left_valid & right_valid)/n_frames:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_recording.py <recording.h5>")
        print("\nExample:")
        print("  python analyze_recording.py data/leap_recording_20240101_120000.h5")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        analyze_recording(filename)
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
