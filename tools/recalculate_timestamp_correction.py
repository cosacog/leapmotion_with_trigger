# -*- coding: utf-8 -*-
"""
Leap Motion Timestamp Correction Tool

Recalculates leap_timestamp_corrected using a stable sync point
(after initial buffer flush period) instead of the first frame.

This fixes the linear drift issue caused by using unstable initial frames
as the synchronization reference point.

Usage:
    python recalculate_timestamp_correction.py <recording.h5> [--stable-delay 2.0] [--inplace]

Options:
    --stable-delay: Seconds to wait for stable sync point (default: 2.0)
    --inplace: Modify the original file instead of creating a new one
"""

import sys
import argparse
import numpy as np
import h5py
import os


def recalculate_timestamp_correction(
    filename: str,
    stable_delay_sec: float = 2.0,
    inplace: bool = False,
    verbose: bool = True
) -> dict:
    """
    Recalculate leap_timestamp_corrected using stable sync point.

    Args:
        filename: Path to the HDF5 file
        stable_delay_sec: Seconds after recording start to use as sync point
        inplace: If True, modify the original file. If False, create a new file.
        verbose: Print detailed information

    Returns:
        dict with correction statistics
    """
    if verbose:
        print(f"Processing: {filename}")
        print(f"Stable delay: {stable_delay_sec} seconds\n")

    # Determine output filename
    if inplace:
        output_filename = filename
        mode = 'r+'
    else:
        base, ext = os.path.splitext(filename)
        output_filename = f"{base}_corrected{ext}"
        # Copy file first
        import shutil
        shutil.copy2(filename, output_filename)
        mode = 'r+'
        if verbose:
            print(f"Output file: {output_filename}\n")

    stats = {}

    with h5py.File(output_filename, mode) as f:
        # Check required datasets
        if 'leap_timestamp' not in f:
            raise ValueError("leap_timestamp dataset not found")
        if 'system_timestamp' not in f:
            raise ValueError("system_timestamp dataset not found")

        leap_timestamps = f['leap_timestamp'][:]
        system_timestamps = f['system_timestamp'][:]

        n_frames = len(leap_timestamps)
        stats['n_frames'] = n_frames

        if n_frames < 2:
            raise ValueError("Not enough frames to calculate correction")

        # Calculate time from start in seconds
        time_from_start_s = (leap_timestamps - leap_timestamps[0]) / 1_000_000

        # Find stable sync point
        stable_start_idx = np.searchsorted(time_from_start_s, stable_delay_sec)

        if stable_start_idx >= n_frames - 1:
            if verbose:
                print(f"Warning: Recording shorter than {stable_delay_sec}s, using first frame")
            stable_start_idx = 0

        stats['stable_start_idx'] = int(stable_start_idx)
        stats['stable_start_time_s'] = float(time_from_start_s[stable_start_idx])

        # Get sync points
        leap_start_us = leap_timestamps[stable_start_idx]
        pc_start = system_timestamps[stable_start_idx]
        leap_end_us = leap_timestamps[-1]
        pc_end = system_timestamps[-1]

        # Calculate scale factor for drift correction
        leap_duration_s = (leap_end_us - leap_start_us) / 1_000_000

        if leap_duration_s <= 0:
            raise ValueError("Invalid timestamp range")

        scale = (pc_end - pc_start) / leap_duration_s
        stats['scale_factor'] = float(scale)
        stats['drift_ppm'] = float((scale - 1.0) * 1_000_000)

        if verbose:
            print("=== Sync Point Analysis ===")
            print(f"Stable start frame: {stable_start_idx}")
            print(f"Stable start time: {time_from_start_s[stable_start_idx]:.3f} s")
            print(f"Scale factor: {scale:.9f}")
            print(f"Clock drift: {stats['drift_ppm']:.1f} ppm")
            print()

        # Calculate corrected timestamps
        leap_timestamps_corrected = pc_start + ((leap_timestamps - leap_start_us) / 1_000_000) * scale

        # Compare with original correction if exists
        if 'leap_timestamp_corrected' in f:
            original_corrected = f['leap_timestamp_corrected'][:]
            diff = leap_timestamps_corrected - original_corrected

            stats['original_exists'] = True
            stats['max_diff_ms'] = float(np.max(np.abs(diff)) * 1000)
            stats['mean_diff_ms'] = float(np.mean(diff) * 1000)

            if verbose:
                print("=== Comparison with Original Correction ===")
                print(f"Max absolute difference: {stats['max_diff_ms']:.3f} ms")
                print(f"Mean difference: {stats['mean_diff_ms']:.3f} ms")
                print(f"Difference at start: {diff[0]*1000:.3f} ms")
                print(f"Difference at end: {diff[-1]*1000:.3f} ms")
                print()

            # Delete old dataset
            del f['leap_timestamp_corrected']
        else:
            stats['original_exists'] = False

        # Save new corrected timestamps
        dset = f.create_dataset('leap_timestamp_corrected', data=leap_timestamps_corrected, dtype='f8')
        dset.attrs['description'] = 'Leap timestamp converted to PC time with drift correction (no USB latency)'
        dset.attrs['unit'] = 'seconds (perf_counter)'
        dset.attrs['note'] = 'Use this for accurate temporal alignment with trigger_onset_times_corrected'
        dset.attrs['stable_start_idx'] = stable_start_idx
        dset.attrs['stable_delay_sec'] = stable_delay_sec
        dset.attrs['recalculated'] = True

        # Update leap_sync if exists
        if 'leap_sync' in f:
            leap_sync = f['leap_sync']
            leap_sync.attrs['stable_start_idx'] = stable_start_idx
            leap_sync.attrs['stable_start_leap_us'] = int(leap_start_us)
            leap_sync.attrs['stable_start_pc_time'] = float(pc_start)
            leap_sync.attrs['corrected_scale'] = float(scale)

        # Validate correction quality
        if verbose:
            print("=== Correction Validation ===")
            # Check linearity of corrected - system difference
            diff_corrected_system = leap_timestamps_corrected - system_timestamps

            # After stable point, the difference should be relatively constant
            stable_diff = diff_corrected_system[stable_start_idx:]
            diff_std = np.std(stable_diff) * 1000  # ms
            diff_range = (np.max(stable_diff) - np.min(stable_diff)) * 1000  # ms

            stats['stable_diff_std_ms'] = float(diff_std)
            stats['stable_diff_range_ms'] = float(diff_range)

            print(f"Stable region diff std: {diff_std:.3f} ms")
            print(f"Stable region diff range: {diff_range:.3f} ms")

            # Check for remaining linear trend
            if len(stable_diff) > 10:
                x = np.arange(len(stable_diff))
                slope, intercept = np.polyfit(x, stable_diff, 1)
                trend_per_sec = slope * (1000 / np.mean(np.diff(time_from_start_s[stable_start_idx:])))
                stats['residual_trend_ms_per_sec'] = float(trend_per_sec * 1000)
                print(f"Residual trend: {trend_per_sec*1000:.3f} ms/s")

            print()

        if verbose:
            print(f"Successfully saved corrected timestamps to: {output_filename}")

    return stats


def analyze_timestamp_quality(filename: str, verbose: bool = True) -> dict:
    """
    Analyze timestamp correction quality without modifying the file.

    Args:
        filename: Path to the HDF5 file
        verbose: Print detailed information

    Returns:
        dict with analysis results
    """
    if verbose:
        print(f"Analyzing: {filename}\n")

    stats = {}

    with h5py.File(filename, 'r') as f:
        leap_timestamps = f['leap_timestamp'][:]
        system_timestamps = f['system_timestamp'][:]

        n_frames = len(leap_timestamps)
        stats['n_frames'] = n_frames

        # Frame interval analysis
        leap_interval_ms = np.diff(leap_timestamps) / 1000  # μs -> ms
        system_interval_ms = np.diff(system_timestamps) * 1000  # s -> ms

        if verbose:
            print("=== Frame Interval Analysis ===")
            print(f"Leap interval - mean: {np.mean(leap_interval_ms):.2f} ms, std: {np.std(leap_interval_ms):.2f} ms")
            print(f"System interval - mean: {np.mean(system_interval_ms):.2f} ms, std: {np.std(system_interval_ms):.2f} ms")
            print()

        # Detect buffer flush period
        # During buffer flush, system_interval is much smaller than leap_interval
        interval_diff = leap_interval_ms - system_interval_ms
        flush_threshold = np.mean(leap_interval_ms) * 0.5  # 50% of mean leap interval

        flush_frames = np.where(interval_diff > flush_threshold)[0]
        if len(flush_frames) > 0:
            flush_end_frame = flush_frames[-1] + 1 if flush_frames[-1] < n_frames // 2 else 0
            flush_duration_s = (leap_timestamps[flush_end_frame] - leap_timestamps[0]) / 1_000_000
        else:
            flush_end_frame = 0
            flush_duration_s = 0

        stats['flush_end_frame'] = int(flush_end_frame)
        stats['flush_duration_s'] = float(flush_duration_s)

        if verbose:
            print("=== Buffer Flush Detection ===")
            print(f"Flush end frame: {flush_end_frame}")
            print(f"Flush duration: {flush_duration_s:.3f} s")
            print()

        # Check existing correction
        if 'leap_timestamp_corrected' in f:
            corrected = f['leap_timestamp_corrected'][:]
            diff = corrected - system_timestamps

            if verbose:
                print("=== Existing Correction Quality ===")
                print(f"Diff at start: {diff[0]*1000:.3f} ms")
                print(f"Diff at end: {diff[-1]*1000:.3f} ms")
                print(f"Diff change (start to end): {(diff[-1] - diff[0])*1000:.3f} ms")

                # Check for linear trend (indicates poor sync point)
                x = np.arange(len(diff))
                slope, _ = np.polyfit(x, diff, 1)
                time_span_s = (leap_timestamps[-1] - leap_timestamps[0]) / 1_000_000
                trend_ms_per_sec = slope * n_frames / time_span_s * 1000

                stats['trend_ms_per_sec'] = float(trend_ms_per_sec)
                print(f"Linear trend: {trend_ms_per_sec:.3f} ms/s")

                if abs(trend_ms_per_sec) > 0.1:
                    print(">> Warning: Significant linear trend detected. Recalculation recommended.")
                print()

            # Check stable_start_idx attribute
            if 'stable_start_idx' in f['leap_timestamp_corrected'].attrs:
                stats['current_stable_idx'] = int(f['leap_timestamp_corrected'].attrs['stable_start_idx'])
                if verbose:
                    print(f"Current stable_start_idx: {stats['current_stable_idx']}")
            else:
                stats['current_stable_idx'] = None
                if verbose:
                    print("No stable_start_idx found (likely using first frame)")
        else:
            if verbose:
                print("No leap_timestamp_corrected found in file")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Recalculate Leap Motion timestamp correction using stable sync point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze existing correction quality
  python recalculate_timestamp_correction.py data/recording.h5 --analyze

  # Recalculate and save to new file
  python recalculate_timestamp_correction.py data/recording.h5

  # Recalculate with custom stable delay
  python recalculate_timestamp_correction.py data/recording.h5 --stable-delay 1.5

  # Modify original file in place
  python recalculate_timestamp_correction.py data/recording.h5 --inplace
        """
    )
    parser.add_argument('filename', help='Path to HDF5 recording file')
    parser.add_argument('--stable-delay', type=float, default=2.0,
                        help='Seconds to wait for stable sync point (default: 2.0)')
    parser.add_argument('--inplace', action='store_true',
                        help='Modify original file instead of creating new one')
    parser.add_argument('--analyze', action='store_true',
                        help='Only analyze, do not modify')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed output')

    args = parser.parse_args()

    try:
        if args.analyze:
            analyze_timestamp_quality(args.filename, verbose=not args.quiet)
        else:
            recalculate_timestamp_correction(
                args.filename,
                stable_delay_sec=args.stable_delay,
                inplace=args.inplace,
                verbose=not args.quiet
            )
    except FileNotFoundError:
        print(f"Error: File not found: {args.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
