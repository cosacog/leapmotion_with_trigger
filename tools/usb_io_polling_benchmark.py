# -*- coding: utf-8 -*-
"""
USB-IO Polling Interval Benchmark Tool

This script measures the actual polling intervals of the USB-IO monitor
to verify if the Windows high resolution timer is working correctly.

It compares polling performance with and without the high resolution timer.

Usage:
    python usb_io_polling_benchmark.py [--duration SECONDS]

Example output:
    === Without High Resolution Timer ===
    Polling interval analysis:
      Requested: 0.10 ms
      Actual avg: 15.62 ms
      Min: 14.89 ms, Max: 31.25 ms
      ...

    === With High Resolution Timer ===
    Polling interval analysis:
      Requested: 0.10 ms
      Actual avg: 1.05 ms
      Min: 0.95 ms, Max: 2.10 ms
      ...
"""

import sys
import os
import time
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from usb_io_monitor import (
    USBIOMonitor,
    enable_high_resolution_timer,
    disable_high_resolution_timer
)


def print_analysis(analysis: dict, title: str):
    """Print polling interval analysis in a formatted way."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

    if 'error' in analysis:
        print(f"  Error: {analysis['error']}")
        return

    print(f"\n  Polling Interval Statistics:")
    print(f"    Requested interval: {analysis['requested_interval_ms']:.2f} ms")
    print(f"    Actual average:     {analysis['avg_ms']:.2f} ms")
    print(f"    Minimum:            {analysis['min_ms']:.2f} ms")
    print(f"    Maximum:            {analysis['max_ms']:.2f} ms")
    print(f"    Median:             {analysis['median_ms']:.2f} ms")
    print(f"    95th percentile:    {analysis['p95_ms']:.2f} ms")
    print(f"    99th percentile:    {analysis['p99_ms']:.2f} ms")
    print(f"    Effective rate:     {analysis['effective_rate_hz']:.1f} Hz")
    print(f"    Sample count:       {analysis['sample_count']}")

    print(f"\n  Interval Distribution:")
    for range_str, count in analysis['histogram'].items():
        pct = 100 * count / analysis['sample_count'] if analysis['sample_count'] > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"    {range_str:12s}: {count:5d} ({pct:5.1f}%) {bar}")


def run_benchmark(duration: float, use_high_res: bool) -> dict:
    """
    Run a benchmark test.

    Args:
        duration: Test duration in seconds
        use_high_res: Whether to enable high resolution timer

    Returns:
        Polling interval analysis dictionary
    """
    pulse_count = 0

    def on_edge(edge_type: str, timestamp: float, pulse_width: float):
        nonlocal pulse_count
        if edge_type == 'falling':
            pulse_count += 1
            print(f"    Pulse #{pulse_count}: width = {pulse_width*1000:.2f} ms")

    monitor = USBIOMonitor(
        pin_mask=0x01,
        poll_interval=0.0001,  # 100μs requested
        edge_callback=on_edge,
        use_high_precision=True
    )

    if not monitor.open():
        print("  Failed to open USB-IO device!")
        return {'error': 'Device open failed'}

    try:
        monitor.start()
        print(f"  Monitoring for {duration} seconds...")
        print(f"  (Press Ctrl+C to stop early)")

        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(0.1)

        monitor.stop()

        analysis = monitor.get_poll_interval_analysis()
        analysis['pulses_detected'] = pulse_count
        return analysis

    except KeyboardInterrupt:
        print("\n  Interrupted by user")
        monitor.stop()
        return monitor.get_poll_interval_analysis()
    finally:
        monitor.close()


def main():
    parser = argparse.ArgumentParser(
        description='USB-IO Polling Interval Benchmark Tool'
    )
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=10.0,
        help='Test duration in seconds (default: 10)'
    )
    parser.add_argument(
        '--skip-normal',
        action='store_true',
        help='Skip the test without high resolution timer'
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print(" USB-IO Polling Interval Benchmark")
    print("="*60)
    print(f"\nThis tool measures actual polling intervals to diagnose")
    print(f"pulse detection issues. A 15ms pulse requires polling")
    print(f"faster than 15ms to reliably detect it.")
    print(f"\nTest duration: {args.duration} seconds per test")

    results = {}

    # Test 1: Without high resolution timer
    if not args.skip_normal:
        print("\n" + "-"*60)
        print(" Test 1: WITHOUT High Resolution Timer")
        print("-"*60)
        print("  (Using Windows default ~15.6ms timer resolution)")

        disable_high_resolution_timer()  # Ensure it's disabled
        results['normal'] = run_benchmark(args.duration, use_high_res=False)
        print_analysis(results['normal'], "Results: WITHOUT High Resolution Timer")

    # Test 2: With high resolution timer
    print("\n" + "-"*60)
    print(" Test 2: WITH High Resolution Timer (1ms)")
    print("-"*60)

    if enable_high_resolution_timer():
        results['high_res'] = run_benchmark(args.duration, use_high_res=True)
        print_analysis(results['high_res'], "Results: WITH High Resolution Timer")
        disable_high_resolution_timer()
    else:
        print("  Failed to enable high resolution timer!")
        results['high_res'] = {'error': 'Timer enable failed'}

    # Summary
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)

    if not args.skip_normal and 'normal' in results and 'error' not in results['normal']:
        normal_avg = results['normal']['avg_ms']
        print(f"\n  Without high-res timer:")
        print(f"    Average polling interval: {normal_avg:.2f} ms")
        if normal_avg > 10:
            print(f"    WARNING: Interval too slow to detect 15ms pulses reliably!")

    if 'high_res' in results and 'error' not in results['high_res']:
        high_res_avg = results['high_res']['avg_ms']
        print(f"\n  With high-res timer:")
        print(f"    Average polling interval: {high_res_avg:.2f} ms")

        if high_res_avg < 5:
            print(f"    OK: Should reliably detect 15ms pulses")
        elif high_res_avg < 10:
            print(f"    MARGINAL: May occasionally miss 15ms pulses")
        else:
            print(f"    WARNING: Still too slow - check USB HID overhead")

        # Recommendation
        print(f"\n  Recommendation:")
        if high_res_avg < 5:
            print(f"    Use enable_high_resolution_timer() at program start")
            print(f"    Add to record_with_trigger.py for reliable pulse detection")
        else:
            print(f"    Consider increasing pulse width to 50ms or more")
            print(f"    USB HID communication overhead may be limiting factor")

    print()


if __name__ == '__main__':
    main()
