# -*- coding: utf-8 -*-
"""
USB-IO Monitor Test Script
Demonstrates event-driven pulse detection with callbacks.
"""

import time
from usb_io_monitor import USBIOMonitor, USBIOMonitorContext


# Callback function for edge detection
def on_edge_detected(edge_type: str, timestamp: float, pulse_width: float):
    """
    Called when an edge is detected.

    Args:
        edge_type: 'rising' or 'falling'
        timestamp: Time of edge detection (seconds since epoch)
        pulse_width: Pulse width in seconds (0 for falling edge)
    """
    if edge_type == 'falling':
        print(f"[{timestamp:.6f}] Pulse START (High→Low)")
    elif edge_type == 'rising':
        pulse_width_ms = pulse_width * 1000
        print(f"[{timestamp:.6f}] Pulse END (Low→High) - Width: {pulse_width_ms:.3f} ms")


def test_basic_monitoring():
    """Basic monitoring test."""
    print("=== Basic Monitoring Test ===")
    print("Monitoring J2-0 pin for 10 seconds...")
    print("Press Ctrl+C to stop early\n")

    monitor = USBIOMonitor(
        pin_mask=0x01,  # J2-0 pin
        poll_interval=0.0001,  # 100μs
        edge_callback=on_edge_detected
    )

    try:
        # Open and start monitoring
        if not monitor.open():
            print("Failed to open device")
            return

        monitor.start()

        # Monitor for 10 seconds
        start_time = time.time()
        while time.time() - start_time < 10.0:
            time.sleep(0.1)

        # Stop and show statistics
        monitor.stop()

        print("\n=== Statistics ===")
        stats = monitor.get_stats()
        print(f"Total pulses detected: {stats['total_pulses']}")
        print(f"Rising edges: {stats['rising_edges']}")
        print(f"Falling edges: {stats['falling_edges']}")

        if stats['total_pulses'] > 0:
            print(f"Min pulse width: {stats['min_pulse_width']*1000:.3f} ms")
            print(f"Max pulse width: {stats['max_pulse_width']*1000:.3f} ms")
            print(f"Avg pulse width: {stats['avg_pulse_width']*1000:.3f} ms")

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        monitor.close()


def test_context_manager():
    """Test using context manager."""
    print("\n=== Context Manager Test ===")
    print("Monitoring for 5 seconds using context manager...")
    print("Press Ctrl+C to stop early\n")

    pulse_count = [0]  # Use list for closure

    def count_pulses(edge_type, timestamp, pulse_width):
        if edge_type == 'rising':
            pulse_count[0] += 1
            print(f"Pulse #{pulse_count[0]}: {pulse_width*1000:.3f} ms")

    try:
        with USBIOMonitorContext(edge_callback=count_pulses) as monitor:
            time.sleep(5.0)

            print("\n=== Statistics ===")
            stats = monitor.get_stats()
            print(f"Total pulses: {stats['total_pulses']}")

    except KeyboardInterrupt:
        print("\nStopped by user")
    except RuntimeError as e:
        print(f"Error: {e}")


def test_continuous_monitoring():
    """Continuous monitoring test (runs until Ctrl+C)."""
    print("\n=== Continuous Monitoring Test ===")
    print("Monitoring J2-0 pin continuously...")
    print("Press Ctrl+C to stop\n")

    pulse_count = [0]
    last_print_time = [time.time()]

    def on_pulse(edge_type, timestamp, pulse_width):
        if edge_type == 'rising':
            pulse_count[0] += 1
            pulse_width_ms = pulse_width * 1000
            print(f"Pulse #{pulse_count[0]}: {pulse_width_ms:.3f} ms at {timestamp:.3f}")

    monitor = USBIOMonitor(
        pin_mask=0x01,
        poll_interval=0.0001,
        edge_callback=on_pulse
    )

    try:
        if not monitor.open():
            print("Failed to open device")
            return

        monitor.start()

        while True:
            time.sleep(1.0)

            # Print statistics every 10 seconds
            current_time = time.time()
            if current_time - last_print_time[0] >= 10.0:
                stats = monitor.get_stats()
                print(f"\n--- Statistics (last 10s) ---")
                print(f"Total pulses: {stats['total_pulses']}")
                if stats['total_pulses'] > 0:
                    print(f"Avg pulse width: {stats['avg_pulse_width']*1000:.3f} ms")
                print()
                last_print_time[0] = current_time

    except KeyboardInterrupt:
        print("\n\nStopping...")
        monitor.stop()

        print("\n=== Final Statistics ===")
        stats = monitor.get_stats()
        print(f"Total pulses detected: {stats['total_pulses']}")
        print(f"Rising edges: {stats['rising_edges']}")
        print(f"Falling edges: {stats['falling_edges']}")

        if stats['total_pulses'] > 0:
            print(f"Min pulse width: {stats['min_pulse_width']*1000:.3f} ms")
            print(f"Max pulse width: {stats['max_pulse_width']*1000:.3f} ms")
            print(f"Avg pulse width: {stats['avg_pulse_width']*1000:.3f} ms")

    finally:
        monitor.close()


if __name__ == "__main__":
    import sys

    print("USB-IO Event-Driven Monitor Test")
    print("=" * 50)

    if len(sys.argv) > 1:
        test_mode = sys.argv[1]
    else:
        print("\nAvailable tests:")
        print("  1. Basic monitoring (10 seconds)")
        print("  2. Context manager test (5 seconds)")
        print("  3. Continuous monitoring (until Ctrl+C)")
        print()
        test_mode = input("Select test (1-3, default=3): ").strip() or "3"

    if test_mode == "1":
        test_basic_monitoring()
    elif test_mode == "2":
        test_context_manager()
    elif test_mode == "3":
        test_continuous_monitoring()
    else:
        print(f"Unknown test mode: {test_mode}")
        sys.exit(1)
