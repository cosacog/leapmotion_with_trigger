# -*- coding: utf-8 -*-
"""
Simplified test version of record_with_trigger.py
For debugging Leap Motion connection and display issues.
"""

import time
import signal
import leap
import cv2
import numpy as np
from usb_io_monitor import USBIOMonitor
from timestamp_sync import TimestampConverter, HighPrecisionTimer

# Global state
is_recording = True
trigger_status = 0
trigger_count = 0
frame_count = 0
high_precision_timer = None
timestamp_converter = TimestampConverter()

def on_trigger_edge(edge_type: str, timestamp: float, pulse_width: float):
    """USB-IO trigger callback."""
    global trigger_status, trigger_count, high_precision_timer

    if high_precision_timer:
        system_timestamp = high_precision_timer.perf_to_time(timestamp)
    else:
        system_timestamp = timestamp

    if edge_type == 'falling':
        trigger_status = 1
        print(f"[TRIGGER] START at {system_timestamp:.6f}")
    elif edge_type == 'rising':
        trigger_status = 0
        trigger_count += 1
        print(f"[TRIGGER] END #{trigger_count}: {pulse_width*1000:.3f} ms")

class SimpleListener(leap.Listener):
    """Simple Leap Motion listener with debug output."""

    def on_connection_event(self, event):
        print(f"[LEAP] Connected")

    def on_device_event(self, event):
        print(f"[LEAP] Device connected: {event.device}")

    def on_tracking_event(self, event):
        global frame_count, timestamp_converter

        # Calibrate timestamp on first frame
        if not timestamp_converter.is_calibrated():
            system_time = timestamp_converter.calibrate(event.timestamp)
            print(f"[LEAP] Timestamp calibrated: offset = {timestamp_converter.get_offset_ms():.3f} ms")
        else:
            system_time = timestamp_converter.leap_to_system(event.timestamp)

        frame_count += 1

        # Print every 30 frames (~0.33s at 90Hz)
        if frame_count % 30 == 0:
            print(f"[LEAP] Frame {frame_count}: {len(event.hands)} hands, timestamp={system_time:.6f}")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global is_recording
    print("\n\n[SIGNAL] Ctrl+C detected! Exiting...")
    is_recording = False

def main():
    global is_recording, high_precision_timer

    print("=== Leap Motion + USB-IO Test ===")
    print("This is a simplified test version.")
    print("Press Ctrl+C to stop.\n")

    # Initialize high precision timer
    high_precision_timer = HighPrecisionTimer()
    print(f"[INIT] High precision timer initialized")

    # Initialize USB-IO monitor
    usb_io_monitor = None
    try:
        usb_io_monitor = USBIOMonitor(
            pin_mask=0x01,
            poll_interval=0.0001,
            edge_callback=on_trigger_edge,
            use_high_precision=True
        )

        if usb_io_monitor.open():
            usb_io_monitor.start()
            print(f"[INIT] USB-IO monitor started")
        else:
            print(f"[INIT] USB-IO not available (will continue without it)")
            usb_io_monitor = None
    except Exception as e:
        print(f"[INIT] USB-IO error: {e}")
        usb_io_monitor = None

    # Initialize Leap Motion
    listener = SimpleListener()
    connection = leap.Connection()
    connection.add_listener(listener)

    print(f"[INIT] Leap Motion listener added")

    # Create simple canvas
    canvas_name = "Leap Test"
    screen_size = [400, 600]

    try:
        with connection.open():
            print(f"[INIT] Leap Motion connection opened")
            print(f"\nWaiting for data...")
            print(f"- Leap Motion frames will be printed every ~0.33s")
            print(f"- USB-IO triggers will be printed immediately")
            print(f"- Press 'q' or ESC in window to exit")
            print(f"- Or close the window to exit\n")

            while is_recording:
                # Create simple display
                img = np.zeros((screen_size[0], screen_size[1], 3), np.uint8)

                # Show status
                cv2.putText(img, f"Frames: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(img, f"Triggers: {trigger_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                trigger_text = "ACTIVE" if trigger_status else "IDLE"
                trigger_color = (0, 255, 255) if trigger_status else (128, 128, 128)
                cv2.putText(img, f"Trigger: {trigger_text}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, trigger_color, 2)

                cv2.putText(img, "Press 'q' or ESC to quit", (10, screen_size[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow(canvas_name, img)

                key = cv2.waitKey(10)
                if key & 0xFF == ord('q') or key == 27:
                    print("\n[EXIT] User pressed 'q'")
                    is_recording = False
                    break

                # Check if window was closed
                try:
                    if cv2.getWindowProperty(canvas_name, cv2.WND_PROP_VISIBLE) < 1:
                        print("\n[EXIT] Window closed")
                        is_recording = False
                        break
                except:
                    pass

    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted by user (Ctrl+C)")
        is_recording = False

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        is_recording = False

    finally:
        is_recording = False

        # Cleanup
        if usb_io_monitor:
            usb_io_monitor.stop()
            usb_io_monitor.close()
            print(f"[CLEANUP] USB-IO monitor stopped")

        connection.remove_listener(listener)
        cv2.destroyAllWindows()

        print(f"\n=== Summary ===")
        print(f"Total Leap frames: {frame_count}")
        print(f"Total USB-IO triggers: {trigger_count}")

        if timestamp_converter.is_calibrated():
            stats = timestamp_converter.get_stats()
            print(f"Timestamp offset: {stats['offset_ms']:.3f} ms")

if __name__ == "__main__":
    # Setup signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    try:
        main()
    except KeyboardInterrupt:
        print("\n[EXIT] KeyboardInterrupt")
        is_recording = False
