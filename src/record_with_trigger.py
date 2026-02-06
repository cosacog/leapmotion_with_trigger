# -*- coding: utf-8 -*-
"""
Leap Motion Hand Tracking Recorder with External Trigger Support
Records Leap Motion data with external trigger input via USB-IO or Arduino.

Usage:
    python record_with_trigger.py [--trigger usb-io|arduino|none] [--port COM9]

Trigger sources:
    usb-io  : USB-IO device (default, 100μs polling)
    arduino : Arduino Nano with hardware interrupt (μs precision)
    none    : No external trigger

HDF5 Output Structure:
    Timestamps (per frame):
        leap_timestamp           : Leap Motion internal clock (microseconds, int64)
        system_timestamp         : PC time when frame was received (perf_counter seconds, float64)
        leap_timestamp_corrected : Leap timestamp converted to PC time with drift correction
                                   (perf_counter seconds, float64)

    Trigger data:
        trigger_onset_times           : PC time when trigger was received via serial
                                        (perf_counter seconds, float64)
        trigger_onset_times_corrected : Arduino timestamp converted to PC time with drift correction
                                        (perf_counter seconds, float64)
        arduino_trigger_times_us      : Raw Arduino micros() timestamps (microseconds, int64)

    Sync groups (for post-processing):
        leap_sync/    : Leap Motion clock synchronization points
        arduino_sync/ : Arduino clock synchronization points

    Time alignment notes:
        - leap_timestamp_corrected and trigger_onset_times_corrected use the same
          PC time base (perf_counter), enabling accurate temporal alignment
        - Linear interpolation between start/end sync points corrects clock drift
        - USB/serial communication delays (2-8ms typical) are eliminated in corrected values
"""

import sys
import time
import queue
import threading
import datetime
import os
import signal
import argparse
import numpy as np
import h5py
import leap
import cv2
from pynput import keyboard

# Conditional imports for trigger sources
try:
    from usb_io_monitor import (
        USBIOMonitor,
        enable_high_resolution_timer,
        disable_high_resolution_timer
    )
    USB_IO_AVAILABLE = True
except ImportError:
    USB_IO_AVAILABLE = False
    def enable_high_resolution_timer(): pass
    def disable_high_resolution_timer(): pass

try:
    from arduino_trigger_monitor import ArduinoTriggerMonitor
    ARDUINO_AVAILABLE = True
except ImportError:
    ARDUINO_AVAILABLE = False

# Configuration
OUTPUT_DIR = 'data'
FILENAME_PREFIX = 'leap_recording_trigger_'
SAVE_INTERVAL = 0.5
QUEUE_SIZE = 10000

# USB-IO Configuration
USB_IO_PIN_MASK = 0x01  # J2-0 pin
USB_IO_POLL_INTERVAL = 0.0001  # 100μs

# Data Schema
class HandData:
    def __init__(self):
        self.valid = False
        self.palm_pos = [0.0, 0.0, 0.0]
        self.palm_ori = [0.0, 0.0, 0.0, 0.0]
        self.wrist_pos = [0.0, 0.0, 0.0]
        self.elbow_pos = [0.0, 0.0, 0.0]
        self.fingers = np.zeros((5, 4, 2, 3), dtype=np.float32)

class FrameData:
    def __init__(self, leap_timestamp, system_timestamp, task_status, trigger_status):
        self.leap_timestamp = leap_timestamp  # Original Leap timestamp (microseconds)
        self.system_timestamp = system_timestamp  # Synchronized system time (seconds)
        self.task_status = task_status  # Keyboard SPACE key
        self.trigger_status = trigger_status  # USB-IO trigger
        self.left = HandData()
        self.right = HandData()

class LiveCanvas:
    def __init__(self):
        self.name = "Leap Live View"
        self.screen_size = [600, 800]
        self.hands_colour = (255, 255, 255)
        self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)
        self.scale = 1.0
        self.offset_x = self.screen_size[1] // 2
        self.offset_y = self.screen_size[0] // 2

    def to_screen_coords(self, pos_3d):
        x = int(pos_3d[0] * self.scale + self.offset_x)
        y = int(pos_3d[2] * self.scale + self.offset_y)
        return x, y

    def render_frame(self, frame_data, queue_size=0, drop_count=0, trigger_count=0):
        self.output_image[:, :] = 0

        if frame_data is None:
            cv2.putText(self.output_image, "Waiting for data...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return self.output_image

        # Draw Status
        task_status = frame_data.task_status
        trigger_status = frame_data.trigger_status
        timestamp = frame_data.leap_timestamp

        # Task status (SPACE key)
        status_text = f"Time: {timestamp} | Task: {'ON' if task_status else 'OFF'}"
        color = (0, 0, 255) if task_status else (0, 255, 0)
        cv2.putText(self.output_image, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Trigger status (USB-IO)
        trigger_text = f"Trigger: {'ACTIVE' if trigger_status else 'IDLE'} (Count: {trigger_count})"
        trigger_color = (0, 255, 255) if trigger_status else (128, 128, 128)
        cv2.putText(self.output_image, trigger_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, trigger_color, 2)

        # Queue and drop info
        queue_text = f"Queue: {queue_size}/{QUEUE_SIZE} | Drops: {drop_count}"
        queue_color = (0, 255, 255) if queue_size < QUEUE_SIZE * 0.8 else (0, 165, 255)
        cv2.putText(self.output_image, queue_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, queue_color, 1)

        # Draw Hands
        self._draw_hand(frame_data.left, "L")
        self._draw_hand(frame_data.right, "R")

        return self.output_image

    def _draw_hand(self, hand_data, label):
        if not hand_data.valid:
            return

        wrist_pos = hand_data.wrist_pos
        elbow_pos = hand_data.elbow_pos

        sx_wrist, sy_wrist = self.to_screen_coords(wrist_pos)
        sx_elbow, sy_elbow = self.to_screen_coords(elbow_pos)

        cv2.circle(self.output_image, (sx_wrist, sy_wrist), 4, (0, 255, 255), -1)
        cv2.circle(self.output_image, (sx_elbow, sy_elbow), 4, (0, 255, 255), -1)
        cv2.line(self.output_image, (sx_wrist, sy_wrist), (sx_elbow, sy_elbow),
                self.hands_colour, 2)

        cv2.putText(self.output_image, label, (sx_wrist - 10, sy_wrist - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        fingers = hand_data.fingers

        for i in range(5):
            for j in range(4):
                bone_start = fingers[i, j, 0]
                bone_end = fingers[i, j, 1]

                s_start = self.to_screen_coords(bone_start)
                s_end = self.to_screen_coords(bone_end)

                cv2.line(self.output_image, s_start, s_end, self.hands_colour, 2)
                cv2.circle(self.output_image, s_start, 2, self.hands_colour, -1)

                if j == 3:
                    cv2.circle(self.output_image, s_end, 3, self.hands_colour, -1)

                if j == 0:
                    cv2.line(self.output_image, (sx_wrist, sy_wrist), s_start,
                            self.hands_colour, 1)

# Global State
is_recording = True
task_status = 0  # SPACE key
trigger_status = 0  # External trigger
frame_drop_count = 0
trigger_pulse_count = 0
trigger_timestamps = []  # TTL pulse onset times (perf_counter)
trigger_source = 'none'  # 'usb-io', 'arduino', or 'none'
arduino_monitor = None  # Arduino monitor instance (for sync data)
leap_listener = None  # Leap Motion listener instance (for sync data)

class LatestFrameContainer:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def set(self, frame):
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            return self._frame

latest_frame_container = LatestFrameContainer()
data_queue = queue.Queue(maxsize=QUEUE_SIZE)

# USB-IO Edge Detection Callback
def on_usb_io_trigger_edge(edge_type: str, timestamp: float, pulse_width: float):
    """
    Called when USB-IO trigger edge is detected.

    Args:
        timestamp: perf_counter timestamp from USBIOMonitor
        pulse_width: Pulse width in seconds
    """
    global trigger_status, trigger_pulse_count, trigger_timestamps

    if edge_type == 'rising':
        trigger_status = 1  # Trigger ACTIVE
        trigger_timestamps.append(timestamp)  # Record perf_counter timestamp directly
        print(f"[USB-IO] Pulse START (perf_counter: {timestamp:.6f})")
    elif edge_type == 'falling':
        trigger_status = 0  # Trigger IDLE
        trigger_pulse_count += 1
        pulse_width_ms = pulse_width * 1000
        print(f"[USB-IO] Pulse END (#{trigger_pulse_count}) - Width: {pulse_width_ms:.3f} ms")


# Arduino Trigger Callback
def on_arduino_trigger(arduino_time_us: int, pc_time: float):
    """
    Called when Arduino detects a trigger.

    Args:
        arduino_time_us: Arduino micros() timestamp
        pc_time: PC perf_counter at receive time
    """
    global trigger_status, trigger_pulse_count, trigger_timestamps

    trigger_status = 1  # Trigger ACTIVE (brief)
    trigger_timestamps.append(pc_time)  # Use PC receive time for now
    trigger_pulse_count += 1
    print(f"[Arduino] Trigger #{trigger_pulse_count} (arduino={arduino_time_us}us, pc={pc_time:.6f}s)")

    # Auto-reset trigger status after short delay (for display)
    def reset_status():
        global trigger_status
        time.sleep(0.05)
        trigger_status = 0
    threading.Thread(target=reset_status, daemon=True).start()

class RecordingListener(leap.Listener):
    def __init__(self):
        super().__init__()
        self._frame_count = 0
        self._first_leap_time = None
        self._first_system_time = None
        self._last_leap_time = None
        self._last_system_time = None

    def on_connection_event(self, event):
        print("[LEAP] Connection event received")

    def on_device_event(self, event):
        print(f"[LEAP] Device event: {event.device}")

    def on_tracking_event(self, event):
        global frame_drop_count
        if not is_recording:
            return

        self._frame_count += 1

        # Use perf_counter directly for consistent timing with TTL trigger
        system_time = time.perf_counter()

        if self._first_leap_time is None:
            self._first_leap_time = event.timestamp
            self._first_system_time = system_time
            print(f"[LEAP] First frame received! system_time base: {system_time:.6f}")

        # Always update last timestamps for sync point calculation
        self._last_leap_time = event.timestamp
        self._last_system_time = system_time

        # Debug: print every 90 frames (1 second at 90Hz)
        if self._frame_count % 90 == 0:
            print(f"[LEAP] Frame {self._frame_count}: {len(event.hands)} hands detected")

        frame_data = FrameData(event.timestamp, system_time, task_status, trigger_status)

        for hand in event.hands:
            h_data = frame_data.left if str(hand.type) == "HandType.Left" else frame_data.right
            h_data.valid = True
            h_data.palm_pos = [hand.palm.position.x, hand.palm.position.y, hand.palm.position.z]
            h_data.palm_ori = [hand.palm.orientation.x, hand.palm.orientation.y,
                              hand.palm.orientation.z, hand.palm.orientation.w]

            wrist = hand.arm.next_joint
            elbow = hand.arm.prev_joint
            h_data.wrist_pos = [wrist.x, wrist.y, wrist.z]
            h_data.elbow_pos = [elbow.x, elbow.y, elbow.z]

            for i, digit in enumerate(hand.digits):
                for j, bone in enumerate(digit.bones):
                    prev = bone.prev_joint
                    next = bone.next_joint
                    h_data.fingers[i, j, 0] = [prev.x, prev.y, prev.z]
                    h_data.fingers[i, j, 1] = [next.x, next.y, next.z]

        latest_frame_container.set(frame_data)

        try:
            data_queue.put_nowait(frame_data)
        except queue.Full:
            frame_drop_count += 1
            if frame_drop_count % 10 == 1:
                print(f"Warning: Data queue full, dropping frame! (Total drops: {frame_drop_count})")

def writer_thread_func(filename):
    global is_recording, frame_drop_count, trigger_pulse_count

    with h5py.File(filename, 'w') as f:
        chunk_size = 1000
        dset_lts = f.create_dataset('leap_timestamp', (0,), maxshape=(None,),
                                    dtype='i8', chunks=(chunk_size,))
        dset_lts.attrs['description'] = 'Leap Motion internal clock timestamp'
        dset_lts.attrs['unit'] = 'microseconds'

        dset_sys_time = f.create_dataset('system_timestamp', (0,), maxshape=(None,),
                                         dtype='f8', chunks=(chunk_size,))
        dset_sys_time.attrs['description'] = 'PC time when frame was received (includes USB latency)'
        dset_sys_time.attrs['unit'] = 'seconds (perf_counter)'

        dset_task = f.create_dataset('task_status', (0,), maxshape=(None,),
                                     dtype='i1', chunks=(chunk_size,))
        dset_task.attrs['description'] = 'Keyboard SPACE key status (1=pressed, 0=released)'

        dset_trigger = f.create_dataset('trigger_status', (0,), maxshape=(None,),
                                       dtype='i1', chunks=(chunk_size,))
        dset_trigger.attrs['description'] = 'External TTL trigger status (1=active, 0=idle)'

        def create_hand_group(group_name):
            g = f.create_group(group_name)
            g.create_dataset('valid', (0,), maxshape=(None,), dtype='bool', chunks=(chunk_size,))
            g.create_dataset('palm_pos', (0, 3), maxshape=(None, 3), dtype='f4', chunks=(chunk_size, 3))
            g.create_dataset('palm_ori', (0, 4), maxshape=(None, 4), dtype='f4', chunks=(chunk_size, 4))
            g.create_dataset('wrist_pos', (0, 3), maxshape=(None, 3), dtype='f4', chunks=(chunk_size, 3))
            g.create_dataset('elbow_pos', (0, 3), maxshape=(None, 3), dtype='f4', chunks=(chunk_size, 3))
            g.create_dataset('fingers', (0, 5, 4, 2, 3), maxshape=(None, 5, 4, 2, 3),
                           dtype='f4', chunks=(chunk_size, 5, 4, 2, 3))
            return g

        g_right = create_hand_group('right_hand')
        g_left = create_hand_group('left_hand')

        class HandBuffers:
            def __init__(self):
                self.valid = []
                self.palm_pos = []
                self.palm_ori = []
                self.wrist_pos = []
                self.elbow_pos = []
                self.fingers = []

            def append(self, h_data):
                self.valid.append(h_data.valid)
                self.palm_pos.append(h_data.palm_pos)
                self.palm_ori.append(h_data.palm_ori)
                self.wrist_pos.append(h_data.wrist_pos)
                self.elbow_pos.append(h_data.elbow_pos)
                self.fingers.append(h_data.fingers)

            def clear(self):
                self.valid = []
                self.palm_pos = []
                self.palm_ori = []
                self.wrist_pos = []
                self.elbow_pos = []
                self.fingers = []

        b_lts, b_sys_time, b_task, b_trigger = [], [], [], []
        b_right = HandBuffers()
        b_left = HandBuffers()

        last_save_time = time.time()

        while is_recording or not data_queue.empty():
            try:
                frame = data_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            b_lts.append(frame.leap_timestamp)
            b_sys_time.append(frame.system_timestamp)
            b_task.append(frame.task_status)
            b_trigger.append(frame.trigger_status)
            b_right.append(frame.right)
            b_left.append(frame.left)

            if time.time() - last_save_time >= SAVE_INTERVAL:
                if len(b_lts) > 0:
                    n_new = len(b_lts)
                    n_curr = dset_lts.shape[0]

                    dset_lts.resize(n_curr + n_new, axis=0)
                    dset_lts[-n_new:] = b_lts

                    dset_sys_time.resize(n_curr + n_new, axis=0)
                    dset_sys_time[-n_new:] = b_sys_time

                    dset_task.resize(n_curr + n_new, axis=0)
                    dset_task[-n_new:] = b_task

                    dset_trigger.resize(n_curr + n_new, axis=0)
                    dset_trigger[-n_new:] = b_trigger

                    def write_hand_data(g, buffers):
                        g['valid'].resize(n_curr + n_new, axis=0)
                        g['valid'][-n_new:] = buffers.valid

                        g['palm_pos'].resize(n_curr + n_new, axis=0)
                        g['palm_pos'][-n_new:] = buffers.palm_pos

                        g['palm_ori'].resize(n_curr + n_new, axis=0)
                        g['palm_ori'][-n_new:] = buffers.palm_ori

                        g['wrist_pos'].resize(n_curr + n_new, axis=0)
                        g['wrist_pos'][-n_new:] = buffers.wrist_pos

                        g['elbow_pos'].resize(n_curr + n_new, axis=0)
                        g['elbow_pos'][-n_new:] = buffers.elbow_pos

                        g['fingers'].resize(n_curr + n_new, axis=0)
                        g['fingers'][-n_new:] = buffers.fingers

                    write_hand_data(g_right, b_right)
                    write_hand_data(g_left, b_left)

                    b_lts, b_sys_time, b_task, b_trigger = [], [], [], []
                    b_right.clear()
                    b_left.clear()

                    f.flush()

                last_save_time = time.time()

        # Save trigger onset times
        if trigger_timestamps:
            dset_trig = f.create_dataset('trigger_onset_times', data=trigger_timestamps, dtype='f8')
            dset_trig.attrs['description'] = 'PC time when trigger was received via serial (includes USB latency)'
            dset_trig.attrs['unit'] = 'seconds (perf_counter)'

        # Save Arduino-specific data for post-processing
        if trigger_source == 'arduino' and arduino_monitor is not None:
            # Save raw Arduino timestamps for high-precision analysis
            if arduino_monitor.trigger_times_us:
                dset_ard = f.create_dataset('arduino_trigger_times_us',
                                data=arduino_monitor.trigger_times_us, dtype='i8')
                dset_ard.attrs['description'] = 'Raw Arduino micros() timestamp when trigger was detected'
                dset_ard.attrs['unit'] = 'microseconds'

            # Save sync points for drift correction
            arduino_sync = f.create_group('arduino_sync')
            arduino_sync.attrs['description'] = 'Synchronization points for Arduino-PC time alignment'
            if arduino_monitor.sync_start_arduino_us is not None:
                arduino_sync.attrs['start_arduino_us'] = arduino_monitor.sync_start_arduino_us
                arduino_sync.attrs['start_pc_time'] = arduino_monitor.sync_start_pc_time
            if arduino_monitor.sync_end_arduino_us is not None:
                arduino_sync.attrs['end_arduino_us'] = arduino_monitor.sync_end_arduino_us
                arduino_sync.attrs['end_pc_time'] = arduino_monitor.sync_end_pc_time
                arduino_sync.attrs['drift_us'] = arduino_monitor.stats.get('sync_drift_us', 0)

            # Save corrected trigger times
            try:
                corrected_times = arduino_monitor.get_trigger_times_pc()
                if corrected_times:
                    dset_trig_corr = f.create_dataset('trigger_onset_times_corrected',
                                    data=corrected_times, dtype='f8')
                    dset_trig_corr.attrs['description'] = 'Arduino trigger times converted to PC time with drift correction (no USB latency)'
                    dset_trig_corr.attrs['unit'] = 'seconds (perf_counter)'
                    dset_trig_corr.attrs['note'] = 'Use this for accurate temporal alignment with leap_timestamp_corrected'
            except:
                pass

        # Save Leap Motion sync points for time alignment
        if leap_listener is not None:
            leap_sync = f.create_group('leap_sync')
            leap_sync.attrs['description'] = 'Synchronization points for Leap Motion-PC time alignment'
            # Save raw first/last frame info (for reference)
            if leap_listener._first_leap_time is not None:
                leap_sync.attrs['raw_start_leap_us'] = leap_listener._first_leap_time
                leap_sync.attrs['raw_start_pc_time'] = leap_listener._first_system_time
            if leap_listener._last_leap_time is not None:
                leap_sync.attrs['end_leap_us'] = leap_listener._last_leap_time
                leap_sync.attrs['end_pc_time'] = leap_listener._last_system_time
                # Calculate drift using raw first frame (for diagnostics)
                if leap_listener._first_leap_time is not None:
                    leap_elapsed_us = leap_listener._last_leap_time - leap_listener._first_leap_time
                    pc_elapsed_us = (leap_listener._last_system_time - leap_listener._first_system_time) * 1_000_000
                    drift_us = leap_elapsed_us - pc_elapsed_us
                    leap_sync.attrs['raw_drift_us'] = drift_us
                    print(f"[LEAP] Raw sync drift: {drift_us:.1f} us over {leap_listener._last_system_time - leap_listener._first_system_time:.1f}s")

            # Convert leap_timestamp to PC time and save as leap_timestamp_corrected
            if (leap_listener._first_leap_time is not None and
                leap_listener._last_leap_time is not None and
                dset_lts.shape[0] > 0):
                try:
                    leap_timestamps = dset_lts[:]
                    system_timestamps = dset_sys_time[:]

                    # Find stable sync point (after initial buffer flush period)
                    STABLE_DELAY_SEC = 2.0
                    time_from_start_s = (leap_timestamps - leap_timestamps[0]) / 1_000_000
                    stable_start_idx = np.searchsorted(time_from_start_s, STABLE_DELAY_SEC)

                    # Ensure we have enough data after the stable point
                    if stable_start_idx >= len(leap_timestamps) - 1:
                        # Recording too short, fall back to first frame
                        stable_start_idx = 0
                        print(f"[LEAP] Warning: Recording shorter than {STABLE_DELAY_SEC}s, using first frame for sync")

                    # Use stable point as sync start instead of first frame
                    leap_start_us = leap_timestamps[stable_start_idx]
                    pc_start = system_timestamps[stable_start_idx]
                    leap_end_us = leap_listener._last_leap_time
                    pc_end = leap_listener._last_system_time

                    # Calculate scale factor for drift correction
                    leap_duration_s = (leap_end_us - leap_start_us) / 1_000_000
                    if leap_duration_s > 0:
                        scale = (pc_end - pc_start) / leap_duration_s

                        # Convert all leap timestamps to PC time
                        # Frames before stable point are extrapolated backwards using the same scale
                        leap_timestamps_corrected = pc_start + ((leap_timestamps - leap_start_us) / 1_000_000) * scale

                        dset_leap_corr = f.create_dataset('leap_timestamp_corrected', data=leap_timestamps_corrected, dtype='f8')
                        dset_leap_corr.attrs['description'] = 'Leap timestamp converted to PC time with drift correction (no USB latency)'
                        dset_leap_corr.attrs['unit'] = 'seconds (perf_counter)'
                        dset_leap_corr.attrs['note'] = 'Use this for accurate temporal alignment with trigger_onset_times_corrected'
                        dset_leap_corr.attrs['stable_start_idx'] = stable_start_idx
                        dset_leap_corr.attrs['stable_delay_sec'] = STABLE_DELAY_SEC
                        print(f"[LEAP] Saved leap_timestamp_corrected ({len(leap_timestamps_corrected)} frames, sync from frame {stable_start_idx})")
                except Exception as e:
                    print(f"[LEAP] Failed to save corrected timestamps: {e}")

        # Save metadata
        f.attrs['total_frames_recorded'] = dset_lts.shape[0]
        f.attrs['frames_dropped'] = frame_drop_count
        f.attrs['trigger_pulses'] = trigger_pulse_count
        f.attrs['trigger_source'] = trigger_source
        f.attrs['queue_size'] = QUEUE_SIZE
        f.attrs['save_interval'] = SAVE_INTERVAL
        f.attrs['note'] = 'system_timestamp and trigger_onset_times use perf_counter base'

        print(f"\nRecording statistics:")
        print(f"  Total frames recorded: {dset_lts.shape[0]}")
        print(f"  Frames dropped: {frame_drop_count}")
        print(f"  Trigger pulses detected: {trigger_pulse_count}")
        if frame_drop_count > 0:
            drop_rate = 100.0 * frame_drop_count / (dset_lts.shape[0] + frame_drop_count)
            print(f"  Drop rate: {drop_rate:.2f}%")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Record Leap Motion data with external trigger support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Trigger sources:
  usb-io  : USB-IO device (100μs polling)
  arduino : Arduino Nano with hardware interrupt (μs precision)
  none    : No external trigger (Leap Motion only)

Examples:
  python record_with_trigger.py
  python record_with_trigger.py --trigger arduino --port COM9
  python record_with_trigger.py --trigger none
        """
    )
    parser.add_argument('--trigger', '-t', choices=['usb-io', 'arduino', 'none'],
                        default='usb-io', help='Trigger source (default: usb-io)')
    parser.add_argument('--port', '-p', type=str, default=None,
                        help='COM port for Arduino (auto-detect if not specified)')
    return parser.parse_args()


def main():
    global is_recording, task_status, trigger_source, arduino_monitor, leap_listener

    # Parse arguments
    args = parse_args()
    trigger_source = args.trigger

    # Enable Windows high resolution timer for accurate polling
    enable_high_resolution_timer()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(OUTPUT_DIR, f"{FILENAME_PREFIX}{timestamp_str}.h5")

    print(f"Starting recording to {filename}")
    print(f"Trigger source: {trigger_source}")
    print("Press SPACE to mark task status = 1")
    print("Press 'q' or ESC in the window to stop recording")
    print("Using high-precision timing (perf_counter)\n")

    # Initialize trigger source
    usb_io_monitor = None
    arduino_monitor = None

    if trigger_source == 'usb-io':
        if not USB_IO_AVAILABLE:
            print("[ERROR] USB-IO module not available")
            return

        usb_io_monitor = USBIOMonitor(
            pin_mask=USB_IO_PIN_MASK,
            poll_interval=USB_IO_POLL_INTERVAL,
            edge_callback=on_usb_io_trigger_edge,
            use_high_precision=True
        )

        if not usb_io_monitor.open():
            print("[INIT] Warning: Failed to open USB-IO device. Continuing without trigger...")
            usb_io_monitor = None
        else:
            usb_io_monitor.start()
            print("[INIT] USB-IO monitor started")

    elif trigger_source == 'arduino':
        if not ARDUINO_AVAILABLE:
            print("[ERROR] Arduino module not available. Install pyserial.")
            return

        arduino_monitor = ArduinoTriggerMonitor(
            port=args.port,
            trigger_callback=on_arduino_trigger,
            auto_connect=False
        )

        if not arduino_monitor.connect(args.port):
            print("[INIT] Warning: Failed to connect to Arduino. Continuing without trigger...")
            arduino_monitor = None
        else:
            arduino_monitor.start()
            print("[INIT] Arduino trigger monitor started")

    elif trigger_source == 'none':
        print("[INIT] No external trigger configured")

    # Start Writer Thread
    writer_thread = threading.Thread(target=writer_thread_func, args=(filename,), daemon=False)
    writer_thread.start()
    print("[INIT] Writer thread started")

    # Start Leap Listener
    listener = RecordingListener()
    leap_listener = listener  # Store reference for sync data access
    connection = leap.Connection()
    connection.add_listener(listener)
    print("[INIT] Leap Motion listener created")

    canvas = LiveCanvas()

    try:
        with connection.open():
            connection.set_tracking_mode(leap.TrackingMode.Desktop)  # desktop mode
            print("Leap Motion connection opened. Waiting for data...\n")

            while is_recording:
                frame = latest_frame_container.get()

                img = canvas.render_frame(frame, data_queue.qsize(),
                                         frame_drop_count, trigger_pulse_count)
                cv2.imshow(canvas.name, img)

                key = cv2.waitKey(10)

                if key & 0xFF == ord('q') or key == 27:
                    print("\nStopping recording...")
                    is_recording = False
                    break

                # Check if window was closed
                try:
                    if cv2.getWindowProperty(canvas.name, cv2.WND_PROP_VISIBLE) < 1:
                        print("\nWindow closed. Stopping...")
                        is_recording = False
                        break
                except:
                    pass

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C). Stopping recording...")
        is_recording = False
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        is_recording = False
    finally:
        is_recording = False
        print("Cleaning up...")

        # Stop USB-IO monitor
        if usb_io_monitor:
            usb_io_monitor.stop()
            usb_io_stats = usb_io_monitor.get_stats()
            print(f"\nUSB-IO Statistics:")
            print(f"  Total pulses: {usb_io_stats['total_pulses']}")
            if usb_io_stats['total_pulses'] > 0:
                print(f"  Min pulse width: {usb_io_stats['min_pulse_width']*1000:.3f} ms")
                print(f"  Max pulse width: {usb_io_stats['max_pulse_width']*1000:.3f} ms")
                print(f"  Avg pulse width: {usb_io_stats['avg_pulse_width']*1000:.3f} ms")
            usb_io_monitor.close()

        # Stop Arduino monitor
        if arduino_monitor:
            arduino_monitor.stop()
            arduino_stats = arduino_monitor.get_stats()
            print(f"\nArduino Statistics:")
            print(f"  Total triggers: {arduino_stats['trigger_count']}")
            print(f"  Sync drift: {arduino_stats['sync_drift_us']:.1f} μs")
            arduino_monitor.disconnect()

        connection.remove_listener(listener)
        writer_thread.join()
        cv2.destroyAllWindows()

        # Restore Windows timer resolution
        disable_high_resolution_timer()
        print("Recording finished.")

def on_press(key):
    global task_status
    if key == keyboard.Key.space:
        task_status = 1

def on_release(key):
    global task_status
    if key == keyboard.Key.space:
        task_status = 0

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global is_recording
    print("\n\n[SIGNAL] Ctrl+C detected! Stopping recording...")
    is_recording = False

if __name__ == "__main__":
    # Setup signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    kb_listener.start()

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt caught in main")
        is_recording = False
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        kb_listener.stop()
        kb_listener.join(timeout=1.0)
        print("[EXIT] Program terminated")
