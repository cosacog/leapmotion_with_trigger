# -*- coding: utf-8 -*-
"""
Leap Motion Hand Tracking Recorder & Visualizer
Records to HDF5 while providing real-time visual feedback.
"""

import sys
import time
import queue
import threading
import datetime
import os
import numpy as np
import h5py
import leap
import cv2
from pynput import keyboard

# Configuration
OUTPUT_DIR = 'data'
FILENAME_PREFIX = 'leap_recording_'
SAVE_INTERVAL = 0.5  # Seconds between file flushes/writes
QUEUE_SIZE = 10000   # Buffer size (increased for long recordings)

# Data Schema Definition (Must match record_handtracking.py)
class HandData:
    def __init__(self):
        self.valid = False
        self.palm_pos = [0.0, 0.0, 0.0]
        self.palm_ori = [0.0, 0.0, 0.0, 0.0] # Quaternion
        self.wrist_pos = [0.0, 0.0, 0.0]
        self.elbow_pos = [0.0, 0.0, 0.0]
        # 5 fingers, 4 bones, 2 joints (prev/next), 3 coords
        self.fingers = np.zeros((5, 4, 2, 3), dtype=np.float32) 

class FrameData:
    def __init__(self, leap_timestamp, task_status):
        self.leap_timestamp = leap_timestamp
        self.task_status = task_status
        self.left = HandData()
        self.right = HandData()

class LiveCanvas:
    def __init__(self):
        self.name = "Leap Live View"
        self.screen_size = [600, 800] # Height, Width
        self.hands_colour = (255, 255, 255)
        self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)
        
        self.scale = 1.0 # pixels per mm
        self.offset_x = self.screen_size[1] // 2
        self.offset_y = self.screen_size[0] // 2

    def to_screen_coords(self, pos_3d):
        x = int(pos_3d[0] * self.scale + self.offset_x)
        y = int(pos_3d[2] * self.scale + self.offset_y) # Z maps to Y
        return x, y

    def render_frame(self, frame_data, queue_size=0, drop_count=0):
        # Clear image
        self.output_image[:, :] = 0

        if frame_data is None:
            cv2.putText(self.output_image, "Waiting for data...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return self.output_image

        # Draw Status
        task_status = frame_data.task_status
        timestamp = frame_data.leap_timestamp
        status_text = f"Time: {timestamp} | Task: {'ON' if task_status else 'OFF'}"
        color = (0, 0, 255) if task_status else (0, 255, 0)
        cv2.putText(self.output_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw queue and drop info
        queue_text = f"Queue: {queue_size}/{QUEUE_SIZE} | Drops: {drop_count}"
        queue_color = (0, 255, 255) if queue_size < QUEUE_SIZE * 0.8 else (0, 165, 255)
        cv2.putText(self.output_image, queue_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, queue_color, 1)
        
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
        
        # Draw Arm (Wrist -> Elbow)
        cv2.circle(self.output_image, (sx_wrist, sy_wrist), 4, (0, 255, 255), -1)
        cv2.circle(self.output_image, (sx_elbow, sy_elbow), 4, (0, 255, 255), -1)
        cv2.line(self.output_image, (sx_wrist, sy_wrist), (sx_elbow, sy_elbow), self.hands_colour, 2)
        
        cv2.putText(self.output_image, label, (sx_wrist - 10, sy_wrist - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
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
                    cv2.line(self.output_image, (sx_wrist, sy_wrist), s_start, self.hands_colour, 1)

# Global State
is_recording = True
task_status = 0 # 0: Off, 1: On
frame_drop_count = 0  # Counter for dropped frames

# Thread-safe container for latest frame (for visualization)
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

class RecordingListener(leap.Listener):
    def on_tracking_event(self, event):
        global frame_drop_count
        if not is_recording:
            return

        # Create frame data object
        frame_data = FrameData(event.timestamp, task_status)

        for hand in event.hands:
            h_data = frame_data.left if str(hand.type) == "HandType.Left" else frame_data.right
            h_data.valid = True
            h_data.palm_pos = [hand.palm.position.x, hand.palm.position.y, hand.palm.position.z]
            h_data.palm_ori = [hand.palm.orientation.x, hand.palm.orientation.y, hand.palm.orientation.z, hand.palm.orientation.w]

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

        # 1. Update Visualization (Non-blocking)
        latest_frame_container.set(frame_data)

        # 2. Push to Recording Queue (Fast)
        try:
            data_queue.put_nowait(frame_data)
        except queue.Full:
            frame_drop_count += 1
            if frame_drop_count % 10 == 1:  # Print every 10th drop to avoid spam
                print(f"Warning: Data queue full, dropping frame! (Total drops: {frame_drop_count})")

def writer_thread_func(filename):
    global is_recording, frame_drop_count

    with h5py.File(filename, 'w') as f:
        # Create datasets with optimized chunk sizes for better performance
        chunk_size = 1000  # Write in chunks of 1000 frames
        dset_lts = f.create_dataset('leap_timestamp', (0,), maxshape=(None,), dtype='i8', chunks=(chunk_size,))
        dset_status = f.create_dataset('task_status', (0,), maxshape=(None,), dtype='i1', chunks=(chunk_size,))
        
        def create_hand_group(group_name):
            g = f.create_group(group_name)
            g.create_dataset('valid', (0,), maxshape=(None,), dtype='bool', chunks=(chunk_size,))
            g.create_dataset('palm_pos', (0, 3), maxshape=(None, 3), dtype='f4', chunks=(chunk_size, 3))
            g.create_dataset('palm_ori', (0, 4), maxshape=(None, 4), dtype='f4', chunks=(chunk_size, 4))
            g.create_dataset('wrist_pos', (0, 3), maxshape=(None, 3), dtype='f4', chunks=(chunk_size, 3))
            g.create_dataset('elbow_pos', (0, 3), maxshape=(None, 3), dtype='f4', chunks=(chunk_size, 3))
            g.create_dataset('fingers', (0, 5, 4, 2, 3), maxshape=(None, 5, 4, 2, 3), dtype='f4', chunks=(chunk_size, 5, 4, 2, 3))
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

        b_lts, b_status = [], []
        b_right = HandBuffers()
        b_left = HandBuffers()

        last_save_time = time.time()

        while is_recording or not data_queue.empty():
            try:
                frame = data_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            b_lts.append(frame.leap_timestamp)
            b_status.append(frame.task_status)
            b_right.append(frame.right)
            b_left.append(frame.left)
            
            if time.time() - last_save_time >= SAVE_INTERVAL:
                if len(b_lts) > 0:
                    n_new = len(b_lts)
                    n_curr = dset_lts.shape[0]
                    
                    dset_lts.resize(n_curr + n_new, axis=0)
                    dset_lts[-n_new:] = b_lts
                    
                    dset_status.resize(n_curr + n_new, axis=0)
                    dset_status[-n_new:] = b_status
                    
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
                    
                    b_lts, b_status = [], []
                    b_right.clear()
                    b_left.clear()
                    
                    f.flush()
                
                last_save_time = time.time()

        # Save metadata including frame drop count
        f.attrs['total_frames_recorded'] = dset_lts.shape[0]
        f.attrs['frames_dropped'] = frame_drop_count
        f.attrs['queue_size'] = QUEUE_SIZE
        f.attrs['save_interval'] = SAVE_INTERVAL

        print(f"\nRecording statistics:")
        print(f"  Total frames recorded: {dset_lts.shape[0]}")
        print(f"  Frames dropped: {frame_drop_count}")
        if frame_drop_count > 0:
            drop_rate = 100.0 * frame_drop_count / (dset_lts.shape[0] + frame_drop_count)
            print(f"  Drop rate: {drop_rate:.2f}%")

def main():
    global is_recording, task_status
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(OUTPUT_DIR, f"{FILENAME_PREFIX}{timestamp_str}.h5")
    
    print(f"Starting recording to {filename}")
    print("Press SPACE to mark event (task status = 1)")
    print("Press 'q' or ESC in the window to stop recording")
    
    # 1. Start Writer Thread
    writer_thread = threading.Thread(target=writer_thread_func, args=(filename,))
    writer_thread.start()
    
    # 2. Start Leap Listener
    listener = RecordingListener()
    connection = leap.Connection()
    connection.add_listener(listener)
    
    canvas = LiveCanvas()
    
    # 3. Main Loop (Visualization & Input)
    # Using pynput is optional here if we have an OpenCV window, we can use waitKey for events.
    # However, to maintain consistent behavior even when window is not focused (if desired), 
    # we could keep pynput. But openCV waitKey is better for a 'visualizer' script.
    # Let's use openCV waitKey for Q/ESC and Space.
    
    try:
        with connection.open():
            while is_recording:
                # Get latest frame safely
                frame = latest_frame_container.get()

                # Render with queue status
                img = canvas.render_frame(frame, data_queue.qsize(), frame_drop_count)
                cv2.imshow(canvas.name, img)
                
                # Handle Input
                key = cv2.waitKey(10) # 10ms ~ 100fps
                
                if key & 0xFF == ord('q') or key == 27: # q or ESC
                    print("\nStopping recording...")
                    is_recording = False
                elif key & 0xFF == ord(' '):
                    # Toggle or Push-to-Talk? 
                    # Original script: on_press=1, on_release=0.
                    # cv2.waitKey doesn't handle 'release' well easily without polling.
                    # Maybe stick to pynput for Space?
                    # Actually, let's use pynput for Space to match original behavior accurately.
                    pass
                
                # If window closed
                if cv2.getWindowProperty(canvas.name, cv2.WND_PROP_VISIBLE) < 1:
                     print("\nWindow closed. Stopping...")
                     is_recording = False

    except KeyboardInterrupt:
        print("\nStopping recording...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        is_recording = False
        connection.remove_listener(listener)
        writer_thread.join()
        cv2.destroyAllWindows()
        print("Recording finished.")

# We need pynput for accurate Space bar behavior (press/release)
def on_press(key):
    global task_status
    if key == keyboard.Key.space:
        task_status = 1

def on_release(key):
    global task_status
    if key == keyboard.Key.space:
        task_status = 0

if __name__ == "__main__":
    # Start Keyboard Listener for Task Status
    kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    kb_listener.start()
    
    main()
    
    kb_listener.stop()
