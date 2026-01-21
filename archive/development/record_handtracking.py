# -*- coding: utf-8 -*-
"""
Leap Motion Hand Tracking Recorder
Records Leap Motion data to HDF5 format with high-precision timing.
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
from leap import datatypes as ldt
from pynput import keyboard

# System timer removed as per request (event.timestamp is sufficient)

# Configuration
OUTPUT_DIR = 'data'
FILENAME_PREFIX = 'leap_recording_'
SAVE_INTERVAL = 0.5  # Seconds between file flushes/writes
QUEUE_SIZE = 2000    # Buffer size

# Data Schema Definition
class HandData:
    def __init__(self):
        self.valid = False
        self.palm_pos = [0.0, 0.0, 0.0]
        self.palm_ori = [0.0, 0.0, 0.0, 0.0] # Quaternion
        self.wrist_pos = [0.0, 0.0, 0.0]
        self.elbow_pos = [0.0, 0.0, 0.0]
        self.fingers = np.zeros((5, 4, 2, 3), dtype=np.float32) # 5 fingers, 4 bones, 2 joints (prev/next), 3 coords

class FrameData:
    def __init__(self, leap_timestamp, task_status):
        self.leap_timestamp = leap_timestamp
        self.task_status = task_status
        self.left = HandData()
        self.right = HandData()

# Global State
data_queue = queue.Queue(maxsize=QUEUE_SIZE)
is_recording = True
task_status = 0 # 0: Off, 1: On

def on_press(key):
    global task_status
    if key == keyboard.Key.space:
        task_status = 1

def on_release(key):
    global task_status
    if key == keyboard.Key.space:
        task_status = 0

class RecordingListener(leap.Listener):
    def on_tracking_event(self, event):
        if not is_recording:
            return

        # Create frame data object
        frame_data = FrameData(event.timestamp, task_status)
        
        for hand in event.hands:
            h_data = frame_data.left if str(hand.type) == "HandType.Left" else frame_data.right
            h_data.valid = True
            h_data.palm_pos = [hand.palm.position.x, hand.palm.position.y, hand.palm.position.z]
            h_data.palm_ori = [hand.palm.orientation.x, hand.palm.orientation.y, hand.palm.orientation.z, hand.palm.orientation.w]
            
            # Arm data (Wrist and Elbow)
            wrist = hand.arm.next_joint
            elbow = hand.arm.prev_joint
            h_data.wrist_pos = [wrist.x, wrist.y, wrist.z]
            h_data.elbow_pos = [elbow.x, elbow.y, elbow.z]
            
            for i, digit in enumerate(hand.digits):
                for j, bone in enumerate(digit.bones):
                    # Record next_joint for each bone
                    # Bone 0: Metacarpal -> next_joint = MCP (Knuckle)
                    # Bone 1: Proximal -> next_joint = PIP
                    # Bone 2: Intermediate -> next_joint = DIP
                    # Bone 3: Distal -> next_joint = Tip
                    prev = bone.prev_joint
                    next = bone.next_joint
                    h_data.fingers[i, j, 0] = [prev.x, prev.y, prev.z]
                    h_data.fingers[i, j, 1] = [next.x, next.y, next.z]
        
        try:
            data_queue.put_nowait(frame_data)
        except queue.Full:
            print("Warning: Data queue full, dropping frame!")

def writer_thread_func(filename):
    global is_recording
    
    # Pre-allocate buffers (list of lists/arrays for chunking)
    buffer_size = int(100 * SAVE_INTERVAL * 2) # Approx buffer size
    
    # Initialize HDF5 file
    with h5py.File(filename, 'w') as f:
        # Create resizable datasets
        dset_lts = f.create_dataset('leap_timestamp', (0,), maxshape=(None,), dtype='i8')
        dset_status = f.create_dataset('task_status', (0,), maxshape=(None,), dtype='i1')
        
        # Helper to create hand groups
        def create_hand_group(group_name):
            g = f.create_group(group_name)
            g.create_dataset('valid', (0,), maxshape=(None,), dtype='bool')
            g.create_dataset('palm_pos', (0, 3), maxshape=(None, 3), dtype='f4')
            g.create_dataset('palm_ori', (0, 4), maxshape=(None, 4), dtype='f4')
            g.create_dataset('wrist_pos', (0, 3), maxshape=(None, 3), dtype='f4')
            g.create_dataset('elbow_pos', (0, 3), maxshape=(None, 3), dtype='f4')
            g.create_dataset('fingers', (0, 5, 4, 2, 3), maxshape=(None, 5, 4, 2, 3), dtype='f4')
            return g

        g_right = create_hand_group('right_hand')
        g_left = create_hand_group('left_hand')
        
        # Buffer containers
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
            
            # Append to local buffers
            b_lts.append(frame.leap_timestamp)
            b_status.append(frame.task_status)
            b_right.append(frame.right)
            b_left.append(frame.left)
            
            # Save if interval passed or buffer large
            if time.time() - last_save_time >= SAVE_INTERVAL:
                if len(b_lts) > 0:
                    n_new = len(b_lts)
                    n_curr = dset_lts.shape[0]
                    
                    # Resize and append global timestamps
                    dset_lts.resize(n_curr + n_new, axis=0)
                    dset_lts[-n_new:] = b_lts
                    
                    dset_status.resize(n_curr + n_new, axis=0)
                    dset_status[-n_new:] = b_status
                    
                    # Helper to write hand data
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
                    
                    # Clear buffers
                    b_lts, b_status = [], []
                    b_right.clear()
                    b_left.clear()
                    
                    f.flush()
                    print(f"\rRecorded {n_curr + n_new} frames...", end="")
                
                last_save_time = time.time()

def main():
    global is_recording
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(OUTPUT_DIR, f"{FILENAME_PREFIX}{timestamp_str}.h5")
    
    print(f"Starting recording to {filename}")
    print("Press SPACE to mark event (task status = 1)")
    print("Press Ctrl+C to stop recording")
    
    # Start Keyboard Listener
    kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    kb_listener.start()
    
    # Start Writer Thread
    writer_thread = threading.Thread(target=writer_thread_func, args=(filename,))
    writer_thread.start()
    
    # Start Leap Listener
    listener = RecordingListener()
    connection = leap.Connection()
    connection.add_listener(listener)
    
    try:
        with connection.open():
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping recording...")
    finally:
        is_recording = False
        connection.remove_listener(listener)
        kb_listener.stop()
        writer_thread.join()
        print("Recording finished.")

if __name__ == "__main__":
    main()
