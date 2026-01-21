
import h5py
import cv2
import numpy as np
import time
import sys
import os

class PlaybackCanvas:
    def __init__(self):
        self.name = "Leap Recording Playback"
        self.screen_size = [600, 800] # Height, Width
        self.hands_colour = (255, 255, 255)
        self.font_colour = (0, 255, 44)
        self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)
        
        # Scaling factors (Leap coordinates are in mm, usually ~(-200, 200) for X, etc.)
        # We need to map mm to pixels.
        # Screen center = (Width/2, Height/2) -> (0, 0, 0)
        self.scale = 1.0 # pixels per mm
        self.offset_x = self.screen_size[1] // 2
        self.offset_y = self.screen_size[0] // 2

    def to_screen_coords(self, pos_3d):
        # pos_3d: [x, y, z] in mm
        # Map X (left/right) -> Screen X
        # Map Z (forward/backward) -> Screen Y (Leap: -Z is forward, +Z is backward towards user)
        # Screen: (0,0) is top-left.
        
        # Leap X: +Right, -Left
        # Leap Z: +User, -Screen
        
        x = int(pos_3d[0] * self.scale + self.offset_x)
        y = int(pos_3d[2] * self.scale + self.offset_y) # Z maps to Y
        return x, y

    def render_frame(self, frame_idx, timestamp, task_status, left_hand, right_hand):
        # Clear image
        self.output_image[:, :] = 0
        
        # Draw Status
        status_text = f"Frame: {frame_idx} | Time: {timestamp} | Task: {'ON' if task_status else 'OFF'}"
        color = (0, 0, 255) if task_status else (0, 255, 0)
        cv2.putText(self.output_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw Hands
        self._draw_hand(left_hand, "L")
        self._draw_hand(right_hand, "R")
        
        return self.output_image

    def _draw_hand(self, hand_data, label):
        if not hand_data['valid']:
            return
            
        wrist_pos = hand_data['wrist_pos']
        elbow_pos = hand_data['elbow_pos']
        
        sx_wrist, sy_wrist = self.to_screen_coords(wrist_pos)
        sx_elbow, sy_elbow = self.to_screen_coords(elbow_pos)
        
        # Draw Arm (Wrist -> Elbow)
        cv2.circle(self.output_image, (sx_wrist, sy_wrist), 4, (0, 255, 255), -1) # Wrist joint Yellow
        cv2.circle(self.output_image, (sx_elbow, sy_elbow), 4, (0, 255, 255), -1) # Elbow joint Yellow
        cv2.line(self.output_image, (sx_wrist, sy_wrist), (sx_elbow, sy_elbow), self.hands_colour, 2)
        
        # Draw Label
        cv2.putText(self.output_image, label, (sx_wrist - 10, sy_wrist - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw Fingers
        # fingers shape: (5, 4, 2, 3) -> (Digit, Bone, Joint[Start/End], Coord)
        fingers = hand_data['fingers']
        
        # To draw palm connections (wrist -> knuckles)
        # We need the start position of Bone 0 (Metacarpal) for each finger.
        # Actually, Metacarpal Start IS roughly the wrist position for most fingers, but Leap data tracks it specifically.
        
        knuckles = [] # To store MCP positions (Bone 0 End / Bone 1 Start)
        
        for i in range(5):
            for j in range(4):
                # Bone j
                bone_start = fingers[i, j, 0]
                bone_end = fingers[i, j, 1]
                
                s_start = self.to_screen_coords(bone_start)
                s_end = self.to_screen_coords(bone_end)
                
                # Draw Bone Line
                cv2.line(self.output_image, s_start, s_end, self.hands_colour, 2)
                
                # Draw Joint (at start of bone)
                cv2.circle(self.output_image, s_start, 2, self.hands_colour, -1)
                
                # If it's the last bone (Distal, j=3), draw the tip (end of bone)
                if j == 3:
                     cv2.circle(self.output_image, s_end, 3, self.hands_colour, -1)
                     
                # Collect Knuckles (Proximal Bone Start = Bone 1 Start)
                # Or Metacarpal End = Bone 0 End
                if j == 0:
                    # Metacarpal Start -> End
                    # Connection from Wrist to Metacarpal Start
                    # (Usually Bone 0 Start is close to wrist/in palm)
                    cv2.line(self.output_image, (sx_wrist, sy_wrist), s_start, self.hands_colour, 1)

def main():
    if len(sys.argv) < 2:
        # Try to find the latest file in data/
        data_dir = "data"
        if os.path.exists(data_dir):
            files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".h5")]
            if files:
                filepath = max(files, key=os.path.getctime)
                print(f"No file specified. Using latest: {filepath}")
            else:
                print("Usage: python visualize_recording.py <path_to_h5_file>")
                return
        else:
             print("Usage: python visualize_recording.py <path_to_h5_file>")
             return
    else:
        filepath = sys.argv[1]

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"Opening {filepath}...")
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Check length of data
            n_frames = f['leap_timestamp'].shape[0]
            print(f"Total frames: {n_frames}")
            
            canvas = PlaybackCanvas()
            
            paused = False
            frame_idx = 0
            
            while True:
                if not paused:
                    if frame_idx >= n_frames:
                        frame_idx = 0 # Loop
                        print("Replaying...")
                    
                    # Read frame data
                    timestamp = f['leap_timestamp'][frame_idx]
                    status = f['task_status'][frame_idx]
                    
                    # Helper to extract hand data
                    def get_hand_data(prefix):
                        return {
                            'valid': f[f'{prefix}/valid'][frame_idx],
                            'palm_pos': f[f'{prefix}/palm_pos'][frame_idx],
                            'wrist_pos': f[f'{prefix}/wrist_pos'][frame_idx],
                            'elbow_pos': f[f'{prefix}/elbow_pos'][frame_idx],
                            'fingers': f[f'{prefix}/fingers'][frame_idx]
                        }

                    left_hand = get_hand_data('left_hand')
                    right_hand = get_hand_data('right_hand')
                    
                    img = canvas.render_frame(frame_idx, timestamp, status, left_hand, right_hand)
                    cv2.imshow(canvas.name, img)
                    
                    frame_idx += 1
                
                key = cv2.waitKey(10) # 10ms delay ~ 100fps max
                
                if key & 0xFF == ord('q') or key == 27: # q or ESC
                    break
                elif key & 0xFF == ord(' '):
                    paused = not paused
                    print(f"Paused: {paused}")
                elif key & 0xFF == 83: # Right Arrow
                     frame_idx = min(frame_idx + 10, n_frames - 1)
                elif key & 0xFF == 81: # Left Arrow
                     frame_idx = max(frame_idx - 10, 0)
                     
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
