import h5py
import cv2
import numpy as np
import time
import sys
import os

from analyze_finger_angles import (
    quaternion_to_palm_vectors,
    bone_to_vector,
    calculate_signed_angle
)

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

    def render_frame(self, frame_idx, timestamp, task_status, left_hand, right_hand,
                     left_angles=None, right_angles=None):
        # Clear image
        self.output_image[:, :] = 0

        # Draw Status
        status_text = f"Frame: {frame_idx} | Time: {timestamp} | Task: {'ON' if task_status else 'OFF'}"
        color = (0, 0, 255) if task_status else (0, 255, 0)
        cv2.putText(self.output_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw Hands
        self._draw_hand(left_hand, "L")
        self._draw_hand(right_hand, "R")

        # Draw Angle Info
        self._draw_angle_info(left_angles, right_angles)

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

        # Draw Palm Orientation Vectors
        self._draw_palm_orientation(hand_data)

    def _draw_palm_orientation(self, hand_data):
        """
        palm位置からpalm orientation（法線と方向）のベクトルを描画

        Args:
            hand_data: 手のデータ（palm_pos, palm_ori, fingersを含む）
        """
        if not hand_data['valid']:
            return

        palm_pos = hand_data['palm_pos']
        palm_ori = hand_data['palm_ori']

        # クォータニオンからpalm_normalとpalm_directionを取得
        palm_normal, palm_direction = quaternion_to_palm_vectors(palm_ori)

        # lateral_axis（横軸）も計算（palm_normalとpalm_directionの外積）
        lateral_axis = np.cross(palm_normal, palm_direction)
        lateral_len = np.linalg.norm(lateral_axis)
        if lateral_len > 1e-10:
            lateral_axis = lateral_axis / lateral_len

        # 描画パラメータ
        dir_length = 40  # finger_direction: 40mm (青)
        normal_length = 30  # palm_normal: 30mm (赤)
        lateral_length = 25  # lateral_axis: 25mm (黄)

        # palm_posの画面座標
        sx_palm, sy_palm = self.to_screen_coords(palm_pos)

        # palm_direction（palm_normalと直交する方向）を青で描画
        dir_end_3d = palm_pos + palm_direction * dir_length
        sx_dir, sy_dir = self.to_screen_coords(dir_end_3d)
        cv2.arrowedLine(self.output_image, (sx_palm, sy_palm), (sx_dir, sy_dir),
                        (255, 150, 0), 2, tipLength=0.3)  # 青

        # palm_normal（手のひら法線）を赤で描画
        normal_end_3d = palm_pos + palm_normal * normal_length
        sx_normal, sy_normal = self.to_screen_coords(normal_end_3d)
        cv2.arrowedLine(self.output_image, (sx_palm, sy_palm), (sx_normal, sy_normal),
                        (0, 0, 255), 2, tipLength=0.3)  # 赤

        # lateral_axis（横軸）を黄で描画
        lateral_end_3d = palm_pos + lateral_axis * lateral_length
        sx_lateral, sy_lateral = self.to_screen_coords(lateral_end_3d)
        cv2.arrowedLine(self.output_image, (sx_palm, sy_palm), (sx_lateral, sy_lateral),
                        (0, 255, 255), 2, tipLength=0.3)  # 黄

        # palm位置にマーカー
        cv2.circle(self.output_image, (sx_palm, sy_palm), 5, (255, 0, 255), -1)  # マゼンタ

    def _draw_angle_info(self, left_angles, right_angles):
        """
        角度情報を画面下部に表示

        Args:
            left_angles: {'mcp_flex': float, 'mcp_abd': float, 'overall_flex': float, 'overall_abd': float} or None
            right_angles: same as left_angles
        """
        y_base = self.screen_size[0] - 80  # 画面下部
        line_height = 20

        # 背景を少し暗くして読みやすく
        cv2.rectangle(self.output_image, (0, y_base - 10), (self.screen_size[1], self.screen_size[0]), (30, 30, 30), -1)

        # 左手の角度
        if left_angles:
            cv2.putText(self.output_image, "Left Index:", (10, y_base),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            cv2.putText(self.output_image,
                        f"MCP: {left_angles['mcp_flex']:+6.1f} / {left_angles['mcp_abd']:+5.1f}",
                        (10, y_base + line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(self.output_image,
                        f"Overall: {left_angles['overall_flex']:+6.1f} / {left_angles['overall_abd']:+5.1f}",
                        (10, y_base + line_height * 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        else:
            cv2.putText(self.output_image, "Left Index: N/A", (10, y_base),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # 右手の角度
        x_right = self.screen_size[1] // 2 + 50
        if right_angles:
            cv2.putText(self.output_image, "Right Index:", (x_right, y_base),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            cv2.putText(self.output_image,
                        f"MCP: {right_angles['mcp_flex']:+6.1f} / {right_angles['mcp_abd']:+5.1f}",
                        (x_right, y_base + line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(self.output_image,
                        f"Overall: {right_angles['overall_flex']:+6.1f} / {right_angles['overall_abd']:+5.1f}",
                        (x_right, y_base + line_height * 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        else:
            cv2.putText(self.output_image, "Right Index: N/A", (x_right, y_base),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # 凡例（角度）
        cv2.putText(self.output_image, "(Flex / Abd)", (self.screen_size[1] - 120, y_base - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # 凡例（Palm Orientation）- 画面上部
        legend_y = 55
        cv2.putText(self.output_image, "Axes:", (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.line(self.output_image, (60, legend_y - 5), (85, legend_y - 5), (255, 150, 0), 2)
        cv2.putText(self.output_image, "Dir", (90, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 150, 0), 1)
        cv2.line(self.output_image, (120, legend_y - 5), (145, legend_y - 5), (0, 0, 255), 2)
        cv2.putText(self.output_image, "Norm", (150, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.line(self.output_image, (190, legend_y - 5), (215, legend_y - 5), (0, 255, 255), 2)
        cv2.putText(self.output_image, "Lat", (220, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)


def calculate_finger_angles_corrected(vec1, vec2, palm_ori, finger_direction):
    """
    2つのベクトル間の屈曲角度と外転角度を計算（中指中手骨方向を使用）

    Args:
        vec1: 基準ベクトル（例: Metacarpal方向）
        vec2: 比較ベクトル（例: Proximal方向、または全体方向）
        palm_ori: [x, y, z, w] - palm orientation quaternion
        finger_direction: 中指中手骨の方向ベクトル（正規化済み）

    Returns:
        flexion_angle: 屈曲/伸展角度（正=屈曲、負=伸展）
        abduction_angle: 外転/内転角度（正=外転、負=内転）
    """
    palm_normal, _ = quaternion_to_palm_vectors(palm_ori)

    # 手の座標系を定義（中指中手骨方向を使用）
    lateral_axis = np.cross(finger_direction, palm_normal)
    lateral_axis = lateral_axis / (np.linalg.norm(lateral_axis) + 1e-10)

    # 正規化（ゼロベクトル検出）
    vec1_len = np.linalg.norm(vec1)
    vec2_len = np.linalg.norm(vec2)
    if vec1_len < 1e-10 or vec2_len < 1e-10:
        return np.nan, np.nan
    vec1_norm = vec1 / vec1_len
    vec2_norm = vec2 / vec2_len

    # === 屈曲/伸展角度 ===
    vec1_proj_flex = vec1_norm - np.dot(vec1_norm, lateral_axis) * lateral_axis
    vec2_proj_flex = vec2_norm - np.dot(vec2_norm, lateral_axis) * lateral_axis

    vec1_proj_flex = vec1_proj_flex / (np.linalg.norm(vec1_proj_flex) + 1e-10)
    vec2_proj_flex = vec2_proj_flex / (np.linalg.norm(vec2_proj_flex) + 1e-10)

    flexion_angle = calculate_signed_angle(vec1_proj_flex, vec2_proj_flex, lateral_axis)

    # === 外転/内転角度 ===
    vec1_proj_abd = vec1_norm - np.dot(vec1_norm, palm_normal) * palm_normal
    vec2_proj_abd = vec2_norm - np.dot(vec2_norm, palm_normal) * palm_normal

    vec1_proj_abd = vec1_proj_abd / (np.linalg.norm(vec1_proj_abd) + 1e-10)
    vec2_proj_abd = vec2_proj_abd / (np.linalg.norm(vec2_proj_abd) + 1e-10)

    abduction_angle = calculate_signed_angle(vec1_proj_abd, vec2_proj_abd, palm_normal)

    return flexion_angle, abduction_angle


def calculate_hand_angles(hand_data):
    """
    手のデータから人差し指の角度を計算（中指中手骨方向を基準として使用）

    Args:
        hand_data: get_hand_data() の戻り値

    Returns:
        dict with angle values, or None if invalid
    """
    if not hand_data['valid']:
        return None

    fingers = hand_data['fingers']
    # 人差し指のデータ (digit index = 1)
    index_finger = fingers[1]  # shape: (4, 2, 3)
    palm_ori = hand_data['palm_ori']

    try:
        # 中指（index=2）の中手骨方向を計算
        middle_finger_metacarpal = fingers[2, 0]  # bone 0 = metacarpal
        finger_direction = middle_finger_metacarpal[1] - middle_finger_metacarpal[0]
        finger_direction_len = np.linalg.norm(finger_direction)
        if finger_direction_len > 1e-10:
            finger_direction = finger_direction / finger_direction_len
        else:
            return None  # 無効なデータ

        # MCP角度（Metacarpal vs Proximal）
        metacarpal_vec = bone_to_vector(index_finger[0])
        proximal_vec = bone_to_vector(index_finger[1])
        mcp_flex, mcp_abd = calculate_finger_angles_corrected(
            metacarpal_vec, proximal_vec, palm_ori, finger_direction)

        # Overall角度（Metacarpal vs 指先方向）
        metacarpal_distal_end = index_finger[0][1]
        fingertip = index_finger[3][1]
        overall_vec = fingertip - metacarpal_distal_end
        overall_flex, overall_abd = calculate_finger_angles_corrected(
            metacarpal_vec, overall_vec, palm_ori, finger_direction)

        return {
            'mcp_flex': mcp_flex,
            'mcp_abd': mcp_abd,
            'overall_flex': overall_flex,
            'overall_abd': overall_abd
        }
    except Exception:
        return None


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

            # Helper to extract hand data
            def get_hand_data(prefix, frame_idx):
                return {
                    'valid': f[f'{prefix}/valid'][frame_idx],
                    'palm_pos': f[f'{prefix}/palm_pos'][frame_idx],
                    'palm_ori': f[f'{prefix}/palm_ori'][frame_idx],
                    'wrist_pos': f[f'{prefix}/wrist_pos'][frame_idx],
                    'elbow_pos': f[f'{prefix}/elbow_pos'][frame_idx],
                    'fingers': f[f'{prefix}/fingers'][frame_idx]
                }

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

                    left_hand = get_hand_data('left_hand', frame_idx)
                    right_hand = get_hand_data('right_hand', frame_idx)

                    # 角度計算
                    left_angles = calculate_hand_angles(left_hand)
                    right_angles = calculate_hand_angles(right_hand)

                    img = canvas.render_frame(frame_idx, timestamp, status, left_hand, right_hand,
                                              left_angles, right_angles)
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
