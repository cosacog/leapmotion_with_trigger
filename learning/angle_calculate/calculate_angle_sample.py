import leap
import numpy as np

def calculate_flexion_angle(bone1, bone2, palm_normal):
    """
    屈曲・伸展角度を計算（符号付き）
    
    戻り値:
      正の値 = 屈曲（曲げ）
      負の値 = 伸展（反り）
      0° = 直線
    """
    def to_vector(bone):
        return np.array([
            bone.next_joint.x - bone.prev_joint.x,
            bone.next_joint.y - bone.prev_joint.y,
            bone.next_joint.z - bone.prev_joint.z
        ])
    
    v1 = to_vector(bone1)
    v2 = to_vector(bone2)
    
    # 正規化
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # 角度の大きさ（内積）
    dot = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
    angle_rad = np.arccos(dot)
    
    # 符号の決定（外積）
    cross = np.cross(v1_norm, v2_norm)
    
    palm_n = np.array([palm_normal.x, palm_normal.y, palm_normal.z])
    
    # 外積と手のひら法線の内積で方向判定
    if np.dot(cross, palm_n) < 0:
        angle_rad = -angle_rad  # 伸展方向
    
    return np.degrees(angle_rad)


class TrackingListener(leap.Listener):
    def on_tracking_event(self, event):
        for hand in event.hands:
            thumb = hand.digits[0]
            
            metacarpal = thumb.bones[0]
            proximal = thumb.bones[1]
            
            angle = calculate_flexion_angle(
                metacarpal, 
                proximal, 
                hand.palm.normal
            )
            
            # 屈曲/伸展の表示
            if angle > 0:
                state = "屈曲"
            elif angle < 0:
                state = "伸展"
            else:
                state = "中立"
            
            print(f"親指 MCP: {angle:+.1f}° ({state})")
```

## 出力例
# ```
# 親指 MCP: +25.3° (屈曲)
# 親指 MCP: -5.2° (伸展)
# 親指 MCP: +0.8° (中立)