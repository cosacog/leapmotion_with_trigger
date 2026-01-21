# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 08:28:37 2024

@author: USER
"""

import time
import os
import numpy as np
import math
import serial
import serial.tools.list_ports
import pandas as pd
import leap
from leap import datatypes as ldt
import datetime
import matplotlib.pyplot as plt
from pynput import mouse
from scipy.stats import variation
from scipy.signal import find_peaks

current = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

######## 
dir_fld = 'C:\\Users\\USER\\Desktop\\leap_data'

subj = 'S1'



###########################################



# Save directory
dir_name = os.path.join(dir_fld, subj)
os.chdir(dir_name)


###################     Setting
# Record duration (s)
record_t = 4



# Record finger
# Finger    [Thumb:0, Index:1, Middle:2, Ring:3, Pinky:4]
finger1 = 0
finger2 = 1
finger3 = 2
finger4 = 3
finger5 = 4


# # label = 1
# def auto_detect_serial_port():
#     ports = list(serial.tools.list_ports.comports())
#     for p in ports:
#         print(p)
#         if 'USB Serial Port' in p.description:
#             return p.device
#         return None

# ser = serial.Serial('COM3',9600,timeout=10)  #

# ser.close()


#%%         Flex and Extention all fingers

"""
Positioning: 
    Forearm: pronated position
    Wrist: middle position
    Finger: relaxation
"""

# Trackin finger top postion [X, Y, Z] axis
def location_end_of_finger(hand: ldt.Hand, digit_idx: int) -> ldt.Vector:
    digit = hand.digits[digit_idx]
    return digit.distal.next_joint


# Position [X axis: Holizontal, Y axis: Vertical, Z axis: Posterior-anterior]
finger1_pos_x = []
finger1_pos_y = []
finger1_pos_z = []

finger2_pos_x = []
finger2_pos_y = []
finger2_pos_z = []

finger3_pos_x = []
finger3_pos_y = []
finger3_pos_z = []

finger4_pos_x = []
finger4_pos_y = []
finger4_pos_z = []

finger5_pos_x = []
finger5_pos_y = []
finger5_pos_z = []

Hand_type = []
class FingerListener(leap.Listener):
    def on_tracking_event(self, event):
        for hand in event.hands:
            hand_type = "Left" if str(hand.type) == "HandType.Left" else "Right"
            Hand_type.append(hand_type)
            # Tracked fingertip position (unit mm)
            finger1_top = location_end_of_finger(hand, finger1)
            finger2_top = location_end_of_finger(hand, finger2)
            finger3_top = location_end_of_finger(hand, finger3)
            finger4_top = location_end_of_finger(hand, finger4)
            finger5_top = location_end_of_finger(hand, finger5)
            
            # Extract XYZ axes data
            # Thumb
            finger1_pos_x.append(finger1_top[0])
            finger1_pos_y.append(finger1_top[1])
            finger1_pos_z.append(finger1_top[2])
            # Index finger
            finger2_pos_x.append(finger2_top[0])
            finger2_pos_y.append(finger2_top[1])
            finger2_pos_z.append(finger2_top[2])
            # Middle                        
            finger3_pos_x.append(finger3_top[0])
            finger3_pos_y.append(finger3_top[1])
            finger3_pos_z.append(finger3_top[2])
            # Ring                        
            finger4_pos_x.append(finger4_top[0])
            finger4_pos_y.append(finger4_top[1])
            finger4_pos_z.append(finger4_top[2])
            # Pinky                        
            finger5_pos_x.append(finger5_top[0])
            finger5_pos_y.append(finger5_top[1])
            finger5_pos_z.append(finger5_top[2])
            # Display realtime finger1 position
            print(
                f"{hand_type} finger1 ({finger1_top[0]}, {finger1_top[1]}, {finger1_top[2]})."
            )               
            
   
# Main function
def main():
    listener = FingerListener()

    connection = leap.Connection()
    connection.add_listener(listener)

    # 
    with connection.open():
        time.sleep(record_t)
        # connection.remove_listener(listener)
        # connection.disconnect()
        # while True:
        #     time.sleep(1)

            
# Run 
if __name__ == "__main__":
    main()
        

# Make dataframe for finger position data
df = pd.DataFrame({'Finger1_X': finger1_pos_x, 'Finger1_Y': finger1_pos_y, 'Finger1_Z': finger1_pos_z,
                   'Finger2_X': finger2_pos_x, 'Finger2_Y': finger2_pos_y, 'Finger2_Z': finger2_pos_z,
                   'Finger3_X': finger3_pos_x, 'Finger3_Y': finger3_pos_y, 'Finger3_Z': finger3_pos_z,
                   'Finger4_X': finger4_pos_x, 'Finger4_Y': finger4_pos_y, 'Finger4_Z': finger4_pos_z,
                   'Finger5_X': finger5_pos_x, 'Finger5_Y': finger5_pos_y, 'Finger5_Z': finger5_pos_z})

# Plot (Correction: subtraction all mean)
plot_df = df - df.mean()
finger = ['thumb', 'index', 'middle', 'ring', 'pinky']
# Finger1
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(12,4))
plot_df['Finger1_X'].plot(ax=ax[0],color='brown',label='MT')
plot_df['Finger1_Y'].plot(ax=ax[0],color='blue',label='VT')
plot_df['Finger1_Z'].plot(ax=ax[0],color='green',label='AP')
# plot_df['Finger1_Z'].plot(ax=ax[0],color='lightgray',linestyle='dotted',label='posterior-anterior')
ax[0].set_ylim(plot_df.values.min()*1.5, plot_df.values.max()*1.5)
ax[0].set_title(finger[finger1])
ax[0].legend()
# Finger2
plot_df['Finger2_X'].plot(ax=ax[1],color='brown',label='MT')
plot_df['Finger2_Y'].plot(ax=ax[1],color='blue',label='VT')
plot_df['Finger2_Z'].plot(ax=ax[1],color='green',label='AP')
# plot_df['Finger2_Z'].plot(ax=ax[1],color='lightgray',linestyle='dotted',label='posterior-anterior')
ax[1].set_ylim(plot_df.values.min()*1.5, plot_df.values.max()*1.5)
ax[1].set_title(finger[finger2])
# Finger3
plot_df['Finger3_X'].plot(ax=ax[2],color='brown',label='MT')
plot_df['Finger3_Y'].plot(ax=ax[2],color='blue',label='VT')
plot_df['Finger3_Z'].plot(ax=ax[2],color='green',label='AP')
ax[2].set_ylim(plot_df.values.min()*1.5, plot_df.values.max()*1.5)
ax[2].set_title(finger[finger3])
# Finger4
plot_df['Finger4_X'].plot(ax=ax[3],color='brown',label='MT')
plot_df['Finger4_Y'].plot(ax=ax[3],color='blue',label='VT')
plot_df['Finger4_Z'].plot(ax=ax[3],color='green',label='AP')
ax[3].set_ylim(plot_df.values.min()*1.5, plot_df.values.max()*1.5)
ax[3].set_title(finger[finger4])
# Finger5
plot_df['Finger5_X'].plot(ax=ax[4],color='brown',label='MT')
plot_df['Finger5_Y'].plot(ax=ax[4],color='blue',label='VT')
plot_df['Finger5_Z'].plot(ax=ax[4],color='green',label='AP')
ax[4].set_ylim(plot_df.values.min()*1.5, plot_df.values.max()*1.5)
ax[4].set_title(finger[finger5])

plt.suptitle(Hand_type[0] + '   (unit: mm)')


#%%         Analysis finger flexion 1   Select Peak (positive peak)   

# Select finger [Thumb:1, index: 2, middle: 3, ring: 4, pinky: 5]
s_finger = 2

# Select axis [MT: X, VT: Y, AP: Z]
s_axis = 'Z' 



# Setting peak detection index range
extra = 10

# Plot target movement
plt.figure()
target_name = 'Finger' + str(s_finger) + '_' + str(s_axis)
plt.plot(plot_df[target_name], color='black')
# plt.hlines(amp_threshold, 0, len(plot_df), linestyle='dashed', color='gray')
plt.title('First: positive peak, Second: negative peak, \nFinished: Click right mouse')
# Define search mouse events
# Extract Peak
move_max_idx = []
move_max_amp = []
for ss in range(300):
    # Extract movement peak and trough
    # Extract mouse's coodinate
    get_cood = plt.ginput(n=1, mouse_stop=3)[0]   # mouse_stop is 'right click'
    get_idx = round(get_cood[0])  # Extract index
    # amp_threshold = 10
    # max_idx = find_peaks(plot_df[target_name][max_idx_rng], height=amp_threshold)[0][0] + get_idx
    # Extract peak index
    max_idx_rng = np.arange(get_idx-extra, get_idx+extra+1)  # Extract index of ± 'extra' min index range
    max_idx = max_idx_rng[np.argmax(plot_df[target_name][max_idx_rng])]
    max_amp = plot_df[target_name][max_idx]
    plt.plot(max_idx, max_amp, '.', color='red')
      
    move_max_idx.append(max_idx)
    move_max_amp.append(max_amp)

    
    
    
#%%         nalysis finger flexion 2   Select Trough (negative peak)

# Extract trough
move_min_idx = []
move_min_amp = []
for ss in range(len(move_max_idx)):
    # Extract movement peak and trough
    get_cood = plt.ginput(n=1)[0]   # Extract mouse's coodinate
    get_idx = round(get_cood[0])  # Extract index
    # Extract peak index
    min_idx_rng = np.arange(get_idx-extra, get_idx+extra+1)  # Extract index of ± 'extra' min index range
    min_idx = min_idx_rng[np.argmin(plot_df[target_name][min_idx_rng])]
    min_amp = plot_df[target_name][min_idx]
    plt.plot(min_idx, min_amp, '.', color='blue')
     
    move_min_idx.append(min_idx)
    move_min_amp.append(min_amp)
        

# Calsulate movement amplitude
move_amp = np.array(move_max_amp) - np.array(move_min_amp)
# Caluculate movement interval
move_interval = np.diff(move_max_idx)

# Hand type right or left
hand_side = Hand_type[0]
# Number of movement
move_num = int(len(move_amp))
# Calculate mean amplitude
move_amp_mean = np.mean(move_amp)
# Mean interval
move_interval_mean = np.mean(move_interval)
# CV interval  (formula check OK excel data)
move_interval_cv = variation(move_interval)
# CV interval
move_amp_cv = variation(move_amp)

# Counpound data
move_interval = np.append(move_interval, np.nan)
result_df = pd.DataFrame({'move_amp':move_amp, 'move_interval':move_interval, 'move_max_idx':move_max_idx, 'move_max_amp':move_max_amp,
                          'move_min_idx':move_min_idx, 'move_min_amp':move_min_amp})
result_df2 = pd.DataFrame({'values':[hand_side, move_num, move_amp_mean, move_amp_cv, move_interval_cv, move_interval_mean]},
                          index=['hand_side', 'move_num', 'move_amp_mean','move_amp_cv','move_interval_cv', 'move_interval_mean'])

# Save 
plot_df.to_csv('grasp_release_task.csv')
result_df.to_csv('grasp_task_result1.csv')
result_df2.to_csv('grasp_task_result2.csv')




#%%              Pinch task    Online pinch check
"""
Positioning: 
    Forearm: pronated position
    Wrist: middle position
    Finger: Ｒｅｌａｘａｔｉｏｎ
"""

# Target pinch fingers 
# Finger    [Thumb:0, Index:1, Middle:2, Ring:3, Pinky:4]
p_finger1 = 0
p_finger2 = 1


# Trackin finger top postion [X, Y, Z] axis
def location_end_of_finger(hand: ldt.Hand, digit_idx: int) -> ldt.Vector:
    digit = hand.digits[digit_idx]    # Digit 
    return digit.distal.next_joint    # Get joint coordinate: next_joint

# Subtracted position between two fingers
def sub_vectors(v1: ldt.Vector, v2: ldt.Vector) -> list:
    return map(float.__sub__, v1, v2)

# Judge of piching and define distance of two fingers
def fingers_pinching(thumb: ldt.Vector, index: ldt.Vector):
    diff = list(map(abs, sub_vectors(thumb, index)))
    # Define threshold of pnich distance
    threshold = 30
    if diff[0] < threshold and diff[1] < threshold and diff[2] < threshold:
        return True, diff
    else:
        return False, diff


Hand_type2 = []
pinch1 = []
pinch_finger1_pos_x = []
pinch_finger1_pos_y = []
pinch_finger1_pos_z = []
class PinchingListener(leap.Listener):
    def on_tracking_event(self, event):
        # Down sampling (if 1 is no down sample)
        if event.tracking_frame_id % 1 == 0:
            for hand in event.hands:
                hand_type = "Left" if str(hand.type) == "HandType.Left" else "Right"
                Hand_type2.append(hand_type)
                
                # Tracking finger position
                target_fing1 = location_end_of_finger(hand, p_finger1)
                target_fing2 = location_end_of_finger(hand, p_finger2)
                # Jugdge of pinch
                pinching, array = fingers_pinching(target_fing1, target_fing2)
                pinch1.append(pinching)
                # Extract XYZ axes data
                pinch_finger1_pos_x.append(array[0])
                pinch_finger1_pos_y.append(array[1])
                pinch_finger1_pos_z.append(array[2])
                
                pinching_str = "not pinching" if not pinching else "" + str("pinching")
                print(
                    f"{hand_type} hand thumb and index {pinching_str} with position diff ({array[0]}, {array[1]}, {array[2]})."
                )


# Main function
def main():
    listener = PinchingListener()

    connection = leap.Connection()
    connection.add_listener(listener)

    with connection.open():
        time.sleep(record_t)


# Run 
if __name__ == "__main__":
    main()



# Relase bool to [0,1]
Pinch1 = np.array(pinch1)*1
Pinch1 = Pinch1*np.array(pinch_finger1_pos_z).max()*1.5
pinch_df = pd.DataFrame({'pinch1':Pinch1,'pinch1_X': pinch_finger1_pos_x, 'pinch1_Y': pinch_finger1_pos_y,
                         'pinch1_Z': pinch_finger1_pos_z})
# Plot pintch data
plt.figure()
plt.scatter(pinch_df.index.values, pinch_df['pinch1'],color='black',label='pinch')
plt.plot(pinch_df['pinch1_X'],color='brown',label='MT')
plt.plot(pinch_df['pinch1_Y'],color='blue',label='VT')
plt.plot(pinch_df['pinch1_Z'],color='green',label='AP')

# plot_df['Finger1_Z'].plot(ax=ax[0],color='lightgray',linestyle='dotted',label='posterior-anterior')
plt.ylim(pinch_df.values.min()*1.5, pinch_df.values.max()*1.5)
plt.title('pinch'+'_['+str(p_finger1)+','+str(p_finger2)+']')
plt.legend()


# Save 
pinch_df.to_csv('pinch_task.csv')


#%%         Analysis  Pinch


# Select axis [MT: X, VT: Y, AP: Z]
s_axis2 = 'Z' 


# Extract pinch index
pinch_idx = np.where(np.diff(pinch_df['pinch1'])  > 50)[0] + 1
# Calculate pinch interval
pinch_interval = np.diff(pinch_idx)


# Extract pinch amplitude
p_max_amp_idx = []
p_max_amp = []
target_name2 = 'pinch1' + '_' + str(s_axis2)
for ss in np.arange(len(pinch_idx)-1):
    max_amp_idx = np.argmax(pinch_df[target_name2][pinch_idx[ss]:pinch_idx[ss+1]]) + [pinch_idx[ss]]
    max_amp = pinch_df[target_name2][max_amp_idx]
    p_max_amp_idx.append(max_amp_idx[0])
    p_max_amp.append(max_amp.values[0])
    
    
# Plot check data
plt.figure()
plt.plot(pinch_df[target_name2], color='black')
# Plot peak amplitude
plt.plot(p_max_amp_idx, p_max_amp, '.', color='red')
# Plot pinch detection
plt.plot(pinch_idx, np.zeros(len(pinch_idx)), '.', color='blue')


# Hand type right or left
hand_side2 = Hand_type2[0]
# Caluculated mean
pinch_num = len(pinch_idx) - 1      # The number of pinch
pinch_amp_mean = np.mean(p_max_amp)  # Amplitude mean
pinch_amp_cv = variation(p_max_amp)  # Amplitude CV
pinch_interval_mean = pinch_interval.mean()  # Inerval mean
pinch_interval_cv = variation(pinch_interval)  # Inerval CV


# Counpound data
pinch_result_df = pd.DataFrame({'p_max_amp':p_max_amp, 'pinch_interval':pinch_interval, 'p_max_amp_idx':p_max_amp_idx})
pinch_result_df2 = pd.DataFrame({'values':[hand_side2, pinch_num, pinch_amp_mean, pinch_amp_cv, pinch_interval_cv, pinch_interval_mean]},
                          index=['hand_side2', 'pinch_num', 'pinch_amp_mean','pinch_amp_cv','pinch_interval_cv', 'pinch_interval_mean'])

# Save 
pinch_df.to_csv('pinch_task.csv')
pinch_result_df.to_csv('pinch_result_df.csv')
pinch_result_df2.to_csv('pinch_result_df2.csv')



#%%    Forearm: rotation   Wrist: Flex, Radial bending
"""
Positioning: 
    Forearm: pronated position
    Wrist: middle position
    Finger: relaxation
"""

# Leap motion: rotation data is quanternion
# Transfomred quanternion to degree data 
def quaternion_to_angle_and_axis(quaternion):
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]
    
    # # クォータニオンの正規化（絶対値が1出ない場合）
    # norm = np.sqrt(x**2 + y**2 + z**2 )
    # nor_x, nor_y, nor_z = x/norm, y/norm, z/norm
    # nor_w = w / np.sqrt(x**2 + y**2 + z**2 + w**2)
    
    # Transfomred quanternion to radian data
    pitch_rad = math.atan2(2*(w*x + y*z), 1-2*(x**2 + y**2))
    yaw_rad = math.asin(2*(w*y - z*x))
    roll_rad = math.atan2(2*(w*z + x*y), 1-2*(y**2 + z**2))
    
    # Transformed radian to degree
    pitch = math.degrees(pitch_rad)
    yaw = math.degrees(yaw_rad)
    roll = math.degrees(roll_rad)

    return pitch, roll, yaw


Hand_type3 = []
arm_rotation_quaternion = []
class ArmListener(leap.Listener):
    def on_tracking_event(self, event):
        # Down sampling (if 1 is no down sample)
        if event.tracking_frame_id %8 == 0:
            for hand in event.hands:
                hand_type = "Left" if str(hand.type) == "HandType.Left" else "Right"
                Hand_type3.append(hand_type)
                
                # Get palsm orientation
                palm_orientation = hand.palm.orientation      
                
                q = [palm_orientation[0],palm_orientation[1],palm_orientation[2],palm_orientation[3]]
                arm_rotation_quaternion.append(q)          
    
                print(
                    f"{hand_type} finger1 ({palm_orientation[0]}, {palm_orientation[1]}, {palm_orientation[2]}, {palm_orientation[3]})"
                )               

        

# Main function
def main():
    listener = ArmListener()

    connection = leap.Connection()
    connection.add_listener(listener)

    with connection.open():
        time.sleep(record_t)
        


# Run 
if __name__ == "__main__":
    main()


# Transformed quaternion to angle
arm_rotation_roll = []
arm_rotation_pitch = []
arm_rotation_yaw = []
for ii in np.arange(len(arm_rotation_quaternion)):
    pitch, roll, yaw = quaternion_to_angle_and_axis(arm_rotation_quaternion[ii])
    arm_rotation_roll.append(roll)
    arm_rotation_pitch.append(pitch)
    arm_rotation_yaw.append(yaw)


# Romove DC component
arm_rotation_roll = arm_rotation_roll - np.mean(arm_rotation_roll)
arm_rotation_pitch = arm_rotation_pitch - np.mean(arm_rotation_pitch)
arm_rotation_yaw = arm_rotation_yaw - np.mean(arm_rotation_yaw)


# Plot ratation 
plt.figure()
plt.plot(arm_rotation_roll,color='brown',label='roll: rotation')
plt.plot(arm_rotation_yaw,color='blue',label='yaw: radial bending')
plt.plot(arm_rotation_pitch,color='black',label='pitch: Flex')
plt.title('No anatomical angle (deg)')
plt.legend()
 
# # plot_df['Finger1_Z'].plot(ax=ax[0],color='lightgray',linestyle='dotted',label='posterior-anterior')
# plt.ylim(0, np.array(arm_rotation_angle).max()*1.5)


# Save 
rotation_df = pd.DataFrame({'roll': arm_rotation_roll, 'yaw': arm_rotation_yaw, 'pitch': arm_rotation_pitch})
rotation_df.to_csv('rotation_task.csv')



#%%         Analysis Forearm rotaion 1   Select Peak (positive peak)   


# Select axis [roll, pitch, yaw]
s_axis3 = 'roll' 



# Setting peak detection index range
extra = 3

# Plot target movement
plt.figure()
plt.plot(rotation_df[s_axis3], color='black')
# plt.hlines(amp_threshold, 0, len(plot_df), linestyle='dashed', color='gray')
plt.title('First: positive peak, Second: negative peak, \nFinished: Click right mouse')
# Define search mouse events
# Extract Peak
arm_max_idx = []
arm_max_amp = []
for ss in range(300):
    # Extract movement peak and trough
    # Extract mouse's coodinate
    get_cood = plt.ginput(n=1, mouse_stop=3)[0]   # mouse_stop is 'right click'
    get_idx = round(get_cood[0])  # Extract index
    # amp_threshold = 10
    # max_idx = find_peaks(plot_df[target_name][max_idx_rng], height=amp_threshold)[0][0] + get_idx
    # Extract peak index
    max_idx_rng = np.arange(get_idx-extra, get_idx+extra+1)  # Extract index of ± 'extra' min index range
    max_idx = max_idx_rng[np.argmax(rotation_df[s_axis3][max_idx_rng])]
    max_amp = rotation_df[s_axis3][max_idx]
    plt.plot(max_idx, max_amp, '.', color='red')
      
    arm_max_idx.append(max_idx)
    arm_max_amp.append(max_amp)




#%%         nalysis finger flexion 2   Select Trough (negative peak)

# Extract trough
arm_min_idx = []
arm_min_amp = []
for ss in range(len(arm_max_idx)):
    # Extract movement peak and trough
    get_cood = plt.ginput(n=1)[0]   # Extract mouse's coodinate
    get_idx = round(get_cood[0])  # Extract index
    # Extract peak index
    min_idx_rng = np.arange(get_idx-extra, get_idx+extra+1)  # Extract index of ± 'extra' min index range
    min_idx = min_idx_rng[np.argmin(rotation_df[s_axis3][min_idx_rng])]
    min_amp = rotation_df[s_axis3][min_idx]
    plt.plot(min_idx, min_amp, '.', color='blue')
     
    arm_min_idx.append(min_idx)
    arm_min_amp.append(min_amp)
        

# Calsulate movement amplitude
arm_amp = np.array(arm_max_amp) - np.array(arm_min_amp)
# Caluculate movement interval
arm_interval = np.diff(arm_max_idx)

# Hand type right or left
hand_side3 = Hand_type3[0]
# Number of movement
arm_num = int(len(arm_amp))
# Calculate mean amplitude
arm_amp_mean = np.mean(arm_amp)
# Mean interval
arm_interval_mean = np.mean(arm_interval)
# CV interval  (formula check OK excel data)
arm_interval_cv = variation(arm_interval)
# CV interval
arm_amp_cv = variation(arm_amp)

# Counpound data
arm_interval = np.append(arm_interval, np.nan)
arm_result_df = pd.DataFrame({'arm_amp':arm_amp, 'arm_interval':arm_interval, 'arm_max_idx':arm_max_idx, 'arm_max_amp':arm_max_amp,
                          'arm_min_idx':arm_min_idx, 'arm_min_amp':arm_min_amp})
arm_result_df2 = pd.DataFrame({'values':[hand_side3, arm_num, arm_amp_mean, arm_amp_cv, arm_interval_cv, arm_interval_mean]},
                          index=['hand_side3', 'arm_num', 'arm_amp_mean','arm_amp_cv','arm_interval_cv', 'arm_interval_mean'])

# Save 
rotation_df.to_csv('arm_task.csv')
arm_result_df.to_csv('arm_task_result1.csv')
arm_result_df2.to_csv('arm_task_result2.csv')




#%%         Arm rotation  trial version

def quaternion_to_angle_and_axis(quaternion):
    x, y, z, w = quaternion
    # クォータニオンの正規化（絶対値が1出ない場合）
    norm = np.sqrt(x**2 + y**2 + z**2 + w**2)
    if norm ==0:
        return 0, np.array([0,0,0])
    
    q = np.array(quaternion) / norm
    
    # 回転角度の計算
    # wはcos(theta/2)なので、theta = 2*acos(w)
    angle_radians = 2*math.acos(q[3])    
    
    # 回転軸の計算
    # 軸は(x,y,z)だが、正規化が必要
    axis_unnormalized = q[:3]
    norm_axis = np.linalg.norm(axis_unnormalized)
    
    if norm_axis ==0:  # 角度が０または１８０度の場合は軸がない
        axis = np.array([0,0,0])
    else:
        axis = axis_unnormalized / norm_axis
        
    return np.degrees(angle_radians), axis
    
    
arm_rotation_quaternion = []
arm_rotation_angle = []
arm_rotation_axis = []
class ArmListener(leap.Listener):
    def on_tracking_event(self, event):
        for hand in event.hands:
            hand_type = "Left" if str(hand.type) == "HandType.Left" else "Right"
            
            # Get palsm orientation
            palm_orientation = hand.palm.orientation
            
            # # Get pitch, yaw, roll directly
            # pitch = hand.pitch*leap.RAD_TO_DEG
            
            # Extract XYZ axes data
            quaternion = [palm_orientation[0],palm_orientation[1],palm_orientation[2],palm_orientation[3]]
            arm_rotation_quaternion.append(quaternion)

            # Transform quaternion →　angle
            angle, axis = quaternion_to_angle_and_axis(quaternion)

            arm_rotation_angle.append(angle)
            arm_rotation_axis.append(axis)

            print(
                f"{hand_type} finger1 ({palm_orientation[0]}, {palm_orientation[1]}, {palm_orientation[2]}, {palm_orientation[3]})"
            )               
        

# Main function
def main():
    listener = ArmListener()

    connection = leap.Connection()
    connection.add_listener(listener)

    with connection.open():
        time.sleep(record_t)
        

# Run 
if __name__ == "__main__":
    main()


# Plot ratation 
plt.figure()
plt.plot(arm_rotation_angle,color='brown',label='ratation')
 
# plot_df['Finger1_Z'].plot(ax=ax[0],color='lightgray',linestyle='dotted',label='posterior-anterior')
plt.ylim(0, np.array(arm_rotation_angle).max()*1.5)
plt.title('arm rotation (deg)')
plt.legend()


