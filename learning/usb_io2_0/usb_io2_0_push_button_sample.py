#%% USBデバイス一覧表示コード
import hid

# Check all  usb device
for d in hid.enumerate(0, 0):
    keys = d.keys()
    #keys.sort()
    for key in keys:
        print ("%s : %s" % (key, d[key]))
    print ("")
#%% プッシュボタン動作確認コード
import hid
import time

VENDOR_ID = 0x1352
PRODUCT_ID = 0x0121
CMD_READ_SEND = 0x20
CMD_SYSCONF_READ = 0xf8
CMD_SYSCONF_WRITE = 0xf9

print("Opening device")
h = hid.device()
h.open(VENDOR_ID, PRODUCT_ID)
print(f"Product: {h.get_product_string()}")

def send_command(cmd, p1=0, p2=0):
    data = [0] * 64
    data[0] = 0
    data[1] = cmd
    data[2] = 1
    data[3] = p1
    data[4] = 2
    data[5] = p2
    data[63] = 0
    h.write(data)
    return h.read(64)

# システムコンフィグレーションを読み取り
print("現在の設定を読み取り中...")
rcvd = send_command(CMD_SYSCONF_READ)
current_p1_dir = rcvd[1]
current_p2_dir = rcvd[2]
print(f"Port1方向設定: {format(current_p1_dir, '08b')} (0=出力, 1=入力)")
print(f"Port2方向設定: {format(current_p2_dir, '04b')}")

# 0番ピンを入力モードに設定（ビット0を1にする）
new_p1_dir = current_p1_dir | 0x01  # ビット0を1に（入力モード）
print(f"\n0番ピンを入力モードに設定中...")
send_command(CMD_SYSCONF_WRITE, new_p1_dir, current_p2_dir)

# 設定確認
rcvd = send_command(CMD_SYSCONF_READ)
print(f"新Port1方向設定: {format(rcvd[1], '08b')}")

# 入力読み取り関数
def readP():
    rcvd = send_command(CMD_READ_SEND, 0, 0)
    p1 = rcvd[1]
    p2 = rcvd[2]
    send_command(CMD_READ_SEND, p1, p2)
    return rcvd

# 0番ピンの監視
print("\n0番ピンを監視中... (Ctrl+Cで終了)")
print("ボタンを押すと状態が表示されます\n")

prev_button = None

try:
    while True:
        rcvd = readP()
        p1 = rcvd[1]
        
        # 0番ピン（ビット0）の状態を取得
        button_state = (p1 & 0x01)
        
        if button_state != prev_button:
            if button_state == 0:
                print("ボタン: 押下 (GND接続)")
            else:
                print("ボタン: 解放 (プルアップ)")
            prev_button = button_state
        
        time.sleep(0.01)
        
except KeyboardInterrupt:
    print("\n監視を終了しました")

h.close()
#%% プッシュボタン動作確認コード
import hid
import time

VENDOR_ID = 0x1352
PRODUCT_ID = 0x0121
CMD_READ_SEND = 0x20
CMD_SYSCONF_READ = 0xf8
CMD_SYSCONF_WRITE = 0xf9

print("Opening device")
h = hid.device()
h.open(VENDOR_ID, PRODUCT_ID)
print(f"Product: {h.get_product_string()}")

def send_command(cmd, p1=0, p2=0):
    data = [0] * 64
    data[0] = 0
    data[1] = cmd
    data[2] = 1
    data[3] = p1
    data[4] = 2
    data[5] = p2
    data[63] = 0
    h.write(data)
    return h.read(64)

# 0番ピンを入力モードに設定
rcvd = send_command(CMD_SYSCONF_READ)
new_p1_dir = rcvd[1] | 0x01
send_command(CMD_SYSCONF_WRITE, new_p1_dir, rcvd[2])
print(f"Port1方向設定: {format(new_p1_dir, '08b')} (ビット0は入力)\n")

# デバッグ：Port1全体の生データを表示
print("Port1の全ピン状態を監視中... (Ctrl+Cで終了)")
print("ビット0（0番ピン）の変化を確認してください\n")
print("Port1 (8bit) | ビット0")
print("-" * 30)

try:
    count = 0
    while True:
        # 単純に読み取り
        rcvd = send_command(CMD_READ_SEND, 0, 0)
        p1 = rcvd[1]
        bit0 = (p1 & 0x01)
        
        # 1秒ごとまたは値が変化したら表示
        if count % 100 == 0:  # 1秒ごと
            print(f"{format(p1, '08b')}  |  {bit0}", end="")
            print(" <- 今ボタンを押してみてください")
        
        time.sleep(0.01)
        count += 1
        
except KeyboardInterrupt:
    print("\n\n監視を終了しました")

# ボタンを何も接続していない状態で確認
print("\n【確認】ボタンを外した状態で再度実行してください：")
rcvd = send_command(CMD_READ_SEND, 0, 0)
p1 = rcvd[1]
print(f"ボタン無し: Port1 = {format(p1, '08b')}, ビット0 = {p1 & 0x01}")

h.close()