#%%
import hid
import time

VENDOR_ID = 0x1352
PRODUCT_ID = 0x0121
CMD_READ_SEND = 0x20

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

def readP():
    rcvd = send_command(CMD_READ_SEND, 0, 0)
    p1 = rcvd[1]
    p2 = rcvd[2]
    send_command(CMD_READ_SEND, p1, p2)
    return rcvd

# J2-0ピンの高速パルス監視
print("\nJ2-0ピンを監視中（2-5msパルス対応）... (Ctrl+Cで終了)")
print("パルス検出時に幅を表示します\n")

prev_state = None
pulse_start_time = None
pulse_count = 0

try:
    while True:
        rcvd = readP()
        p2 = rcvd[2]
        current_state = (p2 & 0x01)
        current_time = time.time()
        
        # エッジ検出
        if current_state != prev_state and prev_state is not None:
            if current_state == 0:  # 立ち下がり（パルス開始）
                pulse_start_time = current_time
                print("パルス開始 (High→Low)")
                
            elif current_state == 1 and pulse_start_time is not None:  # 立ち上がり（パルス終了）
                pulse_width = (current_time - pulse_start_time) * 1000  # ms単位
                pulse_count += 1
                print(f"パルス #{pulse_count}: 幅 = {pulse_width:.3f} ms")
                pulse_start_time = None
        
        prev_state = current_state
        
        # sleepを削除または最小化して高速ポーリング
        time.sleep(0.0001)  # 100μs（必要に応じてコメントアウト）
        
except KeyboardInterrupt:
    print(f"\n監視を終了しました（検出パルス数: {pulse_count}）")

h.close()