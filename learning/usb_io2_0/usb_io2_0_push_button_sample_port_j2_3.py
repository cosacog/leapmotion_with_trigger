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

# J2-3ピンの監視
print("\nJ2-3ピンを監視中... (Ctrl+Cで終了)")
print("ボタンを押すと状態が表示されます\n")

prev_button = None

try:
    while True:
        rcvd = readP()
        p2 = rcvd[2]  # ポート2の状態
        
        # J2-3（ビット3）の状態を取得
        # button_state = (p2 & 0x08) >> 3  # または (p2 & 0b1000) >> 3
        button_state = (p2 & 0x01) >> 0  # または (p2 & 0b1000) >> 3
        
        if button_state != prev_button:
            if button_state == 0:
                print("ボタン: 押下 (GND接続)")
            else:
                print("ボタン: 解放 (プルアップ)")
            prev_button = button_state
        
        time.sleep(0.001)
        
except KeyboardInterrupt:
    print("\n監視を終了しました")

h.close()
#%%
# ```

# ## 主な変更点

# 1. `p2 = rcvd[2]` → **ポート2の状態**を取得
# 2. `button_state = (p2 & 0x08) >> 3` → **J2-3のビット3**をチェック
#    - `0x08` = `0b00001000` (ビット3のマスク)
#    - `>> 3` で右シフトして0または1にする
# 3. システムコンフィグ関連のコードは**削除** → J2はデフォルトで入力モード

# ## 配線
# ```
# USB-IO2.0          スイッチ
#   J2-3 ────────┤  ├──── GND

#%% エッジ検出カウントの例
# rising_edge_count = 0
# falling_edge_count = 0

# while True:
#     rcvd = readP()
#     p2 = rcvd[2]
#     button_state = (p2 & 0x01)
    
#     if button_state != prev_button:
#         if button_state == 1:
#             rising_edge_count += 1
#             print(f"立ち上がり検出 (回数: {rising_edge_count})")
#         else:
#             falling_edge_count += 1
#             print(f"立ち下がり検出 (回数: {falling_edge_count})")
#         prev_button = button_state
    
#     time.sleep(0.0001)  # より高速にポーリング