import hid
import time

VENDOR_ID = 0x1352
PRODUCT_ID = 0x0121
CMD_READ_SEND = 0x20

class USBIO20:
    def __init__(self):
        self.device = hid.device()
        
    def open(self):
        self.device.open(VENDOR_ID, PRODUCT_ID)
        
    def close(self):
        self.device.close()
        
    def send_receive(self, p1, p2):
        data = [0] * 64
        data[0] = 0
        data[1] = CMD_READ_SEND
        data[2] = 1
        data[3] = p1
        data[4] = 2
        data[5] = p2
        data[63] = 0
        self.device.write(data)
        return self.device.read(64)
    
    def pulse_high(self, pin):
        """ピンをHIGH（5Vまたは3.3V）に設定"""
        rcvd = self.send_receive(0, 0)
        p1 = rcvd[1] | (1 << pin)  # 該当ビットを1に
        p2 = rcvd[2]
        self.send_receive(p1, p2)
    
    def pulse_low(self, pin):
        """ピンをLOW（0V）に設定"""
        rcvd = self.send_receive(0, 0)
        p1 = rcvd[1] & ~(1 << pin)  # 該当ビットを0に
        p2 = rcvd[2]
        self.send_receive(p1, p2)
    
    def all_low(self):
        """全ピンをLOWに"""
        self.send_receive(0x00, 0x00)


# TTLパルス生成テスト
io = USBIO20()

try:
    io.open()
    
    print("J1-0でTTLパルスを生成します")
    print("テスターまたはArduinoで確認してください\n")
    
    # # テスト：10秒間HIGHとLOWを交互に
    # print("10秒間、HIGH/LOWを5秒ずつ出力")
    
    # io.pulse_high(0)  # J1-0をHIGH
    # print("HIGH出力中（5秒）- テスターで電圧確認")
    # time.sleep(5)
    
    # io.pulse_low(0)   # J1-0をLOW
    # print("LOW出力中（5秒）- テスターで電圧確認")
    # time.sleep(5)
    
    print("\n高速パルス生成（10回）")
    for i in range(10):
        io.pulse_high(0)
        print(f"{i+1}: HIGH")
        time.sleep(0.5)
        
        io.pulse_low(0)
        print(f"{i+1}: LOW")
        time.sleep(5.0)
    
finally:
    io.pulse_low(0)  # 最後にLOW
    io.close()
    print("完了")