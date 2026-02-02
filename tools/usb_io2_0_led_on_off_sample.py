import hid
import time
'''
VCC -> 330 ohm 抵抗 -> LED -> J1-0
'''

VENDOR_ID = 0x1352
PRODUCT_ID = 0x0121

# コマンド定義
CMD_READ_SEND = 0x20

class USBIO20:
    def __init__(self):
        self.device = hid.device()
        
    def open(self):
        """デバイスをオープン"""
        self.device.open(VENDOR_ID, PRODUCT_ID)
        print(f"接続: {self.device.get_product_string()}")
        
    def close(self):
        """デバイスをクローズ"""
        self.device.close()
        
    def send_receive(self, p1, p2):
        """
        データ送受信
        p1: Port1の値（8ビット）
        p2: Port2の値（4ビット）
        """
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
    
    def read_ports(self):
        """現在のポート状態を読み取り"""
        rcvd = self.send_receive(0, 0)
        p1 = rcvd[1]
        p2 = rcvd[2]
        self.send_receive(p1, p2)  # ダミー書き込み
        return (p1, p2)
    
    def led_on(self, pin):
        """
        指定ピンのLEDを点灯
        pin: 0-7 (J1-0〜J1-7)
        """
        rcvd = self.send_receive(0, 0)
        p1 = rcvd[1] & ~(1 << pin)  # 該当ビットを0に（点灯）
        p2 = rcvd[2]
        self.send_receive(p1, p2)
        
    def led_off(self, pin):
        """
        指定ピンのLEDを消灯
        pin: 0-7 (J1-0〜J1-7)
        """
        rcvd = self.send_receive(0, 0)
        p1 = rcvd[1] | (1 << pin)  # 該当ビットを1に（消灯）
        p2 = rcvd[2]
        self.send_receive(p1, p2)
    
    def all_on(self):
        """全LED点灯"""
        self.send_receive(0x00, 0x00)
        
    def all_off(self):
        """全LED消灯"""
        self.send_receive(0xFF, 0x0F)


# ===== 使用例 =====

if __name__ == "__main__":
    io = USBIO20()
    
    try:
        io.open()
        
        # まず全消灯
        print("全消灯")
        io.all_off()
        time.sleep(1)
        
        # J1-0で点滅（10回）
        print("\nJ1-0で点滅開始")
        for i in range(10):
            io.led_on(0)
            print(f"{i+1}: 点灯")
            time.sleep(0.5)
            
            io.led_off(0)
            print(f"{i+1}: 消灯")
            time.sleep(2.0)
        
        print("\n点滅完了")
        
    finally:
        io.all_off()  # 最後に全消灯
        io.close()
        print("デバイスクローズ")