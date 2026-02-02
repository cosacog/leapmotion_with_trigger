import pywinusb.hid as hid
import time

class UsbIo20:
    VENDOR_ID = 0x1352
    PRODUCT_ID = 0x0121
    
    def __init__(self):
        self.device = None
        self.report = None
        self.report_size = 0
        
    def open(self):
        devices = hid.HidDeviceFilter(
            vendor_id=self.VENDOR_ID,
            product_id=self.PRODUCT_ID
        ).get_devices()
        
        if not devices:
            raise Exception("USB-IO2.0が見つかりません")
        
        self.device = devices[0]
        self.device.open()
        self.report = self.device.find_output_reports()[0]
        self.report_size = len(self.report.get_raw_data())
        print(f"Report size: {self.report_size}")
        
        # ポート方向を出力に設定
        self.set_port_direction(0xFF, 0xFF)  # 全ピン出力
        
    def close(self):
        if self.device:
            self.device.close()
    
    def set_port_direction(self, dir1, dir2):
        """ポート方向設定 (0=入力, 1=出力)"""
        data = [0x00] * self.report_size
        data[0] = 0x01  # コマンド: ポート方向設定
        data[1] = dir1  # Port1方向
        data[2] = dir2  # Port2方向
        self.report.set_raw_data(data)
        self.report.send()
        time.sleep(0.01)
            
    def set_port(self, port1_value, port2_value=0):
        data = [0x00] * self.report_size
        data[0] = 0x02  # コマンド: ポート出力
        data[1] = port1_value
        data[2] = port2_value
        self.report.set_raw_data(data)
        self.report.send()
        
    def send_trigger(self, duration_ms=10):
        """トリガーパルスを送信"""
        print("High")
        self.set_port(0x01)
        time.sleep(duration_ms / 1000)
        print("Low")
        self.set_port(0x00)

# 使用
io = UsbIo20()
io.open()
print("トリガー送信")
io.send_trigger(duration_ms=500)  # 確認しやすいよう長めに
io.close()

#%%
import pywinusb.hid as hid

# Km2Net製デバイスを検索
devices = hid.HidDeviceFilter(vendor_id=0x1352).get_devices()

for d in devices:
    print(f"Product: {d.product_name}")
    print(f"Vendor ID: {hex(d.vendor_id)}")
    print(f"Product ID: {hex(d.product_id)}")
    print(f"---")
    
    d.open()
    reports = d.find_output_reports()
    for i, r in enumerate(reports):
        print(f"Report {i}: size={len(r.get_raw_data())}")
    d.close()

#%%
import pywinusb.hid as hid
import time

devices = hid.HidDeviceFilter(vendor_id=0x1352).get_devices()
device = devices[0]
device.open()

report = device.find_output_reports()[0]
report_size = len(report.get_raw_data())

def send_data(data_values, label):
    data = [0x00] * report_size
    for i, v in enumerate(data_values):
        data[i] = v
    report.set_raw_data(data)
    report.send()
    print(f"{label}: {data_values}")
    time.sleep(2)

# パターン1: data[1]に直接値
send_data([0x00, 0x01], "Pattern 1 - High")
send_data([0x00, 0x00], "Pattern 1 - Low")

# パターン2: コマンド0x20を使用
send_data([0x00, 0x20, 0x01, 0x00], "Pattern 2 - High")
send_data([0x00, 0x20, 0x00, 0x00], "Pattern 2 - Low")

# パターン3: data[0]にコマンド
send_data([0x20, 0x01, 0x00], "Pattern 3 - High")
send_data([0x20, 0x00, 0x00], "Pattern 3 - Low")

# パターン4: 全ビットHigh
send_data([0x00, 0xFF], "Pattern 4 - All High")
send_data([0x00, 0x00], "Pattern 4 - All Low")

device.close()
print("Done")

#%%
import pywinusb.hid as hid
import time

devices = hid.HidDeviceFilter(vendor_id=0x1352).get_devices()
device = devices[0]
device.open()

report = device.find_output_reports()[0]
report_size = len(report.get_raw_data())

def send_data(data_values, label):
    data = [0x00] * report_size
    for i, v in enumerate(data_values):
        data[i] = v
    report.set_raw_data(data)
    report.send()
    print(f"{label}: {data_values}")
    time.sleep(2)

print("=== J2 (Port2) テスト ===")
# data[2]がPort2(J2)の可能性
send_data([0x00, 0x00, 0x01], "J2-0 High")
send_data([0x00, 0x00, 0x00], "J2-0 Low")

send_data([0x00, 0x00, 0xFF], "J2 All High")
send_data([0x00, 0x00, 0x00], "J2 All Low")

print("=== 両ポート同時 ===")
send_data([0x00, 0xFF, 0xFF], "J1+J2 All High")
send_data([0x00, 0x00, 0x00], "J1+J2 All Low")

device.close()
print("Done")

#%%
import pywinusb.hid as hid
import time

devices = hid.HidDeviceFilter(vendor_id=0x1352).get_devices()
device = devices[0]
device.open()

report = device.find_output_reports()[0]
report_size = len(report.get_raw_data())

def send_data(data_values, label):
    data = [0x00] * report_size
    for i, v in enumerate(data_values):
        data[i] = v
    report.set_raw_data(data)
    report.send()
    print(f"{label}: {data_values}")
    time.sleep(0.1)

# 方向設定コマンドの候補を試す
print("=== 方向設定テスト ===")

# パターンA: コマンド0x01で方向設定、その後0x02で出力
send_data([0x00, 0x01, 0xFF, 0xFF], "Dir: all output")
send_data([0x00, 0x02, 0x00, 0x00], "Out: all Low")
print("Check Arduino... (2sec)")
time.sleep(2)

send_data([0x00, 0x02, 0xFF, 0xFF], "Out: all High")
print("Check Arduino... (2sec)")
time.sleep(2)

# パターンB: コマンド0x10で方向、0x20で出力
send_data([0x00, 0x10, 0xFF, 0xFF], "Dir: all output")
send_data([0x00, 0x20, 0x00, 0x00], "Out: all Low")
print("Check Arduino... (2sec)")
time.sleep(2)

# パターンC: data[0]にコマンド
send_data([0x01, 0xFF, 0xFF], "Dir cmd in [0]")
send_data([0x02, 0x00, 0x00], "Out: all Low")
print("Check Arduino... (2sec)")
time.sleep(2)

device.close()
print("Done")
#%%
import pywinusb.hid as hid
import time

devices = hid.HidDeviceFilter(vendor_id=0x1352).get_devices()
device = devices[0]
device.open()

# Feature Reportを探す
feature_reports = device.find_feature_reports()
output_reports = device.find_output_reports()

print(f"Feature Reports: {len(feature_reports)}")
print(f"Output Reports: {len(output_reports)}")

for i, r in enumerate(feature_reports):
    print(f"Feature Report {i}: size={len(r.get_raw_data())}")

# Feature Reportがあれば使用
if feature_reports:
    report = feature_reports[0]
    report_size = len(report.get_raw_data())
    
    def send_feature(data_values, label):
        data = [0x00] * report_size
        for i, v in enumerate(data_values):
            data[i] = v
        report.set_raw_data(data)
        report.send()
        print(f"{label}")
        time.sleep(2)
    
    print("\n=== Feature Report テスト ===")
    send_feature([0x00, 0x00, 0x00], "All Low")
    send_feature([0x00, 0xFF, 0xFF], "All High")
    send_feature([0x00, 0x00, 0x00], "All Low")

device.close()
print("Done")
#%%
import hid
import time

VENDOR_ID = 0x1352
PRODUCT_ID = 0x0121

# デバイス一覧を表示
print("=== デバイス一覧 ===")
for d in hid.enumerate(VENDOR_ID):
    print(f"Product: {d['product_string']}")
    print(f"Path: {d['path']}")
    print(f"Interface: {d['interface_number']}")
    print("---")

# デバイスを開く
device = hid.device()
device.open(VENDOR_ID, PRODUCT_ID)
device.set_nonblocking(1)

print(f"\nManufacturer: {device.get_manufacturer_string()}")
print(f"Product: {device.get_product_string()}")

def send_and_check(data, label):
    print(f"\n{label}: {[hex(x) for x in data]}")
    device.write(data)
    time.sleep(0.5)
    
    # 応答を読む
    response = device.read(65)
    if response:
        print(f"Response: {[hex(x) for x in response[:8]]}")
    time.sleep(1.5)

# 64バイト + レポートID
# パターン1
data = [0x00] * 65
data[1] = 0x00  # Port1 Low
data[2] = 0x00  # Port2 Low
send_and_check(data, "All Low")

data[1] = 0xFF  # Port1 High
data[2] = 0xFF  # Port2 High
send_and_check(data, "All High")

data[1] = 0x00
data[2] = 0x00
send_and_check(data, "All Low")

device.close()
print("\nDone")
