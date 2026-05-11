# -*- coding: utf-8 -*-
"""
USB-IO 2.0 TTL パルス出力サンプル
J1-0 ピンから TTL パルスを発生させる。

概要:
  USB-IO 2.0 の J1（出力ポート）からTTLパルスを生成する。
  動作確認・タイミングテスト・他デバイスへのトリガー送信に使用する。

  J1 = 出力ポート（PC → 外部）
  J2 = 入力ポート（外部 → PC）  ← usb_io_monitor.py はこちらを読む

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
配線 A: テスターで電圧確認する場合
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  USB-IO 2.0           テスター
  J1コネクタ
  ┌─────────┐
  │ J1-0    │─────── 赤棒（＋）
  │         │
  │ GND     │─────── 黒棒（－）
  └─────────┘

  HIGH時: 約5V、LOW時: 約0V が表示される

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
配線 B: Arduino でパルスを受ける場合（動作確認用）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  USB-IO 2.0           Arduino Uno / Nano
  J1コネクタ (出力)
  ┌─────────┐          ┌──────────┐
  │ J1-0    │──────────│ D2       │  ← 割り込み対応ピン
  │         │          │          │
  │ GND     │──────────│ GND      │
  └─────────┘          └──────────┘

  注意: USB-IO 2.0 の J1 出力は 5V TTL。
        3.3V 系デバイス（Arduino Due, Raspberry Pi 等）に直結する場合は
        レベルシフターが必要。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
配線 C: USB-IO 2.0 の J1-0 出力を同じ USB-IO 2.0 の J2-0 で折り返す
        （PC 単体でループバック動作確認する場合）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  USB-IO 2.0
  J1コネクタ (出力)    J2コネクタ (入力)
  ┌─────────┐          ┌─────────┐
  │ J1-0    │──────────│ J2-0    │  ← ジャンパ1本
  │         │          │         │
  │ GND     │          │ GND     │  ← 共通GND（接続不要）
  └─────────┘          └─────────┘

  このスクリプトでパルスを出力しながら usb_io_monitor.py で
  同時受信できる（2つの Python スクリプトを別ターミナルで実行）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USB-IO 2.0 コネクタ配置（参考）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  J1（出力, 8bit）          J2（入力, 8bit）
  ┌───────────────┐         ┌───────────────┐
  │ 1:J1-0 (LSB)  │         │ 1:J2-0 (LSB)  │
  │ 2:J1-1        │         │ 2:J2-1        │
  │ 3:J1-2        │         │ 3:J2-2        │
  │ 4:J1-3        │         │ 4:J2-3        │
  │ 5:J1-4        │         │ 5:J2-4        │
  │ 6:J1-5        │         │ 6:J2-5        │
  │ 7:J1-6        │         │ 7:J2-6        │
  │ 8:J1-7 (MSB)  │         │ 8:J2-7 (MSB)  │
  │ 9:GND         │         │ 9:GND         │
  │10:VCC (+5V)   │         │10:VCC (+5V)   │
  └───────────────┘         └───────────────┘

  ピン番号はコネクタ端から数える（製品マニュアル参照）。

Usage:
    python usb_io2_0_TTL_pulse_sample.py
"""

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