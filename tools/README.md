# ツール一覧 (tools/)

Leap Motion 記録データの解析・可視化、および USB-IO 2.0 の操作・動作確認に使用するユーティリティスクリプト群。

---

## 記録データ解析

### `analyze_recording.py`

HDF5 記録ファイルを解析し、フレーム落ち・タイミング異常・統計情報を出力する。

```bash
python tools/analyze_recording.py data/leap_recording_YYYYMMDD_HHMMSS.h5
```

**主な機能:**
- 記録メタデータの表示
- フレームレート統計（平均fps・標準偏差）
- タイムスタンプのギャップからフレーム落ち検出
- タスクステータス統計
- 手の検出率分析

```
=== 記録メタデータ ===
Total frames recorded: 60000
Frames dropped (during recording): 15

=== フレームレート解析 ===
Average FPS: 89.95

=== フレーム落ち検出 ===
Suspected drop events: 5
Total estimated frames lost: 12
```

---

### `recalculate_timestamp_correction.py`

`leap_timestamp_corrected` を、バッファフラッシュ期間後の安定した同期点を使って再計算する。

**背景:**
Leap Motion 接続直後はバッファされたフレームが一気に流れ、最初の 1〜2 秒間 `system_timestamp` が圧縮される。最初フレームを同期基準にすると補正タイムスタンプに線形誤差が生じる。このツールは安定期（デフォルト: 記録開始 2 秒後）を基準にすることで問題を修正する。

```bash
# 既存補正の品質を分析（読み取りのみ）
python tools/recalculate_timestamp_correction.py data/recording.h5 --analyze

# 再計算して新ファイル（*_corrected.h5）に保存
python tools/recalculate_timestamp_correction.py data/recording.h5

# 安定期の遅延を指定して再計算
python tools/recalculate_timestamp_correction.py data/recording.h5 --stable-delay 1.5

# 元ファイルを直接上書き
python tools/recalculate_timestamp_correction.py data/recording.h5 --inplace
```

**Python API:**
```python
from recalculate_timestamp_correction import (
    recalculate_timestamp_correction,
    analyze_timestamp_quality
)

stats = analyze_timestamp_quality('data/recording.h5')
stats = recalculate_timestamp_correction('data/recording.h5', stable_delay_sec=2.0)
```

---

### `visualize_recording.py`

記録された手トラッキングデータを OpenCV で再生・可視化する。

```bash
python tools/visualize_recording.py data/leap_recording_YYYYMMDD_HHMMSS.h5
```

**主な機能:**
- 手のスケルトンのアニメーション再生
- タスクステータス・トリガーステータスの表示
- フレーム単位のナビゲーション

**操作:**
- `SPACE`: 一時停止 / 再生
- `q` / `ESC` / `Ctrl+C`: 終了
- 矢印キー: フレーム送り（一時停止中）

---

### `analyze_hdf_data.py`

HDF5 データから指の角度変化を取り出し、トリガー（TMS を想定）前後の動きを評価する。`analyze_finger_angles.py` をインポートして使用する。

---

### `analyze_finger_angles.py`

MCP 関節の屈曲/伸展・内転/外転を評価する。Leap Motion 内部の Palm orientation を基準に屈曲/伸展・内外転の平面を決定する。

---

### `validate_flexion_angles.py`

HDF5 データから人差し指の屈曲/伸展角度を検証し、MCP 関節と指全体の角度統計・時系列グラフ・ヒストグラムを生成する。

```bash
python tools/validate_flexion_angles.py <path_to_h5_file> [left|right]
```

---

### `validate_abduction_angles.py`

HDF5 データから人差し指の外転/内転角度を検証し、屈曲角度との相関分析を含む時系列プロット・散布図を出力する。

```bash
python tools/validate_abduction_angles.py <path_to_h5_file> [left|right]
```

---

### `mep_measure.py`

指定ディレクトリ内の複数 CSV ファイルから MEP（運動誘発電位）データを読み込み、時間窓 20〜60 ms 内のピーク振幅を計算する。スクリプト内のディレクトリパスを変更して使用する。

---

## USB-IO 2.0 ユーティリティ

USB-IO 2.0 のポート構成:

| ポート | 方向 | 用途 |
|---|---|---|
| **J1** | 出力（PC → 外部）| TTL パルス送信・LED 制御 |
| **J2** | 入力（外部 → PC）| TTL パルス受信・スイッチ検出 |

---

### `usb_device_info.py`

USB-IO 2.0 のディスクリプタ情報（ポーリング間隔 bInterval を含む）を表示する。HIDapi / pyusb の両方で実行可能。

```bash
python tools/usb_device_info.py
python tools/usb_device_info.py --vid 0x1352 --pid 0x0121
python tools/usb_device_info.py --hid-only
```

---

### `usb_io_polling_benchmark.py`

Windows 高解像度タイマーの有無で USB-IO のポーリング間隔を測定・比較し、パルス検出の信頼性を診断するベンチマークツール。

```bash
python tools/usb_io_polling_benchmark.py
python tools/usb_io_polling_benchmark.py --duration 10
```

---

### `usb_io2_0_push_button_checker.py`

J2-0 ピンを監視してタクトスイッチ等によるパルス信号の立ち上がり・立ち下がりエッジを検出し、パルス幅（ミリ秒）を測定する。

```bash
python tools/usb_io2_0_push_button_checker.py
```

**配線:**
```
タクトスイッチ      USB-IO 2.0
                   J2コネクタ (入力)
┌────────┐         ┌─────────┐
│ 端子A  │─────────│ J2-0    │
│ 端子B  │─────────│ GND     │
└────────┘         └─────────┘
(J2 は内部 10kΩ プルアップ済み。外部電源・抵抗不要)
```

---

### `usb_io2_0_TTL_pulse_sample.py`

J1-0 ピンから TTL パルス（5V/0V）を生成する汎用サンプル。タイミングテスト・他デバイスへのトリガー送信に使用する。

```bash
python tools/usb_io2_0_TTL_pulse_sample.py
```

**配線 A — テスターで確認:**
```
USB-IO 2.0 J1       テスター
┌─────────┐
│ J1-0    │───── 赤棒（＋）
│ GND     │───── 黒棒（－）
└─────────┘
```

**配線 B — Arduino で受ける:**
```
USB-IO 2.0 J1       Arduino
┌─────────┐         ┌──────────┐
│ J1-0    │─────────│ D2       │  ← 割り込み対応ピン
│ GND     │─────────│ GND      │
└─────────┘         └──────────┘
※ J1 出力は 5V TTL。3.3V 系デバイスには直結不可（レベルシフター必要）
```

**配線 C — J1→J2 ループバック（PC 単体動作確認）:**
```
J1コネクタ (出力)    J2コネクタ (入力)
┌─────────┐         ┌─────────┐
│ J1-0    │─────────│ J2-0    │  ← ジャンパ 1 本
└─────────┘         └─────────┘
```
このスクリプトと `usb_io_monitor.py` を別ターミナルで同時実行すると、送信と受信を 1 台の PC で確認できる。

---

### `output_ttl_pulse_by_usb_io2.py`

J1 出力ポートから TTL パルスを生成する。`usb_io2_0_TTL_pulse_sample.py` の発展版で、複数の配線パターン例と実装例を含む。

---

### `usb_io2_0_led_on_off_sample.py`

J1 ポートに接続した LED を制御し、個別点灯/消灯・点滅テストを行う。実行すると J1-0 ピンで 10 回の点滅テストを自動実行する。

```bash
python tools/usb_io2_0_led_on_off_sample.py
```

**配線:**
```
USB-IO 2.0 J1       LED
┌─────────┐         アノード（＋）──[330Ω]──┐
│ J1-0    │────────────────────────────────┘
│ GND     │──── カソード（－）
└─────────┘
```

---

## 動作要件

```
numpy
h5py
opencv-python   # visualize_recording.py
matplotlib      # validate_*.py, analyze_*.py
hidapi          # USB-IO 2.0 スクリプト全般
pyusb           # usb_device_info.py（オプション）
```

## 関連ドキュメント

フレーム落ち防止・検出の技術詳細は `docs/RECORDING_IMPROVEMENTS.md` を参照。
