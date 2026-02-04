# タイムスタンプ同期の仕組み

## 概要

Leap Motion、Arduino、PCはそれぞれ独立したクロックを持っています。
本システムでは、記録終了時に**線形補間によるドリフト補正**を行い、
全てのタイムスタンプを同一のPC時間基準（`perf_counter`）に変換します。

## 3つのクロックソース

| ソース | 形式 | 基準点 | 精度 |
|--------|------|--------|------|
| **Leap Motion** | マイクロ秒 (int64) | デバイス起動時 | ~1μs |
| **Arduino** | マイクロ秒 (int64) | Arduino起動時 | ~4μs |
| **PC (perf_counter)** | 秒 (float64) | プロセス起動時 | ~100ns |

### 問題点

```python
# 同期なし（間違った方法）
leap_timestamp = 1500000      # Leap: デバイス起動から1.5秒
arduino_timestamp = 2000000   # Arduino: Arduino起動から2.0秒
pc_time = 123.456             # PC: perf_counter

# これらは直接比較できない！
# 基準点もクロック速度も異なる
```

## 解決策: 線形補間によるドリフト補正

### 原理

記録開始時と終了時にsyncポイントを取得し、線形補間で変換します。

```
記録開始                                          記録終了
    │                                                │
    ▼                                                ▼
sync_start                                      sync_end
(device_us, pc_time)                       (device_us, pc_time)
    │←───────────── 線形補間 ─────────────────────→│
```

### 変換式

```python
# デバイス時刻（μs）をPC時間（秒）に変換
def convert_to_pc_time(device_us):
    # スケールファクター（ドリフト補正）
    device_duration_s = (end_device_us - start_device_us) / 1_000_000
    pc_duration_s = end_pc_time - start_pc_time
    scale = pc_duration_s / device_duration_s

    # 変換
    elapsed_s = (device_us - start_device_us) / 1_000_000
    return start_pc_time + elapsed_s * scale
```

## HDF5ファイル構造

### タイムスタンプデータ

| データセット | 説明 | 単位 | 用途 |
|-------------|------|------|------|
| `leap_timestamp` | Leap内部クロック | μs (int64) | 生データ保存用 |
| `system_timestamp` | PC受信時刻（USB遅延含む） | 秒 (float64) | 参考用 |
| `leap_timestamp_corrected` | **PC時間に変換済み（ドリフト補正済み）** | 秒 (float64) | **解析に使用** |
| `trigger_onset_times` | トリガー受信時刻（USB遅延含む） | 秒 (float64) | 参考用 |
| `trigger_onset_times_corrected` | **PC時間に変換済み（ドリフト補正済み）** | 秒 (float64) | **解析に使用** |
| `arduino_trigger_times_us` | Arduino内部クロック | μs (int64) | 生データ保存用 |

### Syncグループ

```
leap_sync/
  ├── start_leap_us    # 最初のフレームのLeap timestamp
  ├── start_pc_time    # 最初のフレームのPC time
  ├── end_leap_us      # 最後のフレームのLeap timestamp
  ├── end_pc_time      # 最後のフレームのPC time
  └── drift_us         # クロックドリフト（μs）

arduino_sync/
  ├── start_arduino_us # 記録開始時のArduino micros()
  ├── start_pc_time    # 記録開始時のPC time
  ├── end_arduino_us   # 記録終了時のArduino micros()
  ├── end_pc_time      # 記録終了時のPC time
  └── drift_us         # クロックドリフト（μs）
```

## データフロー

```
┌─────────────────────────────────────────────────────────────┐
│ Leap Motion Device                                          │
│  Hardware Timer: event.timestamp (μs)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ RecordingListener    │
          │ • leap_timestamp     │ ← 生データ保存
          │ • system_timestamp   │ ← PC受信時刻
          │ • sync points記録    │
          └──────────┬───────────┘
                     │
                     ▼ (記録終了時)
          ┌──────────────────────┐
          │ 線形補間変換         │
          │ leap_timestamp       │
          │   → corrected        │
          └──────────┬───────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────────┐
│ HDF5: leap_timestamp_corrected (PC time base)              │
└────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────┐
│ Arduino (Hardware Interrupt)                                │
│  micros() timestamp when TTL pulse detected                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ ArduinoTriggerMonitor│
          │ • trigger_times_us   │ ← 生データ保存
          │ • trigger_times_pc   │ ← PC受信時刻
          │ • sync points記録    │
          └──────────┬───────────┘
                     │
                     ▼ (記録終了時)
          ┌──────────────────────┐
          │ 線形補間変換         │
          │ arduino_trigger_us   │
          │   → corrected        │
          └──────────┬───────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────────┐
│ HDF5: trigger_onset_times_corrected (PC time base)         │
└────────────────────────────────────────────────────────────┘
```

## 使用例

### 基本的な使い方

```python
import h5py
import numpy as np

with h5py.File('data/leap_recording_trigger_xxx.h5', 'r') as f:
    # ドリフト補正済みタイムスタンプを使用（推奨）
    leap_time = f['leap_timestamp_corrected'][:]
    trigger_time = f['trigger_onset_times_corrected'][:]

    # 手の位置データ
    right_palm = f['right_hand/palm_pos'][:]

    # 各トリガーに最も近いフレームを検索
    for i, t in enumerate(trigger_time):
        frame_idx = np.argmin(np.abs(leap_time - t))
        print(f"Trigger #{i+1}: {t:.6f}s -> Frame {frame_idx}")
        print(f"  Palm position: {right_palm[frame_idx]}")
```

### トリガー前後のデータ抽出

```python
def extract_around_trigger(leap_time, data, trigger_time, window_ms=100):
    """トリガー前後のデータを抽出"""
    window_s = window_ms / 1000
    results = []

    for t in trigger_time:
        mask = (leap_time >= t - window_s) & (leap_time <= t + window_s)
        indices = np.where(mask)[0]

        # 相対時刻（トリガーを0とする）
        rel_time = (leap_time[indices] - t) * 1000  # ms

        results.append({
            'trigger_time': t,
            'relative_time_ms': rel_time,
            'data': data[indices],
            'frame_indices': indices
        })

    return results

# 使用例
with h5py.File('recording.h5', 'r') as f:
    leap_time = f['leap_timestamp_corrected'][:]
    trigger_time = f['trigger_onset_times_corrected'][:]
    palm_pos = f['right_hand/palm_pos'][:]

    segments = extract_around_trigger(leap_time, palm_pos, trigger_time)

    for i, seg in enumerate(segments):
        print(f"Trigger #{i+1} at {seg['trigger_time']:.3f}s")
        print(f"  Frames: {len(seg['frame_indices'])}")
        print(f"  Time range: {seg['relative_time_ms'][0]:.1f} to {seg['relative_time_ms'][-1]:.1f} ms")
```

### データセット属性の確認

```python
with h5py.File('recording.h5', 'r') as f:
    # データセットの説明を表示
    for name in ['leap_timestamp', 'leap_timestamp_corrected',
                 'trigger_onset_times', 'trigger_onset_times_corrected']:
        if name in f:
            dset = f[name]
            print(f"{name}:")
            print(f"  Description: {dset.attrs.get('description', 'N/A')}")
            print(f"  Unit: {dset.attrs.get('unit', 'N/A')}")
            print(f"  Note: {dset.attrs.get('note', 'N/A')}")
            print()
```

## タイミング精度

### 各コンポーネントの精度

| 要素 | 精度 |
|------|------|
| Leap Motionハードウェアタイマー | ~1μs |
| Arduino micros() | ~4μs |
| PC perf_counter (Windows) | ~100ns |
| USB通信遅延 | 2-8ms（変動あり） |

### 補正後の精度

| 測定項目 | 補正前 | 補正後 |
|---------|--------|--------|
| Leap-Arduino時刻対応 | 2-8ms（USB遅延） | <1ms |
| クロックドリフト | 累積 | 補正済み |
| フレーム間タイミング | 正確 | 正確 |

### USB遅延の影響

```
TTLパルス発生
    │
    ▼ ← Arduino割り込み（μs精度）★ trigger_onset_times_corrected の基準
    │
    │  [USB転送遅延: 2-8ms]
    │
    ▼
PC受信 ← trigger_onset_times に記録（USB遅延を含む）
```

`*_corrected` データを使用することで、USB遅延の影響を排除できます。

## トラブルシューティング

### Q: leap_timestamp_corrected と trigger_onset_times_corrected の差が数ms以上ある

**A**: 正常な範囲です。以下を確認してください：

```python
with h5py.File('recording.h5', 'r') as f:
    # ドリフト量を確認
    leap_drift = f['leap_sync'].attrs.get('drift_us', 0)
    arduino_drift = f['arduino_sync'].attrs.get('drift_us', 0)

    print(f"Leap drift: {leap_drift:.1f} μs")
    print(f"Arduino drift: {arduino_drift:.1f} μs")
```

### Q: トリガー数が実際より多い/少ない

**A**: 以下を確認してください：

1. Arduinoのデバウンス設定（デフォルト: 1ms）
2. レースコンディションの修正が適用されているか

```python
with h5py.File('recording.h5', 'r') as f:
    print(f"Trigger count: {f.attrs['trigger_pulses']}")
    print(f"Trigger source: {f.attrs['trigger_source']}")
```

### Q: 記録開始/終了時のタイムスタンプが不安定

**A**: Leap Motionの初期化/終了処理による影響です。
解析時は最初と最後の数秒を除外することを推奨します。

```python
# 最初と最後の5秒を除外
stable_mask = (leap_time > leap_time[0] + 5) & (leap_time < leap_time[-1] - 5)
stable_data = data[stable_mask]
```

## まとめ

### 解析に使用すべきデータ

| 用途 | 使用するデータ |
|------|---------------|
| フレームのタイミング | `leap_timestamp_corrected` |
| トリガーのタイミング | `trigger_onset_times_corrected` |
| フレーム-トリガー対応 | 両方の `_corrected` データを使用 |

### 参考用データ

| データ | 用途 |
|--------|------|
| `leap_timestamp` | Leap内部クロックの生データ |
| `system_timestamp` | USB遅延を含むPC受信時刻 |
| `trigger_onset_times` | USB遅延を含むトリガー受信時刻 |
| `arduino_trigger_times_us` | Arduino内部クロックの生データ |

### 精度保証

- **`_corrected` データ同士の比較**: <1ms精度
- **クロックドリフト**: 線形補間で補正済み
- **USB遅延**: `_corrected` データでは排除済み
