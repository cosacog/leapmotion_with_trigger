# Leap Motion Recording Improvements

## 概要
`record_and_visualize.py`に長時間記録（3-10分）の安定性とフレームドロップ検出機能を追加しました。

## 改善内容

### 1. フレームドロップ抑制
- **キューサイズの拡大**: `QUEUE_SIZE` を 2000 → 10000 に増加
  - 100 fps で約100秒分のバッファを確保
  - 一時的な書き込み遅延に対応可能

- **HDF5チャンクサイズの最適化**: 1000フレーム単位でチャンク化
  - ファイル書き込みパフォーマンスの向上
  - ファイル断片化の軽減

### 2. フレームドロップの記録と可視化
- **リアルタイム表示**: 画面上にキューサイズとドロップ数を表示
  - Queue: 現在のキューサイズ / 最大サイズ
  - Drops: 累積ドロップ数

- **統計情報の保存**: HDF5ファイルのメタデータに以下を保存
  - `total_frames_recorded`: 記録されたフレーム数
  - `frames_dropped`: ドロップされたフレーム数
  - `queue_size`: 使用したキューサイズ
  - `save_interval`: 保存間隔

- **記録終了時の統計表示**:
  ```
  Recording statistics:
    Total frames recorded: 60000
    Frames dropped: 15
    Drop rate: 0.02%
  ```

### 3. 記録後の解析ツール
新しいスクリプト `analyze_recording.py` を作成:

```bash
python analyze_recording.py data/leap_recording_20240101_120000.h5
```

**解析内容**:
- 記録メタデータの表示
- フレームレート統計（平均fps、間隔の標準偏差など）
- タイムスタンプギャップからのフレームドロップ検出
- ドロップ発生時刻と推定喪失フレーム数
- タスクステータスの統計
- 手の検出率

**出力例**:
```
=== Frame Drop Detection ===
Drop threshold: 20.00 ms
Suspected drop events: 5

Drop details:
Index    Time (s)     Interval (ms)   Est. Frames Lost
----------------------------------------------------------
1234     12.340       45.23           3
5678     56.780       32.15           2
...

Total estimated frames lost: 12
Estimated drop rate: 0.02%
```

## ファイルサイズの見積もり

### 計算式
- 1フレーム ≈ 600-700 バイト
- 100 fps × 600秒(10分) = 60,000フレーム
- **10分間で約 40-50 MB**

### 長時間記録の場合
- 30分: 約 120-150 MB
- 1時間: 約 240-300 MB
- **1GBに到達するには約3-4時間必要**

想定される3-10分の記録では、ファイルサイズは問題になりません。

## フレームドロップの検出方法

### 記録中の検出
1. **キューフル警告**: キューが満杯になるとコンソールに警告
   ```
   Warning: Data queue full, dropping frame! (Total drops: 10)
   ```
   - 10フレームごとに表示（スパム防止）

2. **リアルタイムモニタリング**: 画面表示
   - Queue値が80%を超えるとオレンジ色で警告

### 記録後の検出
1. **メタデータから**: `frames_dropped` 属性を確認
   ```python
   import h5py
   with h5py.File('recording.h5', 'r') as f:
       print(f.attrs['frames_dropped'])
   ```

2. **タイムスタンプ解析**: `analyze_recording.py` を使用
   - 期待される間隔（10ms @ 100fps）の2倍以上のギャップを検出
   - ドロップ発生時刻と推定喪失フレーム数を表示

## 使用方法

### 記録開始
```bash
python record_and_visualize.py
```

**操作**:
- `SPACE`: タスクマーカー（押している間 task_status = 1）
- `q` または `ESC`: 記録停止

### 記録解析
```bash
python analyze_recording.py data/leap_recording_YYYYMMDD_HHMMSS.h5
```

## トラブルシューティング

### フレームドロップが多い場合
1. **SAVE_INTERVALを調整**: `record_and_visualize.py:22`
   ```python
   SAVE_INTERVAL = 0.3  # より頻繁に保存（デフォルト: 0.5）
   ```

2. **QUEUE_SIZEをさらに拡大**: `record_and_visualize.py:23`
   ```python
   QUEUE_SIZE = 20000  # さらに大きなバッファ（デフォルト: 10000）
   ```

3. **HDDからSSDに変更**: ファイル書き込み速度が向上

4. **他のプロセスを停止**: CPU使用率を確認

### ファイル断片化について
- **問題ではない**: HDF5の `resize()` 操作による内部断片化は軽微
- 読み込み速度への影響はほぼ無視できるレベル
- 別ファイルに分割されるわけではない

## 技術的詳細

### スレッド構成
1. **Leap Listenerスレッド**: トラッキングイベントを受信
2. **Writer Thread**: キューからデータを取り出しHDF5に書き込み
3. **Main Thread**: 可視化とユーザー入力処理

### データフロー
```
Leap Motion → Listener → Queue (10000) → Writer Thread → HDF5
                    ↓
            Visualization Container → Main Thread → OpenCV Window
```

### メモリ使用量
- キューサイズ 10000 × 700バイト/フレーム ≈ **7 MB**
- 通常のメモリ使用量: 20-30 MB程度

## 既知の制限事項
- タイムスタンプ解析による検出精度は、Leap Motionのタイムスタンプ精度に依存
- 非常に短い（1-2フレーム）のドロップは検出されない可能性がある
- ネットワークドライブへの保存は推奨されない（書き込み遅延の原因）

## 今後の改善案
- 適応的なSAVE_INTERVAL（キューサイズに応じて動的に調整）
- ドロップ発生時の自動アラート音
- リアルタイムグラフ（フレームレート推移）
