# プロジェクト構成説明

このプロジェクトは、Leap Motion手指トラッキングと外部トリガー（USB-IO 2.0 または Arduino）を統合した記録システムです。

## ディレクトリ構成

```
leapmotion_with_trigger/
├── src/                          # メインソースコード
│   ├── record_with_trigger.py    # メイン実行ファイル
│   ├── usb_io_monitor.py         # USB-IO 2.0モニター
│   └── arduino_trigger_monitor.py # Arduinoトリガーモニター
├── arduino/                      # Arduinoファームウェア
│   └── ttl_trigger/
│       └── ttl_trigger.ino       # TTLトリガー検出スケッチ
├── tests/                        # テストファイル
│   └── test_usb_io_monitor.py    # USB-IOモニター単体テスト
├── docs/                         # ドキュメント
│   ├── PROJECT_STRUCTURE.md      # このファイル
│   ├── USB_IO_INTEGRATION.md     # USB-IO統合の詳細
│   ├── TIMESTAMP_SYNC.md         # タイムスタンプ同期の説明
│   └── RECORDING_IMPROVEMENTS.md # 記録改善に関するメモ
├── data/                         # 記録データ出力先
├── archive/                      # 開発履歴ファイル
├── run_record_with_trigger.py    # ルートからの実行用ラッパー
├── requirements.txt              # Python依存関係
└── README.md                     # プロジェクト概要
```

## メインファイル

### `src/record_with_trigger.py` - **メイン実行ファイル**
- **目的**: Leap Motionの手指トラッキングデータと外部トリガーを同期して記録
- **機能**:
  - リアルタイム可視化（OpenCV）
  - HDF5形式でデータ保存
  - 高精度タイムスタンプ同期（perf_counter基準）
  - キーボード入力（SPACE）でタスクマーカー記録
  - トリガーソース選択: `--trigger usb-io|arduino|none`
  - ドリフト補正済みタイムスタンプの自動計算（`leap_timestamp_corrected`, `trigger_onset_times_corrected`）
- **使い方**:
  - `python run_record_with_trigger.py` (ルートから)
  - `python record_with_trigger.py --trigger arduino --port COM9` (Arduinoトリガー)
  - `python record_with_trigger.py --trigger none` (トリガーなし)
- **終了方法**: 'q'キー、ESCキー、Ctrl+C、またはウィンドウを閉じる

## コアモジュール

### `src/usb_io_monitor.py`
- **目的**: USB-IO 2.0デバイスのピン状態を監視
- **機能**:
  - スレッドベースのポーリング監視
  - エッジ検出（立ち上がり/立ち下がり）コールバック
  - パルス幅測定・統計
  - ポーリング間隔の実測統計（`get_poll_interval_analysis()`）
  - 高精度タイミング（perf_counter）
- **実効ポーリング間隔**: 約1~3ms（USB HID Full Speedの1msフレーム間隔が律速）
- **使用**: `record_with_trigger.py`から`--trigger usb-io`で使用

### `src/arduino_trigger_monitor.py`
- **目的**: Arduino経由のTTLトリガー検出
- **機能**:
  - ハードウェア割り込みによるμs精度のトリガー検出
  - シリアル通信（115200 baud）によるリアルタイムイベント通知
  - 開始/終了同期プロトコルによるArduino-PC間のクロック同期
  - `micros()`タイムスタンプの線形補間によるドリフト補正（`convert_to_pc_time()`）
  - COMポートの自動検出（CH340, Arduino等のデバイス識別子）
- **制限**: Arduino RAMの制約で最大200トリガーまで保存。`micros()`は約70分でオーバーフロー
- **使用**: `record_with_trigger.py`から`--trigger arduino`で使用

## Arduinoファームウェア

### `arduino/ttl_trigger/ttl_trigger.ino`
- **目的**: TTLパルスをハードウェア割り込みで検出し、シリアルでPCに通知
- **ハードウェア**: Arduino Nano（互換品対応）
- **配線**: Pin 2 (INT0) にTTL信号、GNDにグランド
- **シリアルプロトコル** (115200 baud):
  - PC → Arduino: `S`(同期開始), `E`(同期終了+データ送信), `R`(リセット), `P`(Ping), `C`(カウント)
  - Arduino → PC: `T,<micros>`(トリガー検出), `S,<micros>`(同期応答), `PONG`(Ping応答)
- **デバウンス**: 1ms（`DEBOUNCE_US`で変更可能）

## テスト/デバッグ用ファイル

### `tests/test_usb_io_monitor.py`
- **目的**: USB-IOモニターの単体テスト
- **使い方**: `cd tests && python test_usb_io_monitor.py`
- **機能**: トリガー検出の動作確認

## 出力データ

### `data/` ディレクトリ
- **形式**: HDF5 (`.h5`)
- **ファイル名**: `leap_recording_trigger_YYYYMMDD_HHMMSS.h5`
- **共通データセット**:
  - `leap_timestamp`: Leap Motionオリジナルタイムスタンプ（マイクロ秒）
  - `system_timestamp`: フレーム受信時のPC時刻（秒、perf_counterベース、USB遅延含む）
  - `leap_timestamp_corrected`: ドリフト補正済みLeapタイムスタンプ（秒、perf_counterベース）
  - `task_status`: SPACEキー状態（0/1）
  - `trigger_status`: トリガー状態（0/1）
  - `trigger_onset_times`: トリガーパルス開始時刻（秒、perf_counterベース）
  - `right_hand/`, `left_hand/`: 右手/左手のトラッキングデータ
    - `valid`, `palm_pos`, `palm_ori`, `wrist_pos`, `elbow_pos`, `fingers`
  - `leap_sync/`: Leap Motion-PC間のクロック同期ポイント
- **Arduinoトリガー使用時の追加データセット**:
  - `arduino_trigger_times_us`: Arduinoの`micros()`による生タイムスタンプ（マイクロ秒）
  - `trigger_onset_times_corrected`: ドリフト補正済みArduinoトリガー時刻（秒、perf_counterベース）
  - `arduino_sync/`: Arduino-PC間のクロック同期ポイント

## 依存関係

```
record_with_trigger.py
├── usb_io_monitor.py       (--trigger usb-io 時)
├── arduino_trigger_monitor.py (--trigger arduino 時)
├── leap (Ultraleap Gemini SDK)
├── cv2 (OpenCV)
├── h5py
├── pynput
└── serial (pyserial, Arduino時のみ)
```

## 実行順序の推奨

1. **USB-IOを使う場合**: `tests/test_usb_io_monitor.py` でUSB-IOが正常に動作するか確認
2. **Arduinoを使う場合**: Arduino IDEで`arduino/ttl_trigger/ttl_trigger.ino`を書き込み後、`python src/arduino_trigger_monitor.py [COMポート]`で単体テスト
3. **本番記録**: `python run_record_with_trigger.py` または `python src/record_with_trigger.py --trigger <ソース>`

## 重要な技術仕様

- **Leap Motionサンプリングレート**: デバイスネイティブ（約100 Hz）
- **USB-IOポーリング間隔**: 約1~3ms（USB HID Full Speedが律速）
- **Arduinoトリガー精度**: μs（ハードウェア割り込み + `micros()`）
- **タイムスタンプ精度**: 約100ナノ秒 (Windows環境、`perf_counter`)
- **HDF5保存間隔**: 0.5秒
- **キューサイズ**: 10,000フレーム

## バージョン管理

Git履歴から主要な変更を確認できます：
```bash
git log --oneline
```

最新の動作する実装は `src/record_with_trigger.py` です。
