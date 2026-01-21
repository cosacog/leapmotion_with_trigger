# プロジェクト構成説明

このプロジェクトは、Leap Motion手指トラッキングとUSB-IO 2.0トリガーを統合した記録システムです。

## メインファイル

### `record_with_trigger.py` ⭐ **メイン実行ファイル**
- **目的**: Leap Motionの手指トラッキングデータとUSB-IOトリガーを同期して記録
- **機能**:
  - リアルタイム可視化（OpenCV）
  - HDF5形式でデータ保存
  - 高精度タイムスタンプ同期（perf_counter基準）
  - キーボード入力（SPACE）でタスクマーカー記録
  - イベントドリブン＋スレッドベースアーキテクチャ
- **使い方**: `python record_with_trigger.py`
- **終了方法**: 'q'キーまたはESCキー、ウィンドウを閉じる

## コアモジュール

### `usb_io_monitor.py`
- **目的**: USB-IO 2.0デバイスのピン状態を監視
- **機能**:
  - イベントドリブン監視（スレッドベース）
  - エッジ検出（立ち上がり/立ち下がり）
  - パルス幅測定
  - 高精度タイミング（perf_counter）
- **使用**: `record_with_trigger.py`から自動的に使用

### `timestamp_sync.py`
- **目的**: 異なるクロック間のタイムスタンプ同期
- **主要クラス**:
  - `HighPrecisionTimer`: perf_counterベースの高精度タイマー
  - `TimestampConverter`: Leap Motionハードウェアタイマーとシステム時刻の変換
- **精度**: 約100ナノ秒（Windows環境）

## テスト/デバッグ用ファイル

### `test_usb_io_monitor.py`
- **目的**: USB-IOモニターの単体テスト
- **使い方**: `python test_usb_io_monitor.py`
- **機能**: トリガー検出の動作確認

### `test_record_trigger.py`
- **目的**: 簡易版統合テスト（デバッグ用）
- **特徴**: `TimestampConverter`を使用（複雑な同期ロジック含む）
- **注意**: 過去にデッドロック問題があったため、本番では`record_with_trigger.py`を使用

### `record_simple.py`
- **目的**: タイムスタンプ同期なしの最小版（トラブルシューティング用）
- **用途**: 問題の切り分け

## ドキュメント

### `USB_IO_INTEGRATION.md`
- **内容**: イベントドリブンアーキテクチャの詳細説明
- **トピック**:
  - 従来方式（busy loop）との比較
  - パフォーマンス改善（CPU使用率85-90%削減）
  - 使用例
  - トラブルシューティング

### `TIMESTAMP_SYNC.md`
- **内容**: タイムスタンプ同期の詳細説明
- **トピック**:
  - Leap Motionハードウェアタイマーの特性
  - 同期戦略
  - 精度仕様（time.time vs perf_counter）
  - 実装詳細

## 出力データ

### `data/` ディレクトリ
- **形式**: HDF5 (`.h5`)
- **ファイル名**: `leap_recording_trigger_YYYYMMDD_HHMMSS.h5`
- **データセット**:
  - `leap_timestamp`: Leap Motionオリジナルタイムスタンプ（マイクロ秒）
  - `system_timestamp`: 同期されたシステム時刻（秒、perf_counterベース）
  - `task_status`: SPACEキー状態（0/1）
  - `trigger_status`: USB-IOトリガー状態（0/1）
  - `right/left`: 右手/左手のトラッキングデータ
    - `valid`, `palm_pos`, `palm_ori`, `wrist_pos`, `elbow_pos`, `fingers`

## 依存関係

```
record_with_trigger.py
├── usb_io_monitor.py
├── timestamp_sync.py
│   └── HighPrecisionTimer
├── leap (Ultraleap Gemini SDK)
├── cv2 (OpenCV)
├── h5py
└── pynput
```

## 実行順序の推奨

1. **初めての場合**: `test_usb_io_monitor.py` でUSB-IOが正常に動作するか確認
2. **統合テスト**: `record_with_trigger.py` で本番記録
3. **問題が発生した場合**: `record_simple.py` で問題を切り分け

## 重要な技術仕様

- **Leap Motionサンプリングレート**: 90Hz
- **USB-IOポーリング間隔**: 100μs (0.0001秒)
- **タイムスタンプ精度**: 約100ナノ秒 (Windows環境)
- **HDF5保存間隔**: 0.5秒
- **キューサイズ**: 10,000フレーム

## バージョン管理

Git履歴から主要な変更を確認できます：
```bash
git log --oneline
```

最新の動作する実装は `record_with_trigger.py` で、簡略化されたタイムスタンプ同期を使用しています（ロックなし）。
