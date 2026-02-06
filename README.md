# Leap Motion Hand Tracking with External Trigger Integration

外部トリガーデバイス（USB-IO 2.0 または Arduino）と統合したLeap Motionハンドトラッキングシステムで、同期データ記録を実現します。

## プロジェクト概要

このプロジェクトは、Leap Motionハンドトラッキングと外部トリガーデバイスを組み合わせ、高精度なタイムスタンプ同期を実現します。トリガーソースとしてUSB-IO 2.0（ポーリング方式）またはArduino Nano（ハードウェア割り込み方式）を選択でき、外部トリガー信号と共にハンドトラッキングデータを記録し、実験セットアップにおける同期データ収集を可能にします。

## 主な機能

- **高精度タイムスタンプ同期** (`time.perf_counter()`による約100nsの分解能)
- **複数のトリガーソース対応**:
  - **USB-IO 2.0**: USB HIDポーリングによるエッジ検出（約1~3ms精度）
  - **Arduino Nano**: ハードウェア割り込み（`micros()`によるμs精度）
  - **なし**: トリガーなしでLeap Motionのみ記録
- **スレッドセーフな実装** (最大4つの並行スレッド):
  - Leap Motionリスナースレッド
  - トリガーモニタースレッド（USB-IOまたはArduinoシリアルリーダー）
  - HDF5ライタースレッド
  - メイン可視化スレッド
- **OpenCVによるリアルタイム可視化**
- **HDF5データ形式** (チャンキングによる効率的なストレージ)
- **フレームドロップ防止** (キューベースのバッファリング)

## プロジェクト構造

詳細なファイル説明とディレクトリ構成については、[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)を参照してください。

## インストール

### 前提条件

1. **Ultraleap SDK** (Hyperion v 6以上(と思います))
   - [Ultraleap Developer Site](https://www.ultraleap.com/downloads/leap-motion-controller-2/)からダウンロード
   - OSの選択 -> Ultraleap Hyperion (作成時 v 6.2.0)
   - デフォルトの場所にインストールするか、`LEAPSDK_INSTALL_LOCATION`環境変数を設定

2. **USB-IO 2.0デバイス** (オプション、トリガーソースの一つ)
   - Windowsではドライバー不要 (と思います)

3. **Arduino Nano** (オプション、トリガーソースの一つ)
   - Arduino Nano互換品で動作確認済み（CH340チップ搭載品など）
   - CH340ドライバーが必要な場合あり
   - `arduino/ttl_trigger/ttl_trigger.ino` スケッチを書き込み
   - 配線: Pin 2 (D2) にTTL信号、GNDにグランド接続
   - pyserialが必要（`pip install pyserial`）

### セットアップ：ai作成のドキュメントで間違いがあるかも知れません

```bash
# リポジトリのクローン
git clone <repository-url>
cd leapmotion_handtracking

# 仮想環境の作成と有効化
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 依存関係のインストール
pip install -r requirements.txt

# Leap Python APIのインストール
pip install -e leapc-python-api
```

## 使い方

### メイン記録アプリケーション

プロジェクトルートディレクトリから実行:

```bash
python run_record_with_trigger.py
```

もしくは srcディレクトリに移動(cd src)して以下実行：

```bash
python record_with_trigger.py
```

もしarduinoを利用する時は
```bash
python record_with_trigger.py --trigger arduino
```

**操作方法:**
- **SPACEキー**: タスクステータス = 1をマーク (手動イベントマーキング用)
- **'q'またはESCまたはCtrl+C**: 記録を停止
- **ウィンドウを閉じる**: 記録を停止

**出力:**
- データは`data/leap_recording_trigger_YYYYMMDD_HHMMSS.h5`に保存
- HDF5ファイルの内容:
  - `leap_timestamp`: オリジナルのLeapタイムスタンプ (マイクロ秒)
  - `system_timestamp`: 同期されたシステム時刻 (秒、perf_counterベース)
  - `leap_timestamp_corrected`: Leapタイムスタンプをドリフト補正してPC時刻に変換 (秒、perf_counterベース)
  - `task_status`: SPACEキーの状態 (0/1)
  - `trigger_status`: トリガーの状態 (0/1)
  - `trigger_onset_times`: トリガーパルス開始時刻 (秒、perf_counterベース)
  - `right_hand/`, `left_hand/`: ハンドトラッキングデータ (手のひら、手首、肘、指)
  - `leap_sync/`: Leap Motionクロック同期ポイント
  - Arduinoトリガー使用時の追加フィールド:
    - `arduino_trigger_times_us`: Arduinoの`micros()`による生タイムスタンプ (マイクロ秒)
    - `trigger_onset_times_corrected`: Arduinoタイムスタンプをドリフト補正してPC時刻に変換 (秒、perf_counterベース)
    - `arduino_sync/`: Arduino-PC間のクロック同期ポイント

### USB-IOモニターのテスト

```bash
cd tests
python test_usb_io_monitor.py
```

USB-IOデバイスの接続とエッジ検出をテストします。

## 技術仕様

- **Leap Motionサンプリングレート**: デバイスネイティブ（約100 Hz）
- **USB-IOポーリング間隔**: 約1~3 ms（USB HID Full Speedの1msフレーム間隔が律速。コード上のsleep設定は100μsだが実効値はUSB通信時間に支配される）
- **タイムスタンプ精度**: 約100ナノ秒 (Windows): 下記タイムスタンプ同期を参照ください。
- **HDF5保存間隔**: 0.5秒
- **フレームキューサイズ**: 10,000フレーム
- **USB-IOピン**: J2-0 とGNDをBNCに接続することを想定しています (`src/record_with_trigger.py`で設定可能)
- **arduino**: arduino nanoの互換品で動作を見ました。D2とGNDにBNCで接続することを想定しています。

## アーキテクチャ

### タイムスタンプ同期

すべてのタイムスタンプは`time.perf_counter()`を共通ベース(システム時刻)として使用:
- **Leap Motion**: ハードウェアタイマーをシステム時刻に変換
- **USB-IO**: 直接perf_counterタイムスタンプを使用. 注意としてUSB-IOへの通信で+- 3 ms程度ジッターがあるようです。USB2.0 (full speed)の制約で約1 ms間隔でサンプリングする限界とUSB通信が割り込みされるなどの状況が関係しているようです。
- **arduino**: arduinoのタイムスタンプをシステム時刻に変換
- **精度**: Windowsで約100ns
- **注意点**: PCとleap motion, arduinoは5分で100 ms単位のドリフトがあるようです。hdf5ファイルのleap_timestamp_corrected, arduino

### スレッドモデル

1. **Leap Listenerスレッド**: ハンドトラッキングイベントをキャプチャ
2. **トリガーモニタースレッド** (トリガーソースに応じて1つ):
   - USB-IO: USB HIDラウンドトリップ（約1~3ms）間隔でポーリング
   - Arduino: シリアルポートからのデータ受信待ち（115200 baud）
3. **Writerスレッド**: 0.5秒ごとにバッファされたデータをHDF5に保存
4. **Mainスレッド**: OpenCV可視化とユーザー入力を処理

### スレッド安全性

- `LatestFrameContainer`: `threading.Lock()`を使用した安全なフレーム共有
- `queue.Queue`: Leap listenerとwriter間のスレッドセーフバッファ
- トリガーモニターと他のスレッド間に共有状態なし (コールバックベース)

## 開発

### テストの実行

```bash
# USB-IOモニターテスト
cd tests
python test_usb_io_monitor.py
```

### アーカイブされたファイル

開発履歴ファイルは`archive/`ディレクトリに保存されています。詳細は`archive/README_ARCHIVE.md`を参照してください。

**注意**: アーカイブファイルは参照用です。本番環境では`src/record_with_trigger.py`を使用してください。

## トラブルシューティング

### 問題: 高精度タイマーの初期化に失敗

**症状**: "High precision timer initialization failed"でプログラムが終了

**解決策**: これは予期された動作です。タイマーはUSB-IO同期に不可欠で、失敗した場合はデータの整合性を保証できません。

### 問題: USB-IOデバイスが見つからない

**症状**: "Warning: Failed to open USB-IO device"

**解決策**:
- USB-IOデバイスの接続を確認
- USB-IOドライバーをインストール
- プログラムはトリガー機能なしで続行します

### 問題: Arduinoが接続できない

**症状**: "Failed to connect to Arduino" または "No Arduino found"

**解決策**:
- CH340ドライバーがインストールされているか確認
- デバイスマネージャーでCOMポート番号を確認し、`--port COM9` のように指定
- Arduino IDEのシリアルモニタが開いていないか確認（ポートの競合）
- `arduino/ttl_trigger/ttl_trigger.ino` がArduinoに書き込まれているか確認

### 問題: Arduinoのトリガーが記録されない

**症状**: トリガー信号を送っているがカウントが増えない

**解決策**:
- TTL信号がPin 2 (D2)に接続されているか確認
- GNDが共通接続されているか確認
- TTL信号が3.3V以上のHIGHレベルであるか確認
- デバウンス設定（デフォルト1ms）より短いパルスは無視される

### 問題: Leap Motionが検出されない

**症状**: フレームが受信されない、空の可視化

**解決策**:
- Ultraleap Trackingソフトウェアが実行中であることを確認
- デバイスの接続を確認
- 公式のUltraleap VisualizerでLeap Motionが動作することを確認

## ドキュメント

詳細なプロジェクト構造とファイル説明については、[プロジェクト構造ドキュメント](docs/PROJECT_STRUCTURE.md)を参照してください。

追加の技術ドキュメント:
- **英語**:
  - `docs/USB_IO_INTEGRATION.md` - USB-IO統合の詳細
  - `docs/TIMESTAMP_SYNC.md` - タイムスタンプ同期の説明

- **日本語**:
  - `docs/PROJECT_STRUCTURE.md` - 詳細なプロジェクト構造説明

## ライセンス

詳細はLICENSE.mdを参照してください。

## サポート

以下に関する問題について:
- **Leap Motion SDK**: [Ultraleap Support](mailto:support@ultraleap.com)
- **このプロジェクト**: GitHubでissueを開いてください

## バージョン履歴

- **v0.1.0** (2026-01-22): 初回構造化リリース
  - イベント駆動型USB-IO統合
  - 高精度タイムスタンプ同期
  - 安定したマルチスレッドアーキテクチャ
