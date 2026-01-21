# アーカイブファイルについて

このディレクトリには、プロジェクトの開発過程で作成された旧バージョンのファイルが保管されています。

## ディレクトリ構成

```
archive/
├── development/           # 開発過程のコードとドキュメント
│   ├── record_and_visualize.py
│   ├── record_and_visualize260120.py
│   ├── record_simple.py
│   ├── record_handtracking.py
│   ├── test_record_trigger.py
│   └── RECORDING_IMPROVEMENTS.md
└── planning/             # 開発計画ドキュメント
    └── implementation_plan.md
```

## development/ - 開発履歴ファイル

### `record_and_visualize.py`
- **作成時期**: 初期開発段階
- **目的**: Leap Motionデータの記録と可視化を統合した最初のバージョン
- **特徴**:
  - 基本的なLeap Motionトラッキング
  - リアルタイム可視化
  - HDF5形式での保存
- **制限事項**: USB-IO統合なし、タイムスタンプ同期なし

### `record_and_visualize260120.py`
- **作成時期**: 2026年1月20日
- **目的**: `record_and_visualize.py`の改良版
- **特徴**: フレームドロップ防止機能の追加
- **注意**: まだUSB-IO統合前のバージョン

### `record_simple.py`
- **作成時期**: デバッグ段階
- **目的**: 最小限の機能でLeap Motionデータを記録（トラブルシューティング用）
- **特徴**:
  - タイムスタンプ同期なし
  - USB-IO統合なし
  - シンプルな実装で問題の切り分けに使用
- **用途**: `record_with_trigger.py`で問題が発生した際の比較用

### `test_record_trigger.py`
- **作成時期**: USB-IO統合のデバッグ段階
- **目的**: USB-IO統合の簡易版テスト
- **特徴**:
  - `TimestampConverter`を使用（複雑な同期ロジック含む）
  - OpenCV可視化あり
- **過去の問題**: デッドロック問題が発生したため、本番では使用しない
- **現在の状態**: 参考用として保存

### `record_handtracking.py`
- **作成時期**: プロジェクト初期
- **目的**: 基本的な手指トラッキングデータの記録
- **特徴**: シンプルな記録機能のみ

### `RECORDING_IMPROVEMENTS.md`
- **内容**: `record_and_visualize.py`の改善内容とフレームドロップ対策の技術文書
- **トピック**:
  - フレームドロップ抑制の実装（キューサイズ拡大、HDF5最適化）
  - フレームドロップ検出と統計機能
  - `analyze_recording.py`の使用方法
  - ファイルサイズ見積もりと技術的詳細
- **価値**: フレームドロップ防止の技術的判断の記録

## planning/ - 開発計画

### `implementation_plan.md`
- **作成者**: antigravity（開発初期のペアプログラミング相手）
- **内容**: プロジェクト初期の実装計画
- **価値**: 開発の意思決定の記録

## 開発の経緯

1. **初期**: `record_and_visualize.py` - 基本的な記録機能
2. **改良**: `record_and_visualize260120.py` - フレームドロップ対策
3. **統合開始**: `test_record_trigger.py` - USB-IO統合の試み（デッドロック発生）
4. **デバッグ**: `record_simple.py` - 問題の切り分け
5. **完成**: `../src/record_with_trigger.py` - シンプルなタイムスタンプ同期で安定動作

## 本番で使用するファイル

現在の本番用ファイルは `../src/record_with_trigger.py` です。

- 高精度タイムスタンプ同期 (perf_counter)
- USB-IO 2.0統合
- スレッドセーフな実装
- デッドロック問題を解決

## 参照方法

これらのファイルは、以下の目的で保管されています：

- **学習**: 開発過程の理解
- **比較**: 過去の実装との違いを確認
- **トラブルシューティング**: 問題発生時の切り分け
- **履歴**: 技術的な意思決定の記録

本番環境では使用しないでください。
