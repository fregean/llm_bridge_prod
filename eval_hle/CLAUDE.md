# HLE評価システム (eval_hle)

HLE (Humanity's Last Exam) ベンチマークを用いた大規模言語モデルの評価システム

## フォルダ構成

```
eval_hle/
├── CLAUDE.md                   # このファイル - HLE評価システムの説明
├── README.md                   # 一般的な説明
├── conf/
│   └── config.yaml            # 評価設定ファイル
├── hle_benchmark/
│   ├── __init__.py
│   ├── _configs.py            # 設定データクラス定義
│   ├── vllm_predictions.py    # vLLM用予測生成
│   ├── openai_predictions.py  # OpenAI API用予測生成
│   ├── ollama_predictions.py  # Ollama用予測生成
│   └── run_judge_results.py   # 評価実行エンジン
├── judge.py                   # 評価メインスクリプト
├── predict.py                 # 予測生成メインスクリプト
├── notebooks/
│   └── count_token.ipynb      # トークン数計算用ノートブック
├── requirements.txt           # 依存関係
├── predictions/               # 予測結果保存先 (実行時自動作成)
├── judged/                   # 評価結果保存先 (実行時自動作成)
└── leaderboard/              # 最終結果保存先 (実行時自動作成)
```

## 主要ファイル詳細

### 1. 訓練済みモデルテストファイル

#### `predict.py`
- **機能**: 訓練済みモデル（DeepSeekなど）をテストし予測を生成
- **使用方法**: 
  ```bash
  python predict.py
  ```
- **設定**: `conf/config.yaml`で以下を設定
  - `provider: vllm`
  - `model`: テスト対象モデル名
  - `base_url`: vLLMサーバーのURL

#### `hle_benchmark/vllm_predictions.py`
- **機能**: vLLMサーバー経由でモデル推論を実行
- **特徴**:
  - OpenAI互換API使用
  - 非同期処理でバッチ実行
  - マルチモーダル対応（画像なしのテキストのみフィルタリング）
  - エラー時の再実行対応

<details>
<summary><strong>vLLMサーバー設定例</strong></summary>
<div>

**単一ノード（8 GPU）**
```bash
vllm serve "$MODEL_PATH" \
  --served-model-name deepseek-r1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --distributed-executor-backend ray \
  --trust-remote-code \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --host 0.0.0.0 --port 8000 \
  --api-key p10-deepseek-key
```

**2ノード（16 GPU, 256 experts対応）**
```bash
vllm serve "$MODEL_PATH" \
  --served-model-name deepseek-r1 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --enable-expert-parallel \
  --distributed-executor-backend ray \
  --trust-remote-code \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 12 \
  --max-num-batched-tokens 6144 \
  --host 0.0.0.0 --port 8000 \
  --api-key p10-deepseek-key
```

<details>
<summary><strong>vLLMパラメータ詳細説明</strong></summary>
<div>

### 並列処理設定
- **`--tensor-parallel-size 8`**: GPU間でのモデル層分散数。モデルの重みを8つのGPUに分散して並列処理
- **`--pipeline-parallel-size 2`**: パイプライン並列処理グループ数。モデルを2つのステージに分けて異なるデバイスで順次処理
- **`--enable-expert-parallel`**: MoE（Mixture of Experts）モデル用の専門家並列化を有効化。DeepSeek-R1の256 expertsを効率的に処理

### 分散実行設定
- **`--distributed-executor-backend ray`**: 分散実行バックエンド。複数GPU/ノード環境では`ray`、単一ノードでは`mp`（multiprocessing）を使用

### メモリ・処理能力設定
- **`--gpu-memory-utilization 0.85`**: GPU メモリ使用率（0-1）。デフォルト0.9から0.85に下げて安定性を向上
- **`--max-model-len 4096`**: モデルコンテキスト長。入力+出力の最大トークン数
- **`--max-num-seqs 12`**: 1回の反復で処理する最大シーケンス数。同時処理可能なリクエスト数
- **`--max-num-batched-tokens 6144`**: 1回の反復で処理する最大トークン数。バッチ処理時のトークン上限

### その他
- **`--trust-remote-code`**: HuggingFaceからの外部コード実行を許可
- **`--dtype auto`**: モデルの精度を自動選択（通常はbf16/fp16）

</div>
</details>

**対応するconfig.yaml設定**
```yaml
provider: vllm
base_url: http://compute-node:8000/v1
model: deepseek-r1
api_key: p10-deepseek-key
max_completion_tokens: 38000
reasoning: true
num_workers: 2500
max_samples: 2500
```

**クライアント側の実装**
```python
# vllm_predictions.py内の設定
def main(args: Config):
    global client
    client = AsyncOpenAI(
        base_url=args.base_url,      # "http://compute-node:8000/v1"
        api_key=args.api_key,        # "p10-deepseek-key"
        timeout=86400,
        max_retries=3,
    )

# API呼び出し（基本）
response = await client.chat.completions.create(
    model=args.model,                # "deepseek-r1"
    messages=messages,
    max_completion_tokens=args.max_completion_tokens,
    stream=False,
)

# API呼び出し（多数サンプリング時）
response = await client.chat.completions.create(
    model=args.model,                # "deepseek-r1"
    messages=messages,
    n=16,                           # 16回サンプリング
    temperature=0.6,                # 温度設定
    max_completion_tokens=args.max_completion_tokens,
    stream=False,
)

# 16サンプルの処理
if args.n_samples and args.n_samples > 1:
    choices = [choice.message.content for choice in response.choices]
    # 集約処理（majority vote, self-consistency等）
    final_content = aggregate_responses(choices)
else:
    final_content = response.choices[0].message.content
```

</div>
</details>

#### 回答形式指定
問題タイプに応じて以下の形式で回答するようモデルに指示：

**厳密回答形式** (exact_match問題):
```
Explanation: {your explanation for your final answer}
Exact Answer: {your succinct, final answer}
Confidence: {your confidence score between 0% and 100% for your answer}
```

**選択肢形式** (multiple choice問題):
```
Explanation: {your explanation for your answer choice}
Answer: {your chosen answer}
Confidence: {your confidence score between 0% and 100% for your answer}
```

### 2. 予測結果の出力形式

#### `predictions/hle_{model_name}.json`
```json
{
  "question_id": {
    "model": "model_name",
    "response": "Explanation: 数学的には三角関数の性質から...\nExact Answer: 42\nConfidence: 95%",
    "usage": {
      "prompt_tokens": 123,
      "completion_tokens": 456,
      "total_tokens": 579
    }
  }
}
```

**注意**: 
- `response`フィールドには上記の指定形式に従った生の回答テキストがそのまま格納されます。構造化された抽出は評価段階で行われます。
- `usage`情報は**vLLMやOpenAI APIでは自動付与**されますが、**評価処理には不要**です。HuggingFaceモデルでの自前実装時は省略可能です。

### 3. API評価ファイル

#### `judge.py`
- **機能**: OpenAI APIモデル（o3-mini）による自動評価
- **使用方法**:
  ```bash
  python judge.py
  ```
- **評価者**: `conf/config.yaml`の`judge`パラメータで指定

#### `hle_benchmark/run_judge_results.py`
- **機能**: 予測結果の正誤判定と評価指標計算
- **評価プロセス**:
  1. 構造化プロンプトによる回答抽出
  2. 正解との照合
  3. 信頼度スコア抽出
  4. キャリブレーション誤差計算

**重要**: `calib_err`関数は**サンプル数が100問未満の場合は0.0を返す**仕様のため、`Calibration Error: 0.0`が表示されます。正確な評価には最低100問が必要です。

### 4. 評価結果の出力形式

#### `judged/judged_hle_{model_name}.json`
```json
{
  "question_id": {
    "model": "model_name",
    "response": "元の回答",
    "usage": {...},
    "judge_response": {
      "correct_answer": "正解",
      "model_answer": "抽出された回答",
      "reasoning": "判定理由",
      "correct": "yes/no",
      "confidence": 85
    }
  }
}
```

#### `leaderboard/{timestamp}/results.jsonl`
```jsonl
{"id": "q1", "category": "Math", "question": "...", "prediction": "...", "gold": "...", "correct": 1, "judgement": "..."}
{"id": "q2", "category": "Physics", "question": "...", "prediction": "...", "gold": "...", "correct": 0, "judgement": "..."}
```

#### `leaderboard/{timestamp}/summary.json`
```json
{
  "model_name": "model_name",
  "overall_accuracy": 75.5,
  "accuracy_per_category": {
    "Math": 80.2,
    "Physics": 70.1,
    "Biology/Medicine": 72.3,
    "Computer Science/AI": 78.9,
    "Engineering": 65.4,
    "Chemistry": 69.7,
    "Humanities/Social Science": 77.2,
    "Other": 73.8
  },
  "num_questions": 2500,
  "timestamp": "2025-01-31T12:34:56.789"
}
```

## 設定ファイル (`conf/config.yaml`)

```yaml
dataset: cais/hle                    # HuggingFace データセット
provider: vllm                       # 推論プロバイダー
base_url: http://localhost:8000/v1   # vLLMサーバーURL
model: deepseek/deepseek-r1-0528:free # テスト対象モデル
max_completion_tokens: 38000         # 最大生成トークン数
reasoning: true                      # 推論過程出力
num_workers: 2500                    # 並列処理数
max_samples: 2500                    # 最大サンプル数
judge: o3-mini-2025-01-31           # 評価用モデル
```

## 実行手順

### 1. 環境セットアップ
```bash
module load cuda/12.6 miniconda/24.7.1-py312 cudnn/9.6.0 nccl/2.24.3
conda activate llmbench
```

### 2. vLLMサーバー起動
```bash
vllm serve deepseek/deepseek-r1-0528:free --tensor-parallel-size 8 --max-model-len 131072
```

### 3. 予測生成
```bash
cd eval_hle
python predict.py
```

### 4. 評価実行
```bash
export OPENAI_API_KEY="your_api_key"
python judge.py
```

## 出力ファイル説明

- **predictions/**: モデルの生回答（JSON形式）
- **judged/**: 評価済み結果（正誤判定付き）
- **leaderboard/**: 最終評価結果（JSONL + サマリー）

評価結果は`leaderboard/`内のタイムスタンプ付きフォルダに保存され、リーダーボード統合用の標準形式で出力されます。

<details>
<summary><strong>JSONL→CSV変換ツール（オプション）</strong></summary>
<div>

`convert_jsonl_to_csv.py`スクリプトを使用して、results.jsonlファイルをCSV形式に変換できます。

**使用方法:**
```bash
# 基本使用法（出力ファイル名は自動生成: results.jsonl → results.csv）
python convert_jsonl_to_csv.py leaderboard/2025_07_28_14_21_28/results.jsonl

# 出力ファイル名を指定
python convert_jsonl_to_csv.py leaderboard/2025_07_28_14_21_28/results.jsonl my_results.csv

# 全フォルダを一括変換
for folder in leaderboard/*/; do python convert_jsonl_to_csv.py "${folder}results.jsonl"; done
```

**変換される項目:**
`id,category,question,user_prompt,answer_type,prediction,gold,correct,judgement`

**特徴:**
- UTF-8エンコーディング対応
- エラーハンドリング（無効なJSON行をスキップ）
- 進行状況と結果の表示

</div>
</details>

<details>
<summary><strong>Leaderboard形式変換</strong></summary>
<div>

### 変換処理フロー

判定結果から最終的なリーダーボード形式への変換処理：

```python
for k,v in predictions.items():
    data = next(filter(lambda x: x["id"] == k, all_questions))
    results.append({
        "id": k,                                          # 問題ID
        "category": data["category"],                     # 問題カテゴリ
        "question": data["question"],                     # 問題文
        "user_prompt": "",                               # TODO（未実装）
        "answer_type": data["answer_type"],              # 回答タイプ
        "prediction": v["judge_response"]["model_answer"],  # 抽出された回答
        "gold": v["judge_response"]["correct_answer"],    # 正解
        "correct": 1 if v["judge_response"]["correct"] == "yes" else 0,  # 1/0
        "judgement": v["judge_response"]["reasoning"],    # 判定理由
    })
```

### フィールド詳細

- **id**: 問題の一意識別子
- **category**: 問題分野（Math, Physics, Computer Science/AI, Biology/Medicine, Engineering, Chemistry, Humanities/Social Science, Other）
- **question**: 元の問題文テキスト
- **user_prompt**: モデルに送信された完全なプロンプト（現在未実装）
- **answer_type**: `exact_match` または `multiple_choice`
- **prediction**: AIが生成した回答から抽出された最終答え
- **gold**: データセットの正解
- **correct**: 正誤判定（1=正解, 0=不正解）
- **judgement**: OpenAI APIによる判定理由

### 出力例

```jsonl
{"id": "q001", "category": "Math", "question": "What is 2+2?", "user_prompt": "", "answer_type": "exact_match", "prediction": "4", "gold": "4", "correct": 1, "judgement": "The extracted answer '4' exactly matches the correct answer."}
{"id": "q002", "category": "Physics", "question": "What is the speed of light in vacuum?", "user_prompt": "", "answer_type": "exact_match", "prediction": "300000000", "gold": "299792458", "correct": 0, "judgement": "The extracted answer is approximately correct but not exact for the speed of light."}
```

</div>
</details>

<details>
<summary><strong>User Prompt拡張機能</strong></summary>
<div>

### 現在の実装状況
`user_prompt`フィールドは現在未実装（空文字）ですが、拡張により以下が可能：

### 1. 基本プロンプト記録
```python
user_prompt = f"""System: {system_prompt}

User: {question_text}"""
```

### 2. Test Time Scaling適用時
```python
user_prompt = f"""System: {system_prompt}

Please think step by step and show detailed reasoning. Take your time to consider multiple approaches before providing your final answer.

User: {question_text}"""
```

### 3. Self-MoA (Self-Mixture of Agents)適用時
```python
user_prompt = f"""System: {system_prompt}

You are an expert mathematician. First, approach this problem from a mathematical perspective.
Then, consider it as a physics expert would.
Finally, synthesize both perspectives for your comprehensive final answer.

User: {question_text}"""
```

### 4. 多数サンプリング時
```python
# API呼び出し時のハイパーパラメータ
response = await client.chat.completions.create(
    model=args.model,
    messages=messages,
    n=16,           # 16回サンプリング
    temperature=0.8,
    stream=False,
)

# user_promptには使用した設定を記録
user_prompt = f"""System: {system_prompt}

User: {question_text}

Applied Settings: n=16, temperature=0.8, aggregation=majority_vote"""
```

**注意**: 
- DeepSeek公式では多数サンプリング時に`n=16`を推奨しています
- `n`, `temperature`等はAPI呼び出し時のハイパーパラメータです
- `user_prompt`には使用した設定を**記録目的**で追記します

### 実装方法
`hle_benchmark/run_judge_results.py`の206行目を修正：
```python
"user_prompt": construct_full_prompt(args, data),  # 実装要
```

これにより高度な推論手法の追跡・再現・比較が可能になります。

</div>
</details>

<details>
<summary><strong>DeepSeek-R1 0528最適化設定</strong></summary>
<div>

### モデル仕様
- **パラメータ数**: 671B（6710億パラメータ）
- **アーキテクチャ**: Mixture of Experts（MoE）with 256 experts
- **メモリ要件**: 162GB+（量子化版）、715GB（フルモデル）
- **推奨温度**: 0.5-0.7（0.6推奨）

### 推奨vLLM設定

#### 8GPU構成（80GB VRAM each）
```bash
vllm serve "$MODEL_PATH" \
  --served-model-name deepseek-r1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --distributed-executor-backend ray \
  --trust-remote-code \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --max-num-seqs 8-12 \
  --max-num-batched-tokens 4096-6144 \
  --host 0.0.0.0 --port 8000
```

#### 16GPU構成（40GB VRAM each）
```bash
vllm serve "$MODEL_PATH" \
  --served-model-name deepseek-r1 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --enable-expert-parallel \
  --distributed-executor-backend ray \
  --trust-remote-code \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --max-num-seqs 6-8 \
  --max-num-batched-tokens 3072-4096 \
  --host 0.0.0.0 --port 8000
```

### 安全な初期設定
メモリ不足を避けるため、以下の保守的な設定から開始することを推奨：
```bash
--max-num-seqs 4
--max-num-batched-tokens 2048
--gpu-memory-utilization 0.8
```

その後、GPUメモリ使用量を監視しながら段階的に増加させてください。

### 考慮事項
1. **KVキャッシュ**: 長い推論（max 38000トークン）でメモリ使用量が大幅増加
2. **Expert routing**: MoEモデルの特性上、バッチによってメモリ使用量が変動
3. **Temperature設定**: 0.6推奨で多様性のある生成のため、バッチ効率が低下する可能性

### 参考リンク
- [DeepSeek-R1 Hardware Requirements](https://huggingface.co/deepseek-ai/DeepSeek-R1/discussions/19)
- [vLLM DeepSeek-R1 Optimization](https://developers.redhat.com/articles/2025/03/19/how-we-optimized-vllm-deepseek-r1)
- [DeepSeek-R1 Local Deployment Guide](https://medium.com/@isaakmwangi2018/a-simple-guide-to-deepseek-r1-architecture-training-local-deployment-and-hardware-requirements-300c87991126)
- [Unsloth DeepSeek-R1 Setup](https://docs.unsloth.ai/basics/deepseek-r1-0528-how-to-run-locally)

</div>
</details>

## Usage情報について

- **自動付与**: vLLMやOpenAI APIを使用する場合、トークン数情報は自動的に計算・付与されます
- **評価への影響**: usage情報は評価処理では使用されないため、省略しても評価品質に影響しません
- **用途**: 主にトークン数統計、コスト計算、パフォーマンス分析用
- **自前実装**: HuggingFaceモデルで直接推論する場合、usage情報は省略可能です