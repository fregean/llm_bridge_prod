# MIT License
# 
# Copyright (c) 2020 centerforaisafety
# Copyright (c) 2025 Kazuma Matsumoto
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import json
import argparse
import asyncio
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
import re
from collections import Counter

load_dotenv()

client = None

def format_message(args, question, system_prompt):
    question_text = question['question']

    text_content = dict(type="text", text=question_text)
    if question['image']: # "" if not multi-modal
        image_content = dict(type="image_url", image_url=dict(url=question['image']))
        content = [text_content, image_content]
    else:
        content = [text_content]

    system_role = "user" if "o1" in args.model else "system" # o1 no sys prompt
    messages = [
        {"role": system_role, "content": system_prompt}, 
        {"role": "user", "content": content}
    ]
    return messages

async def attempt_all(args, questions):
    # この内部関数が、1つの質問に対するループと集約の全処理を担います。
    async def process_question(question):
        try:
            # 質問タイプに応じて使用するプロンプトのリストを決定
            answer_type = question['answer_type']
            if answer_type == 'exact_match':
                variations = args.prompts.exact_answer_variations
            else:
                variations = args.prompts.mc_variations
            
            choices = []
            total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

            # TTSのためにn_samples回ループ処理を行う
            for i in range(args.n_samples):
                # このイテレーションで使うプロンプトを選択（リストを循環して使用）
                system_prompt = variations[i % len(variations)]
                messages = format_message(args, question, system_prompt)
                
                api_params = {
                    "model": args.model,
                    "max_tokens": args.max_completion_tokens,
                    "messages": messages,
                    "stream": False,
                }

                # deepseek-reasonerモデルではtemperature/top_pが効かないため、パラメータに追加しない
                is_reasoner = "reasoner" in args.model
                
                if not is_reasoner:
                    # o1モデルでもtemperatureは無効
                    if hasattr(args, 'temperature') and args.temperature is not None and "o1" not in args.model:
                        api_params["temperature"] = args.temperature
                    if hasattr(args, 'top_p') and args.top_p is not None:
                        api_params["top_p"] = args.top_p
                
                # 1回のAPI呼び出し
                response = await client.chat.completions.create(**api_params)
                
                content = response.choices[0].message.content
                if content:
                    choices.append(f"--- Response {i+1} ---\n{content}")
                
                # トークン使用量を合算する
                if response.usage:
                    total_usage["prompt_tokens"] += response.usage.prompt_tokens
                    total_usage["completion_tokens"] += response.usage.completion_tokens
                    total_usage["total_tokens"] += response.usage.total_tokens

            if not choices:
                return None
            # -------------------------------------------------------------
            # Self-MoA Aggregation Step
            # -------------------------------------------------------------
            # 集約プロンプトを組み立てる
            all_responses_str = "\n\n".join(choices)
            aggregator_prompt = args.prompts.aggregator_prompt_template.format(
                original_question=question['question'],
                num_responses=len(choices),
                all_responses=all_responses_str
            )
            
            # 質問タイプに応じて、最終的な出力フォーマット指示を追加する
            if question['answer_type'] == 'exact_match':
                final_format_instruction = "\n\nYour final aggregated response should use the Exact Answer format."
            else:
                final_format_instruction = "\n\nYour final aggregated response should use the Answer format."
            
            # 集約用のAPIを呼び出す
            messages = [
                {"role": "user", "content": aggregator_prompt + final_format_instruction}
            ]
            
            agg_response = await client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=args.max_completion_tokens,
                stream=False,
            )

            final_content = agg_response.choices[0].message.content
            # -------------------------------------------------------------

            # トークン使用量はループ分と集約分を合算する必要がある（ここでは簡略化）
            if agg_response.usage:
                total_usage["prompt_tokens"] += agg_response.usage.prompt_tokens
                total_usage["completion_tokens"] += agg_response.usage.completion_tokens
                total_usage["total_tokens"] += agg_response.usage.total_tokens

            if final_content is None:
                return None
            
            return question["id"], final_content, total_usage

        except Exception as e:
            print(f"Error processing question {question['id']}: {e}")
            return None

    # 並列処理の管理部分は元の構造を維持
    async def bound_func(question):
        async with semaphore:
            return await process_question(question)
            
    semaphore = asyncio.Semaphore(args.num_workers)
    tasks = [bound_func(q) for q in questions]
    results = await tqdm_asyncio.gather(*tasks)
    return results

    
def main(args):
    global client
    
    # Initialize client based on base_url
    if hasattr(args, 'base_url') and args.base_url == "https://api.deepseek.com":
        client = AsyncOpenAI(
            base_url=args.base_url,
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            timeout=600.0,
            max_retries=1,
        )
    else:
        client = AsyncOpenAI(
            timeout=600.0,
            max_retries=1,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    
    assert args.num_workers > 1, "num_workers must be 2 or greater"
    dataset = load_dataset(args.dataset, split="test")
    dataset = dataset.filter(lambda item: item['image'] == "")
    dataset = dataset.to_dict()

    # convert to list of json for async parallelism
    questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]
    
    # Question selection with priority: indices > range > max_samples
    if hasattr(args, 'question_indices') and args.question_indices:
        # Select specific indices
        questions = [questions[i] for i in args.question_indices if i < len(questions)]
        print(f"Selected {len(questions)} questions by indices: {args.question_indices}")
    elif hasattr(args, 'question_range') and args.question_range and len(args.question_range) == 2:
        # Select range
        start, end = args.question_range
        questions = questions[start:end]
        print(f"Selected questions from index {start} to {end} ({len(questions)} questions)")
    elif args.max_samples:
        # Default: select first N questions
        questions = questions[:args.max_samples]
        print(f"Selected first {len(questions)} questions (max_samples: {args.max_samples})")
    
    output_filepath = f"predictions/hle_{os.path.basename(args.model)}.json"   


    # load only questions without responses
    if os.path.exists(output_filepath):
        with open(output_filepath, "r") as f:
            predictions = json.load(f)
        questions = [q for q in questions if q["id"] not in predictions]
    else:
        predictions = {}

    results = asyncio.run(attempt_all(args, questions))

    # You can rerun this script multiple times if there are failed API calls
    for result in results:
        if result is None: # API call failed
            continue
        unique_id, response, usage = result
        predictions[unique_id] = {
            "model": args.model,
            "response": response,
            "usage": usage
        }

    # cache responses
    with open(output_filepath, "w") as f:
        json.dump(predictions, f, indent=4)