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

SYSTEM_EXACT_ANSWER = "Your response should be in the following format:\nExplanation: {your explanation for your final answer}\nExact Answer: {your succinct, final answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"

SYSTEM_MC = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"


def extract_final_answer(response_text):
    """Extract the final answer from response text for aggregation"""
    # For exact_match questions
    exact_match = re.search(r'Exact Answer:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE)
    if exact_match:
        return exact_match.group(1).strip()
    
    # For multiple choice questions
    mc_match = re.search(r'Answer:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE)
    if mc_match:
        return mc_match.group(1).strip()
    
    # Fallback: return the last line if no pattern matches
    lines = response_text.strip().split('\n')
    return lines[-1].strip() if lines else ""


def aggregate_responses(responses):
    """Aggregate multiple responses using majority voting"""
    if not responses:
        return None
    
    # If only one response, return it
    if len(responses) == 1:
        return responses[0]
    
    # Extract final answers from all responses
    final_answers = []
    for response in responses:
        if response:
            answer = extract_final_answer(response)
            if answer:
                final_answers.append(answer)
    
    if not final_answers:
        return responses[0]  # Fallback to first response
    
    # Count occurrences and find majority
    answer_counts = Counter(final_answers)
    most_common_answer = answer_counts.most_common(1)[0][0]
    
    # Find the first response that contains the majority answer
    for response in responses:
        if response and most_common_answer in extract_final_answer(response):
            return response
    
    # Fallback to first response
    return responses[0]


def format_message(args, question):
    answer_type = question['answer_type']
    system_prompt = SYSTEM_EXACT_ANSWER if answer_type == 'exact_match' else SYSTEM_MC
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


async def attempt_question(args, question):
    messages = format_message(args, question)
    try:
        # Build API parameters
        api_params = {
            "model": args.model,
            "max_completion_tokens": args.max_completion_tokens,
            "messages": messages,
            "stream": False,
        }
        
        # Add temperature if specified and not o1 model
        if hasattr(args, 'temperature') and args.temperature is not None and "o1" not in args.model:
            api_params["temperature"] = args.temperature
        
        # Add top_p if specified
        if hasattr(args, 'top_p') and args.top_p is not None:
            api_params["top_p"] = args.top_p
        
        # Add n_samples if specified and > 1
        if hasattr(args, 'n_samples') and args.n_samples is not None and args.n_samples > 1:
            api_params["n"] = args.n_samples
        
        response = await client.chat.completions.create(**api_params)
        
        # Handle multiple samples
        if hasattr(args, 'n_samples') and args.n_samples is not None and args.n_samples > 1:
            choices = [choice.message.content for choice in response.choices]
            final_content = aggregate_responses(choices)
        else:
            final_content = response.choices[0].message.content
        
        tokens = json.loads(response.usage.model_dump_json())
    except Exception as e:
        print("Error:", e)
        return None
    
    if final_content is None: # failed
        return None
    
    return question["id"], final_content, tokens


async def attempt_all(args, questions):
    async def bound_func(question):
        async with semaphore:
            content = await attempt_question(args, question)
            return content
            
    semaphore = asyncio.Semaphore(args.num_workers)
    async with semaphore:
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