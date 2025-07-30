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
import yaml
import asyncio
import numpy as np
from typing import Literal, Dict, List, Any
from pydantic import BaseModel
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(
    timeout=300.0,
    max_retries=1,
    api_key=os.environ.get("OPENAI_API_KEY"),
)

class RubricEvaluation(BaseModel):
    score: Literal[1, 2, 3, 4, 5]
    reasoning: str
    category_assessment: str
    logical_consistency: str
    factual_accuracy: str

def load_rubric(rubric_path: str) -> List[Dict]:
    """Load evaluation rubric from YAML file"""
    with open(rubric_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_rubric_prompt(rubric: List[Dict], question: str, correct_answer: str, 
                        rationale: str, model_response: str, category: str) -> str:
    """Create evaluation prompt with rubric criteria"""
    
    # Format rubric criteria
    rubric_text = "\n".join([
        f"**Score {item['Score']}** - {item['Definition (Overall Evaluation)']}:\n"
        f"{item['Typical Characteristics']}\n"
        f"Example: {item['Example (Summary)']}\n"
        for item in rubric
    ])
    
    prompt = f"""You are an expert evaluator assessing the quality of AI model responses using a 5-point rubric.

**EVALUATION RUBRIC:**
{rubric_text}

**QUESTION DETAILS:**
- Category: {category}
- Question: {question}
- Correct Answer: {correct_answer}
- Expert Rationale: {rationale}

**MODEL RESPONSE TO EVALUATE:**
{model_response}

**EVALUATION INSTRUCTIONS:**
1. Assess the model's response against the correct answer and expert rationale
2. Consider logical consistency, factual accuracy, and reasoning quality
3. Assign a score from 1-5 based on the rubric criteria
4. Provide detailed reasoning for your assessment

Please evaluate the response and provide:
- score: Integer from 1-5 matching the rubric criteria
- reasoning: Detailed explanation of your assessment (200-400 words)
- category_assessment: How well the response addresses the specific {category} domain (50-100 words)
- logical_consistency: Assessment of logical flow and coherence (50-100 words)  
- factual_accuracy: Assessment of factual correctness compared to expert answer (50-100 words)
"""
    return prompt

async def evaluate_response(args, question_data: Dict, prediction_data: Dict, rubric: List[Dict]) -> Dict:
    """Evaluate a single response using the rubric"""
    prompt = create_rubric_prompt(
        rubric=rubric,
        question=question_data["question"],
        correct_answer=question_data["answer"],
        rationale=question_data["rationale"],
        model_response=prediction_data["response"],
        category=question_data["category"]
    )
    try:
        response = await client.beta.chat.completions.parse(
            model=args.judge,
            max_completion_tokens=2048,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format=RubricEvaluation,
        )
        
        evaluation = response.choices[0].message.parsed
        return {
            "score": evaluation.score,
            "reasoning": evaluation.reasoning,
            "category_assessment": evaluation.category_assessment,
            "logical_consistency": evaluation.logical_consistency,
            "factual_accuracy": evaluation.factual_accuracy
        }
    except Exception as e:
        print(f"Error evaluating {question_data['id']}: {e}")
        return None

async def add_rubric_evaluation(args, question_data: Dict, predictions: Dict, rubric: List[Dict]):
    """Add rubric evaluation to a single prediction"""
    question_id = question_data["id"]
    if question_id not in predictions:
        return None, None
    
    prediction_data = predictions[question_id]
    
    # Skip if already evaluated
    if "rubric_evaluation" in prediction_data:
        return question_id, prediction_data
    
    evaluation = await evaluate_response(args, question_data, prediction_data, rubric)
    
    if evaluation:
        prediction_data["rubric_evaluation"] = evaluation
        return question_id, prediction_data
    else:
        return None, None

async def evaluate_all_responses(args, questions: List[Dict], predictions: Dict, rubric: List[Dict]):
    """Evaluate all responses using the rubric"""
    async def bound_func(question):
        async with semaphore:
            return await add_rubric_evaluation(args, question, predictions, rubric)
    
    semaphore = asyncio.Semaphore(args.num_workers)
    async with semaphore:
        tasks = [bound_func(q) for q in questions]
        results = await tqdm_asyncio.gather(*tasks)
    return results

def generate_rubric_statistics(predictions: Dict) -> Dict:
    """Generate statistics from rubric evaluations"""
    scores = []
    category_scores = {}
    
    for pred in predictions.values():
        if "rubric_evaluation" in pred:
            score = pred["rubric_evaluation"]["score"]
            scores.append(score)
            
            # Get category from original question data (we'll need to pass this)
            # For now, we'll skip category breakdown
    
    if not scores:
        return {"error": "No evaluations found"}
    
    stats = {
        "total_evaluations": len(scores),
        "average_score": round(np.mean(scores), 2),
        "median_score": int(np.median(scores)),
        "score_distribution": {
            str(i): scores.count(i) for i in range(1, 6)
        },
        "std_deviation": round(np.std(scores), 2)
    }
    
    return stats

def save_rubric_results(args, predictions: Dict, statistics: Dict):
    """Save rubric evaluation results"""
    output_folder = "rubric_evaluated"
    os.makedirs(output_folder, exist_ok=True)
    
    # Save detailed results
    output_filepath = f"{output_folder}/rubric_hle_{os.path.basename(args.model)}.json"
    with open(output_filepath, "w") as f:
        json.dump(predictions, f, indent=4)
    
    # Save statistics
    stats_filepath = f"{output_folder}/rubric_stats_{os.path.basename(args.model)}.json"
    with open(stats_filepath, "w") as f:
        json.dump({
            "model_name": predictions[list(predictions.keys())[0]]["model"],
            "evaluation_timestamp": datetime.now().isoformat(),
            "statistics": statistics
        }, f, indent=4)
    
    print(f"\n*** Rubric Evaluation Results ***")
    print(f"Total evaluations: {statistics['total_evaluations']}")
    print(f"Average score: {statistics['average_score']}/5")
    print(f"Median score: {statistics['median_score']}/5")
    print(f"Standard deviation: {statistics['std_deviation']}")
    print(f"Score distribution: {statistics['score_distribution']}")
    print(f"\nResults saved to: {output_filepath}")
    print(f"Statistics saved to: {stats_filepath}")

def main(args):
    """Main function for rubric evaluation"""
    assert args.num_workers > 1, "num_workers must be 2 or greater"
    
    # Load rubric
    rubric = load_rubric("utils/evaluation_rubric.yaml")
    
    # Load dataset
    dataset = load_dataset(args.dataset, split="test").to_dict()
    all_questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]
    
    # Load predictions
    if args.predictions_file:
        predictions_file = args.predictions_file
    else:
        predictions_file = f"predictions/hle_{os.path.basename(args.model)}.json"
    
    with open(predictions_file, "r") as f:
        predictions = json.load(f)
    
    # Load existing evaluations if they exist
    output_filepath = f"rubric_evaluated/rubric_hle_{os.path.basename(args.model)}.json"
    if os.path.exists(output_filepath):
        with open(output_filepath, "r") as f:
            evaluated_predictions = json.load(f)
    else:
        evaluated_predictions = {}
    
    # Filter questions that need evaluation
    questions_to_evaluate = [
        q for q in all_questions 
        if q["id"] in predictions and q["id"] not in evaluated_predictions
    ]
    
    print(f"Total questions in dataset: {len(all_questions)}")
    print(f"Questions with predictions: {len([q for q in all_questions if q['id'] in predictions])}")
    print(f"Questions needing evaluation: {len(questions_to_evaluate)}")
    
    if questions_to_evaluate:
        # Run evaluation
        results = asyncio.run(evaluate_all_responses(args, questions_to_evaluate, predictions, rubric))
        
        # Update evaluated predictions
        for question_id, prediction in results:
            if question_id:
                evaluated_predictions[question_id] = prediction
    
    # Generate statistics and save results
    statistics = generate_rubric_statistics(evaluated_predictions)
    save_rubric_results(args, evaluated_predictions, statistics)