#!/usr/bin/env python3
"""
Rubric Evaluation Results to CSV Converter

This script converts rubric evaluation JSON files to CSV format for easier analysis.
Handles rubric_evaluated/rubric_hle_{model}.json files and converts them to CSV.

Usage:
    python convert_rubric_to_csv.py rubric_evaluated/rubric_hle_deepseek-reasoner.json
    python convert_rubric_to_csv.py rubric_evaluated/rubric_hle_deepseek-reasoner.json output.csv
"""

import json
import csv
import sys
import os
from typing import Dict, List, Any

def load_rubric_results(json_file: str) -> Dict[str, Any]:
    """Load rubric evaluation results from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def convert_to_csv_rows(rubric_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert rubric results to CSV-ready rows"""
    rows = []
    
    for question_id, data in rubric_results.items():
        if "rubric_evaluation" not in data:
            print(f"Warning: No rubric evaluation found for {question_id}, skipping...")
            continue
        
        rubric_eval = data["rubric_evaluation"]
        
        # Create CSV row
        row = {
            "question_id": question_id,
            "model": data.get("model", "unknown"),
            "score": rubric_eval["score"],
            "reasoning": rubric_eval["reasoning"].replace("\n", " ").replace("\r", ""),
            "category_assessment": rubric_eval["category_assessment"].replace("\n", " ").replace("\r", ""),
            "logical_consistency": rubric_eval["logical_consistency"].replace("\n", " ").replace("\r", ""),
            "factual_accuracy": rubric_eval["factual_accuracy"].replace("\n", " ").replace("\r", ""),
            # Include original response for reference (truncated if too long)
            "model_response": data["response"][:500] + "..." if len(data["response"]) > 500 else data["response"],
            # Include usage info if available
            "prompt_tokens": data.get("usage", {}).get("prompt_tokens", ""),
            "completion_tokens": data.get("usage", {}).get("completion_tokens", ""),
            "total_tokens": data.get("usage", {}).get("total_tokens", "")
        }
        
        rows.append(row)
    
    return rows

def save_to_csv(rows: List[Dict[str, Any]], output_file: str):
    """Save rows to CSV file"""
    if not rows:
        print("No data to save!")
        return
    
    fieldnames = [
        "question_id",
        "model", 
        "score",
        "reasoning",
        "category_assessment",
        "logical_consistency", 
        "factual_accuracy",
        "model_response",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens"
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Successfully converted {len(rows)} evaluations to {output_file}")

def generate_statistics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate basic statistics from the data"""
    if not rows:
        return {}
    
    scores = [int(row["score"]) for row in rows]
    
    stats = {
        "total_evaluations": len(scores),
        "average_score": round(sum(scores) / len(scores), 2),
        "score_distribution": {str(i): scores.count(i) for i in range(1, 6)},
        "model_name": rows[0]["model"] if rows else "unknown"
    }
    
    return stats

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_rubric_to_csv.py <input_json_file> [output_csv_file]")
        print("\nExample:")
        print("  python convert_rubric_to_csv.py rubric_evaluated/rubric_hle_deepseek-reasoner.json")
        print("  python convert_rubric_to_csv.py rubric_evaluated/rubric_hle_deepseek-reasoner.json my_results.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Generate output filename if not provided
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # rubric_hle_model.json -> rubric_hle_model.csv
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.csv"
    
    try:
        # Load and convert data
        print(f"Loading rubric results from: {input_file}")
        rubric_results = load_rubric_results(input_file)
        
        print(f"Converting {len(rubric_results)} evaluations to CSV format...")
        rows = convert_to_csv_rows(rubric_results)
        
        # Save to CSV
        save_to_csv(rows, output_file)
        
        # Show statistics
        stats = generate_statistics(rows)
        if stats:
            print(f"\n*** Statistics ***")
            print(f"Model: {stats['model_name']}")
            print(f"Total evaluations: {stats['total_evaluations']}")
            print(f"Average score: {stats['average_score']}/5")
            print(f"Score distribution: {stats['score_distribution']}")
        
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_file}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()