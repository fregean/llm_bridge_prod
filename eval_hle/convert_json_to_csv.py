#!/usr/bin/env python3
"""
Convert JSON or JSONL files to CSV format.

Supports both:
- JSONL files (one JSON object per line)
- JSON files (single JSON array or object containing arrays)

Usage:
   python convert_jsonl_to_csv.py <input_file> [output_csv_file]

   output_csv_fileが指定されない場合、入力ファイル名と同じディレクトリに自動生成されます。
"""

import json
import csv
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Union

def detect_file_format(input_file: str) -> str:
    """
    Detect if the input file is JSON, JSONL, or rubric format.
    
    Returns:
        'json' for JSON files
        'jsonl' for JSONL files
        'rubric' for rubric evaluation files
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # Read entire file to properly parse JSON
            content = f.read().strip()
            if not content:
                return 'unknown'
            
            # Try to parse as complete JSON
            try:
                data = json.loads(content)
                
                # Check if it's an array
                if isinstance(data, list):
                    return 'json'
                elif isinstance(data, dict):
                    # Check for rubric evaluation format first
                    if _is_rubric_format(data):
                        return 'rubric'
                    
                    # Check if it contains arrays (standard JSON with nested arrays)
                    for value in data.values():
                        if isinstance(value, list) and len(value) > 0:
                            return 'json'
                    
                    # Single object, treat as JSON
                    return 'json'
                else:
                    return 'unknown'
                    
            except json.JSONDecodeError:
                # If full JSON parsing fails, try JSONL (line by line)
                lines = content.split('\n')
                valid_json_lines = 0
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            json.loads(line)
                            valid_json_lines += 1
                        except json.JSONDecodeError:
                            break
                
                if valid_json_lines > 1:
                    return 'jsonl'
                else:
                    return 'unknown'
                
    except Exception:
        return 'unknown'
    
    return 'unknown'


def _is_rubric_format(data: Dict[str, Any]) -> bool:
    """
    Check if the JSON data follows rubric evaluation format.
    
    Rubric format: {question_id: {model: ..., rubric_evaluation: {...}}}
    """
    if not isinstance(data, dict):
        return False
    
    # Check if values are objects with rubric_evaluation field
    for key, value in data.items():
        if isinstance(value, dict) and 'rubric_evaluation' in value:
            return True
    
    return False


def process_rubric_file(input_file: str) -> List[Dict[str, Any]]:
    """
    Process a rubric evaluation JSON file and flatten it for CSV conversion.
    
    Input format: {question_id: {model: ..., rubric_evaluation: {...}}}
    Output: List of flattened dictionaries
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, dict):
        raise ValueError(f"Expected object for rubric format, got {type(data)}")
    
    records = []
    
    for question_id, question_data in data.items():
        if not isinstance(question_data, dict):
            print(f"Warning: Skipping non-object data for question {question_id}")
            continue
        
        # Create flattened record
        record = {
            'question_id': question_id,
            'model': question_data.get('model', ''),
        }
        
        # Add main fields
        if 'response' in question_data:
            # Truncate long responses for readability
            response = question_data['response']
            if len(response) > 500:
                record['model_response'] = response[:500] + '...'
            else:
                record['model_response'] = response
        
        # Add usage information if available
        usage = question_data.get('usage', {})
        record['prompt_tokens'] = usage.get('prompt_tokens', '')
        record['completion_tokens'] = usage.get('completion_tokens', '')
        record['total_tokens'] = usage.get('total_tokens', '')
        
        # Add rubric evaluation fields
        if 'rubric_evaluation' in question_data:
            rubric_eval = question_data['rubric_evaluation']
            record['score'] = rubric_eval.get('score', '')
            
            # Clean up text fields (remove newlines for CSV compatibility)
            for field in ['reasoning', 'category_assessment', 'logical_consistency', 'factual_accuracy']:
                if field in rubric_eval:
                    text = rubric_eval[field].replace('\n', ' ').replace('\r', '')
                    record[field] = text
        
        records.append(record)
    
    return records


def process_json_file(input_file: str) -> List[Dict[str, Any]]:
    """
    Process a JSON file and extract records for CSV conversion.
    
    Handles:
    - JSON arrays: [obj1, obj2, ...]
    - JSON objects with array values: {"results": [obj1, obj2, ...]}
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    
    if isinstance(data, list):
        # Direct array of objects
        records = data
    elif isinstance(data, dict):
        # Object containing arrays - find the array(s)
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                # Check if the list contains objects (not primitives)
                if isinstance(value[0], dict):
                    records.extend(value)
        
        # If no arrays found, treat the single object as a record
        if not records:
            records = [data]
    else:
        raise ValueError(f"Unsupported JSON structure: expected array or object, got {type(data)}")
    
    return records


def process_jsonl_file(input_file: str) -> List[Dict[str, Any]]:
    """
    Process a JSONL file and extract records for CSV conversion.
    """
    records = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    records.append(record)
                    line_count += 1
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_count + 1}: {e}")
                    continue
    
    return records


def convert_to_csv(input_file: str, output_file: str = None) -> bool:
    """
    Convert JSON or JSONL file to CSV format.
    
    Args:
        input_file: Path to input JSON/JSONL file
        output_file: Path to output CSV file (auto-generated if None)
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    # Auto-generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}.csv"
    
    try:
        # Detect file format
        file_format = detect_file_format(input_file)
        print(f"Detected format: {file_format.upper()}")
        
        # Process file based on format
        if file_format == 'json':
            records = process_json_file(input_file)
        elif file_format == 'jsonl':
            records = process_jsonl_file(input_file)
        elif file_format == 'rubric':
            records = process_rubric_file(input_file)
        else:
            print(f"Error: Unable to determine file format for {input_file}")
            return False
        
        if not records:
            print(f"Error: No valid records found in {input_file}")
            return False
        
        # Get fieldnames from first record
        fieldnames = list(records[0].keys())
        
        # Write to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in records:
                writer.writerow(record)
        
        print(f"Successfully converted {len(records)} records from {input_file} to {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_jsonl_to_csv.py <input_file> [output_csv_file]")
        print("\nSupports JSON, JSONL, and rubric evaluation formats:")
        print("  JSON:   Single array or object containing arrays")
        print("  JSONL:  One JSON object per line")
        print("  RUBRIC: Nested object structure from rubric evaluations")
        print("\nExamples:")
        print("  python convert_json_to_csv.py data.json")
        print("  python convert_json_to_csv.py results.jsonl")
        print("  python convert_json_to_csv.py rubric_evaluated/rubric_hle_model.json")
        print("  python convert_json_to_csv.py leaderboard/2025_07_28_14_21_28/results.jsonl results.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)
    
    success = convert_to_csv(input_file, output_file)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()