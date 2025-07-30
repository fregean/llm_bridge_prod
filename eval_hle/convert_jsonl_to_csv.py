#!/usr/bin/env python3
"""
Convert results.jsonl from HLE evaluation to CSV format.

Usage:
   python convert_jsonl_to_csv.py <input_jsonl_file> [output_csv_file]

   output_csv_fileが指定されない場合、入力ファイル名と同じディレクトリに自動生成されます。
"""

import json
import csv
import sys
import os
from pathlib import Path

def convert_jsonl_to_csv(input_file, output_file=None):
    
    # Auto-generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}.csv"
    
    try:
        with open(input_file, 'r', encoding='utf-8') as jsonl_file:
            # Read first line to get column names
            first_line = jsonl_file.readline().strip()
            if not first_line:
                print(f"Error: {input_file} is empty")
                return False
            
            first_record = json.loads(first_line)
            fieldnames = list(first_record.keys())
            
            # Reset file pointer to beginning
            jsonl_file.seek(0)
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                
                line_count = 0
                for line in jsonl_file:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            writer.writerow(record)
                            line_count += 1
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping invalid JSON on line {line_count + 1}: {e}")
                            continue
                
                print(f"Successfully converted {line_count} records from {input_file} to {output_file}")
                return True
                
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_jsonl_to_csv.py <input_jsonl_file> [output_csv_file]")
        print("\nExample:")
        print("  python convert_jsonl_to_csv.py leaderboard/2025_07_28_14_21_28/results.jsonl")
        print("  python convert_jsonl_to_csv.py leaderboard/2025_07_28_14_21_28/results.jsonl results.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)
    
    success = convert_jsonl_to_csv(input_file, output_file)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()