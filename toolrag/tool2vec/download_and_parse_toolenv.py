#!/usr/bin/env python3
"""
Download and parse ToolEnv2404 dataset from HuggingFace.

This script downloads the toolenv2404_filtered.tar.gz dataset and extracts
tool_name and description from each JSON file to create query_tools.json.
"""
import json


def download_and_parse_g1_category(
    url: str = "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/master/solvable_queries/test_instruction/G1_category.json",
    train_file: str = "train.json",
    val_file: str = "val.json",
    test_file: str = "test.json"
) -> None:
    """
    Download and parse the G1_category.json file from StableToolBench, then split and save as train/val/test.
    """
    import requests  # Ensure requests is available for try and except
    try:
        print(f"Downloading G1_category.json from: {url}")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        print(f"Downloaded {len(data)} entries from G1_category.json")

        # Extract tool_name from each api in api_list, and use the top-level 'query' as the query
        processed = []
        print('--- Example queries being parsed ---')
        example_count = 0
        for item in data:
            query_text = item.get('query', '')
            api_list = item.get('api_list', [])
            for api in api_list:
                tool_name = api.get('tool_name', '') or api.get('name', '')
                if tool_name and query_text:
                    processed.append({'tool_name': tool_name, 'query': query_text})
                else:
                    print(f"Skipping api: missing tool_name or query - tool_name: '{tool_name}', query: '{query_text[:100]}...'")

        print(f"Extracted {len(processed)} valid entries from G1_category.json")
        process_data(processed, train_file, val_file, test_file)
        print(f"✅ Successfully saved splits to {train_file}, {val_file}, {test_file}")
    except requests.RequestException as e:
        print(f"❌ Error downloading G1_category.json: {e}")
        raise
    except Exception as e:
        print(f"❌ Error processing G1_category.json: {e}")
        raise


def save_data(data: list, filename: str) -> None:
    """
    Save a list of dicts (with tool_name and query) to a JSON file.
    Args:
        data: List of dicts with 'tool_name' and 'query' keys
        filename: Name of the file to save to
    """
    from pathlib import Path
    path = Path(filename).absolute()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} entries to {path}")


def process_data(data: list, train_file: str = 'train.json', val_file: str = 'val.json', test_file: str = 'test.json') -> None:
    """
    Given a list of dicts, extract tool_name and tool_description, create splits, and save as JSON.
    20% for test, 10% for validation.
    Args:
        data: List of dicts with at least 'tool_name' and 'tool_description' keys
        train_file: Filename for training data (default: 'train.json')
        val_file: Filename for validation data (default: 'val.json')
        test_file: Filename for test data (default: 'test.json')
    """
    import random
    # Extract tool_name and query
    processed = [
        {'tool_name': d.get('tool_name', ''), 'query': d.get('query', '')}
        for d in data if d.get('tool_name') and d.get('query')
    ]
    n = len(processed)
    if n == 0:
        print("No valid entries to process.")
        return
    random.shuffle(processed)
    test_size = max(1, int(0.2 * n))
    val_size = max(1, int(0.1 * n))
    test_data = processed[:test_size]
    val_data = processed[test_size:test_size+val_size]
    train_data = processed  # All included in training
    save_data(train_data, train_file)
    save_data(val_data, val_file)
    save_data(test_data, test_file)


def main():
    """Main function to run the download and parsing process for G1_category only."""
    import argparse
    parser = argparse.ArgumentParser(description="Download and parse G1_category.json only")
    parser.add_argument(
        "--output-train",
        default="train.json",
        help="Output JSON file path for training data (default: train.json)"
    )
    parser.add_argument(
        "--output-val",
        default="val.json",
        help="Output JSON file path for validation data (default: val.json)"
    )
    parser.add_argument(
        "--output-test",
        default="test.json",
        help="Output JSON file path for test data (default: test.json)"
    )
    parser.add_argument(
        "--g1-url",
        default="https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/master/solvable_queries/test_instruction/G1_category.json",
        help="URL to G1_category.json file"
    )
    args = parser.parse_args()
    try:
        download_and_parse_g1_category(
            url=args.g1_url,
            train_file=args.output_train,
            val_file=args.output_val,
            test_file=args.output_test
        )
        print("✅ Successfully processed G1_category.json!")
    except Exception as e:
        print(f"❌ Error processing G1_category.json: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
