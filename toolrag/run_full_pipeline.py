#!/usr/bin/env python3
"""
Full pipeline script for tool2vec training and evaluation.
This script orchestrates the entire workflow:
1. Download and parse data
2. Generate embeddings
3. Train the model
4. Evaluate the model
"""

import os
import sys
import subprocess
import json
import shutil
import argparse
from pathlib import Path
import shlex

def run_command(cmd, cwd=None):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    print(f"Success: {cmd}")
    return True

def generate_embeddings(train_file, val_file, test_file):
    """Generate embeddings for all datasets."""
    print("=== Step 1: Generating embeddings ===")
    embedding_commands = [
        f"python tool2vec/embedding_generator.py --data_path {train_file} --output_path output/ --output_file_name train_embeddings_embedded.json",
        f"python tool2vec/embedding_generator.py --data_path {val_file} --output_path output/ --output_file_name val_embeddings_embedded.json",
        f"python tool2vec/embedding_generator.py --data_path {test_file} --output_path output/ --output_file_name test_embeddings_embedded.json"
    ]
    for cmd in embedding_commands:
        if not run_command(cmd):
            return False
    print("✓ Embedding generation completed")
    return True

def train_model(val_data, batch_size, num_epochs, num_tools_presented, train_file):
    """Train the tool reranker model."""
    print("=== Step 2: Training model ===")
    checkpoint_path = "toolrefiner/checkpoints/model_epoch_1.pt"
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Removed existing checkpoint: {checkpoint_path}")
    cmd = f"""python toolrefiner/train_query_nt.py \
        --model "microsoft/deberta-v3-base" \
        --lr 8e-4 \
        --use_amp \
        --std 0.2 \
        --wd 0.01 \
        --num_linear_warmup_steps 100 \
        --training_data_dir output/train_embeddings_embedded.json \
        --test_data_dir {val_data} \
        --tool_embedding_dir output/train_embeddings_embedded.pkl \
        --tool_name_dir {train_file} \
        --batch_size {batch_size} \
        --num_tools_to_be_presented {num_tools_presented} \
        --num_epochs {num_epochs} \
        --train_tool_top_k_retrieval_dir train_top_k.json \
        --valid_tool_top_k_retrieval_dir val_top_k.json \
        --checkpoint_dir toolrefiner/checkpoints/"""
    if not run_command(cmd):
        return False
    print("✓ Model training completed")
    return True

def evaluate_with_selector(test_file):
    """Evaluate the trained model using query_tool_selector.py for each test query."""
    print("=== Step 3: Evaluating model with query_tool_selector.py ===")
    checkpoint_dir = "toolrefiner/checkpoints"
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory {checkpoint_dir} not found")
        return False
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        print(f"Error: No checkpoint files found in {checkpoint_dir}")
        return False
    latest_checkpoint = sorted(checkpoints)[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    with open(test_file, "r") as f:
        test_queries = json.load(f)
    for i, entry in enumerate(test_queries):
        query = entry["query"]
        output_file = f"output/selected_tools_{i}.json"
        safe_query = shlex.quote(query)
        cmd = (
            f"python toolrefiner/query_tool_selector.py "
            f"--query {safe_query} "
            f"--model_name microsoft/deberta-v3-base "
            f"--checkpoint_path {checkpoint_path} "
            f"--tool_embeddings_path output/test_embeddings_embedded.pkl "
            f"--tool_names_path {test_file} "
            f"--output_file {output_file} "
            f"--tool_embedding_dim 768 "
        )
        print(f"\nEvaluating query {i+1}/{len(test_queries)}: {query}")
        if not run_command(cmd):
            print(f"Failed to evaluate query: {query}")
    print("\n✓ Model evaluation with query_tool_selector.py completed")
    return True

def parse_args():
    parser = argparse.ArgumentParser(description="Run the full tool2vec pipeline")
    parser.add_argument("--train-file", type=str, required=True, help="Path to train.json")
    parser.add_argument("--val-file", type=str, required=True, help="Path to val.json")
    parser.add_argument("--test-file", type=str, required=True, help="Path to test.json")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training and evaluation (default: 4)")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of training epochs (default: 1)")
    parser.add_argument("--num-tools-presented", type=int, default=16, help="Number of tools to be presented during training/evaluation (default: 16)")
    return parser.parse_args()

def main():
    args = parse_args()
    print("Starting tool2vec full pipeline...")
    print(f"Configuration:")
    print(f"  Train file: {args.train_file}")
    print(f"  Val file: {args.val_file}")
    print(f"  Test file: {args.test_file}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of epochs: {args.num_epochs}")
    print(f"  Number of tools presented: {args.num_tools_presented}")
    print()
    # Step 1: Generate embeddings
    if not generate_embeddings(args.train_file, args.val_file, args.test_file):
        print("Pipeline failed at embedding generation step")
        return False
    # Step 2: Train model
    val_data = "output/val_embeddings_embedded.json"
    if not train_model(val_data, args.batch_size, args.num_epochs, args.num_tools_presented, args.train_file):
        print("Pipeline failed at model training step")
        return False
    # Step 3: Evaluate model
    if not evaluate_with_selector(args.test_file):
        print("Pipeline failed at model evaluation step")
        return False
    print("\n🎉 Full pipeline completed successfully!")
    print("Results should be available in the output files and evaluation logs.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
