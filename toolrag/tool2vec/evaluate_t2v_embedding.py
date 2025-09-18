"""
Evaluate the tool2vec embedding using recall@k metric.
Updated to work with new tool_embeddings_test.json format.
Usage:
    python tool2vec/evaluate_t2v_embedding.py
"""

import json
import argparse
import numpy as np
import pickle

from pathlib import Path

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--valid_data_path",
    type=str,
    help="Path to the file containing valid/test data (tool_embeddings_test.json)",
    required=True,
)
argparser.add_argument(
    "--t2v_embedding_path",
    type=str,
    help="Path to tool2vec embedding pickle file (tool_embeddings_test.pkl)",
    required=True,
)
argparser.add_argument(
    "--top_k_list",
    type=int,
    nargs="+",
    default=[1, 3, 5, 7, 10, 12, 32, 64, 128, 256],
    help="Top k list, default is [1, 3, 5, 7, 10, 12, 32, 64, 128, 256]",
)
argparser.add_argument("--debug", action="store_true", help="Debug mode")
argparser.add_argument("--output_file_name", type=str, help="Output file")

args = argparser.parse_args()


def compute_recall_k(
    valid_data: list,
    t2v_data: np.array,
    tool_to_idx: dict[str, int],
    top_k: int = 5,
) -> tuple[float, float]:
    """
    Compute recall@k for the given data and t2v embeddings.

    Args:
        valid_data (list): The valid/test data with query_embedding and tool_name.
        t2v_data (np.array): The t2v embeddings.
        tool_to_idx (dict): The tool to index mapping.
        top_k (int): The top k value.

    Returns:
        tuple: (recall@k, ndcg@k)
    """
    total_recall = 0
    total_ndcg = 0

    for _, valid in enumerate(valid_data):
        # Use query_embedding from the new format
        valid_tool_embedding = np.array(valid["query_embedding"])

        # calculate cosine similarity
        valid_tool_embedding = valid_tool_embedding.reshape(1, -1)
        cosine_sim = np.dot(t2v_data, valid_tool_embedding.T)

        # get top k
        top_k_idx = np.argsort(cosine_sim.flatten())[::-1][:top_k]

        # Use tool_name from the new format (single tool per entry)
        correct_tool_idx = set([tool_to_idx.get(valid["tool_name"], -1)])

        # calculate recall
        recall = len(set(top_k_idx) & correct_tool_idx) / len(correct_tool_idx)
        total_recall += recall

        # calculate ndcg
        dcg = 0
        for idx, tool in enumerate(top_k_idx):
            if tool in correct_tool_idx:
                dcg += 1 / np.log2(2 + idx)

        max_dcg = 0
        for idx in range(min(top_k, len(correct_tool_idx))):
            max_dcg += 1 / np.log2(2 + idx)
        ndcg = dcg / max_dcg if max_dcg > 0 else 0
        total_ndcg += ndcg

    return total_recall / len(valid_data), total_ndcg / len(valid_data)


def load_tool_embeddings_data(valid_data_path: Path, t2v_embedding_path: Path):
    """
    Load data from the new tool_embeddings_test.json format.
    
    Args:
        valid_data_path: Path to tool_embeddings_test.json
        t2v_embedding_path: Path to tool_embeddings_test.pkl
        
    Returns:
        tuple: (valid_data, t2v_embedding, tool_to_idx, idx_to_tool)
    """
    print("Loading tool_embeddings_test.json...")
    with open(valid_data_path, "r") as f:
        valid_data = json.load(f)

    print("Loading tool_embeddings_test.pkl...")
    with open(t2v_embedding_path, "rb") as f:
        t2v_embedding = pickle.load(f)

    print(f"Loaded {len(valid_data)} entries from {valid_data_path}")
    print(f"Loaded {len(t2v_embedding)} tool embeddings from {t2v_embedding_path}")

    # Create tool name to index mapping from the pickle file
    idx_to_tool, tool_to_idx = {}, {}
    for idx, tool in enumerate(t2v_embedding.keys()):
        idx_to_tool[idx] = tool
        tool_to_idx[tool] = idx

    print(f"Found {len(tool_to_idx)} unique tools in embeddings")

    return valid_data, t2v_embedding, tool_to_idx, idx_to_tool


def main():
    valid_data_path = Path(args.valid_data_path)
    t2v_embedding_path = Path(args.t2v_embedding_path)

    if not valid_data_path.exists():
        raise FileNotFoundError(f"Valid data file not found: {valid_data_path}")
    
    if not t2v_embedding_path.exists():
        raise FileNotFoundError(f"T2V embedding file not found: {t2v_embedding_path}")

    # Load the data
    valid_data, t2v_embedding, tool_to_idx, idx_to_tool = load_tool_embeddings_data(
        valid_data_path, t2v_embedding_path
    )

    # Convert embeddings to numpy array
    t2v_data_np = np.array([t2v_embedding[idx_to_tool[idx]] for idx in idx_to_tool])
    print(f"T2V data shape: {t2v_data_np.shape}")

    # Calculate recalls at different k values
    recalls_at_k = {}
    ndcgs_at_k = {}
    
    for k in args.top_k_list:
        if k <= len(tool_to_idx):  # Only evaluate k values that make sense
            recall_k, ndcg_k = compute_recall_k(
                valid_data=valid_data,
                t2v_data=t2v_data_np,
                tool_to_idx=tool_to_idx,
                top_k=k,
            )
            print(f"Recall@{k}: {recall_k:.4f}")
            print(f"NDCG@{k}: {ndcg_k:.4f}")
            recalls_at_k[k] = recall_k
            ndcgs_at_k[k] = ndcg_k
        else:
            print(f"Skipping k={k} (greater than number of tools: {len(tool_to_idx)})")

    # Save results if output file specified
    if args.output_file_name:
        with open(args.output_file_name, "w") as f:
            f.write("Evaluation Results for tool_embeddings_test.json\n")
            f.write("=" * 50 + "\n\n")
            for k in sorted(recalls_at_k.keys()):
                f.write(f"Recall@{k}: {recalls_at_k[k]:.4f}\n")
                f.write(f"NDCG@{k}: {ndcgs_at_k[k]:.4f}\n")
                f.write("-" * 20 + "\n")
        print(f"Results saved to {args.output_file_name}")

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total test entries: {len(valid_data)}")
    print(f"Total unique tools: {len(tool_to_idx)}")
    print(f"Embedding dimension: {t2v_data_np.shape[1]}")
    print("\nTop-k Results:")
    for k in sorted(recalls_at_k.keys()):
        print(f"  k={k:2d}: Recall={recalls_at_k[k]:.4f}, NDCG={ndcgs_at_k[k]:.4f}")


if __name__ == "__main__":
    main()
