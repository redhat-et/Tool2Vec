"""
Tool Selection Script

This script provides a function to take a query and tool embeddings,
then return and save the selected tools based on the trained model.
"""

import json
import pickle
import torch
import torch.nn as nn
from transformers import AutoTokenizer, DebertaV2Model
from pathlib import Path
from typing import List, Dict, Any
import argparse


class T2VPretrainedReranker(nn.Module):
    """Same model class as in the training script"""
    EMB_DIM_SIZE = {
        "bert-base-uncased": 768,
        "microsoft/deberta-v3-xsmall": 384,
        "microsoft/deberta-v3-base": 768,
        "microsoft/deberta-v3-large": 1024,
    }

    def __init__(
        self, model_name, std=0.2, num_layer_to_freeze=0, use_cls=False, use_sep=False, tool_embedding_dim=384
    ):
        super().__init__()
        self.use_cls = use_cls
        self.use_sep = use_sep
        self.num_layer_to_freeze = num_layer_to_freeze
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dynamic embedding projection based on tool embedding dimension
        model_emb_size = self.EMB_DIM_SIZE[model_name]
        self.embedding_projection: nn.Linear = nn.Linear(tool_embedding_dim, model_emb_size)
        nn.init.normal_(self.embedding_projection.weight, mean=0, std=std)

        deberta = DebertaV2Model.from_pretrained(model_name)
        self.encoder = deberta.encoder
        self.linear = nn.Linear(model_emb_size, 1)
        self.embedding = deberta.embeddings

        # Store token IDs
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        cls_token_id: int = tokenizer.cls_token_id
        sep_token_id: int = tokenizer.sep_token_id

        self.register_buffer("cls_token_id", torch.tensor(cls_token_id))
        self.register_buffer("sep_token_id", torch.tensor(sep_token_id))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tool_embedding: torch.Tensor,
    ):
        input_ids = input_ids.to(self.device)
        tool_embedding = tool_embedding.to(self.device)

        # Project tool embeddings
        tool_embedding_proj = self.embedding_projection(tool_embedding)

        if self.use_cls:
            cls_token_id = self.cls_token_id.unsqueeze(0).expand(input_ids.shape[0], 1)
            input_ids = torch.cat([cls_token_id, input_ids], dim=1)
            attention_mask = torch.cat(
                [
                    torch.ones(attention_mask.shape[0], 1).to(self.device),
                    attention_mask,
                ],
                dim=1,
            )

        if self.use_sep:
            sep_token_id = self.sep_token_id.unsqueeze(0).expand(input_ids.shape[0], 1)
            input_ids = torch.cat([input_ids, sep_token_id], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(attention_mask.shape[0], 1).to(self.device),
                ],
                dim=1,
            )

        input_embeddings = self.embedding(input_ids=input_ids)
        num_tokens = input_embeddings.shape[1]

        # Concatenate embeddings
        embeddings = torch.cat([input_embeddings, tool_embedding_proj], dim=1)

        # Update attention mask
        num_tools = tool_embedding_proj.shape[1]
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(attention_mask.shape[0], num_tools).to(self.device),
            ],
            dim=1,
        )

        # Get encoder output
        encoder_output = self.encoder(embeddings, attention_mask).last_hidden_state[
            :, num_tokens:, :
        ]

        # Final output
        out = self.linear(encoder_output).squeeze(2)
        return out


def load_model_and_data(
    model_name: str,
    checkpoint_path: str,
    tool_embeddings_path: str,
    tool_names_path: str,
    tool_embedding_dim: int = 384
) -> tuple:
    """
    Load the trained model and tool data.
    
    Args:
        model_name: Name of the model (e.g., "microsoft/deberta-v3-base")
        checkpoint_path: Path to the model checkpoint
        tool_embeddings_path: Path to the tool embeddings pickle file
        tool_names_path: Path to the tool names JSON file
        tool_embedding_dim: Dimension of tool embeddings
        
    Returns:
        tuple: (model, tokenizer, tool_embeddings, tool_names)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # Load model
    model = T2VPretrainedReranker(
        model_name=model_name,
        std=0.2,
        num_layer_to_freeze=0,
        use_cls=True,
        use_sep=True,
        tool_embedding_dim=tool_embedding_dim
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load tool embeddings
    with open(tool_embeddings_path, 'rb') as f:
        tool_embeddings = pickle.load(f)
    
    # Load tool names
    with open(tool_names_path, 'r') as f:
        tool_names = json.load(f)
    
    return model, tokenizer, tool_embeddings, tool_names


def select_tools_for_query(
    query: str,
    model,
    tokenizer,
    tool_embeddings: Dict[str, List[float]],
    tool_names: Dict[str, str],
    top_k: int = 10,
    threshold: float = 0.5,
    save_results: bool = True,
    output_file: str = "selected_tools.json"
) -> List[Dict[str, Any]]:
    """
    Select tools for a given query using the trained model.
    
    Args:
        query: The input query string
        model: The trained T2VPretrainedReranker model
        tokenizer: The tokenizer for the model
        tool_embeddings: Dictionary mapping tool names to embeddings
        tool_names: Dictionary mapping tool names to tool names (for consistency)
        top_k: Number of top tools to return
        threshold: Minimum confidence threshold for tool selection
        save_results: Whether to save results to a JSON file
        output_file: Path to save the results
        
    Returns:
        List of selected tools with their scores and names
    """
    device = next(model.parameters()).device
    
    # Tokenize the query
    inputs = tokenizer(
        query,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Prepare tool embeddings for all tools
    tool_names_list = list(tool_embeddings.keys())
    tool_embeddings_tensor = torch.stack([
        torch.tensor(tool_embeddings[tool_name], dtype=torch.float32) for tool_name in tool_names_list
    ]).unsqueeze(0).to(device)  # Shape: [1, num_tools, embedding_dim]
    
    # Get model predictions
    with torch.no_grad():
        # Repeat input for all tools
        batch_size = len(tool_names_list)
        input_ids_batch = input_ids.repeat(batch_size, 1)
        attention_mask_batch = attention_mask.repeat(batch_size, 1)
        
        # Get predictions for each tool
        outputs = []
        for i in range(batch_size):
            single_tool_embedding = tool_embeddings_tensor[:, i:i+1, :]  # [1, 1, embedding_dim]
            output = model(input_ids, attention_mask, single_tool_embedding)
            outputs.append(output.item())
    
    # Convert to probabilities
    import numpy as np
    probabilities = torch.sigmoid(torch.tensor(outputs)).numpy()
    
    # Get top-k tools
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    
    # Filter by threshold and prepare results
    selected_tools = []
    for idx in top_indices:
        if probabilities[idx] >= threshold:
            tool_name = tool_names_list[idx]
            selected_tools.append({
                "tool_name": tool_name,
                "score": float(probabilities[idx]),
                "rank": len(selected_tools) + 1
            })
    
    # Save results if requested
    if save_results:
        results = {
            "query": query,
            "selected_tools": selected_tools,
            "total_tools_evaluated": len(tool_names_list),
            "threshold_used": threshold,
            "top_k_requested": top_k
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_file}")
    
    return selected_tools


def main():
    """Example usage of the tool selection function"""
    parser = argparse.ArgumentParser(description="Select tools for a query")
    parser.add_argument("--query", type=str, required=True, help="Input query")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base", help="Model name")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tool_embeddings_path", type=str, required=True, help="Path to tool embeddings pickle file")
    parser.add_argument("--tool_names_path", type=str, required=True, help="Path to tool names JSON file")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top tools to return")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum confidence threshold")
    parser.add_argument("--output_file", type=str, default="selected_tools.json", help="Output file path")
    parser.add_argument("--tool_embedding_dim", type=int, default=768, help="Tool embedding dimension")
    
    args = parser.parse_args()
    
    # Load model and data
    print("Loading model and data...")
    model, tokenizer, tool_embeddings, tool_names = load_model_and_data(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        tool_embeddings_path=args.tool_embeddings_path,
        tool_names_path=args.tool_names_path,
        tool_embedding_dim=args.tool_embedding_dim
    )
    
    print(f"Loaded {len(tool_embeddings)} tool embeddings")
    
    # Select tools for the query
    print(f"Selecting tools for query: '{args.query}'")
    selected_tools = select_tools_for_query(
        query=args.query,
        model=model,
        tokenizer=tokenizer,
        tool_embeddings=tool_embeddings,
        tool_names=tool_names,
        top_k=args.top_k,
        threshold=args.threshold,
        save_results=True,
        output_file=args.output_file
    )
    
    # Print results
    print(f"\nSelected {len(selected_tools)} tools:")
    for tool in selected_tools:
        print(f"  {tool['rank']}. {tool['tool_name']} (score: {tool['score']:.4f})")


if __name__ == "__main__":
    main()
