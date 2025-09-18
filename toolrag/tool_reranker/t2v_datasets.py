from torch.utils.data import Dataset
import torch


class T2VDatasetQueryNT(Dataset):
    def __init__(self, data_dir, tool_name_dir, tool_embedding_dir, tool_top_k_retrieval_dir=None, is_valid=False, num_tools_to_be_presented=64):
        # Dummy loader: just loads preprocessed data
        import json, pickle
        with open(data_dir, 'r') as f:
            self.data = json.load(f)
        with open(tool_name_dir, 'r') as f:
            self.tool_names = json.load(f)
        with open(tool_embedding_dir, 'rb') as f:
            self.tool_embeddings = pickle.load(f)

        self.num_tools = num_tools_to_be_presented
        self.is_valid = is_valid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # Handle both old format (functions, instruction) and new format (query, tool_name)
        if "query" in entry and "tool_name" in entry:
            # New format from tool_embeddings_test.json
            query = entry["query"]
            tool_name = entry["tool_name"]
            
            # Get tool embedding
            if tool_name in self.tool_embeddings:
                tool_embedding = torch.tensor(self.tool_embeddings[tool_name]).unsqueeze(0)
            else:
                # Fallback: use a zero embedding if tool not found
                if self.tool_embeddings:
                    first_embedding = next(iter(self.tool_embeddings.values()))
                    embedding_dim = len(first_embedding) if hasattr(first_embedding, '__len__') else 384
                else:
                    embedding_dim = 384
                tool_embedding = torch.zeros(1, embedding_dim)
            
            return {
                "query": query,
                "label": torch.tensor([1.0]),
                "tool_embedding": tool_embedding,
                "true_tools": [tool_name],
                "labeled_tools": [tool_name],
            }
        else:
            # Old format (backward compatibility)
            query = entry.get("refined_instruction") or entry.get("instruction", "")
            functions = entry.get("functions", [])
            
            if functions and functions[0] in self.tool_embeddings:
                tool_embedding = torch.tensor(self.tool_embeddings[functions[0]]).unsqueeze(0)
            else:
                # Fallback: use a zero embedding if tool not found
                if self.tool_embeddings:
                    first_embedding = next(iter(self.tool_embeddings.values()))
                    embedding_dim = len(first_embedding) if hasattr(first_embedding, '__len__') else 384
                else:
                    embedding_dim = 384
                tool_embedding = torch.zeros(1, embedding_dim)
            
            return {
                "query": query,
                "label": torch.tensor([1.0]),
                "tool_embedding": tool_embedding,
                "true_tools": functions,
                "labeled_tools": functions,
            }
    
def t2v_collator_query_nt(batch, tokenizer):
    queries = [item["query"] for item in batch]
    encodings = tokenizer(queries, return_tensors="pt", padding=True, truncation=True)

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "tool_embedding": torch.stack([item["tool_embedding"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "true_tools": [item["true_tools"] for item in batch],
        "labeled_tools": [item["labeled_tools"] for item in batch],
    }