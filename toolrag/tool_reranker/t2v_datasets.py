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
        return {
            "query": entry["refined_instruction"] or entry["instruction"],
            "label": torch.tensor([1.0]),
            "tool_embedding": torch.tensor(self.tool_embeddings[entry["functions"][0]]).unsqueeze(0),
            "true_tools": entry["functions"],
            "labeled_tools": entry["functions"],
        }
    
def t2v_collator_query_nt(batch, tokenizer):
    queries = [item["query"] for item in batch]
    encodings = tokenizer(queries, return_tensors="pt", padding=True, truncation=True)

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "tool_embedding": torch.stack([item["tool_embedding"].squeeze(0) for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "true_tools": [item["true_tools"] for item in batch],
        "labeled_tools": [item["labeled_tools"] for item in batch],
    }