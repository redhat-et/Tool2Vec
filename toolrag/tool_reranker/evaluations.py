import torch

def top_k_recall(output_logits, candidate_tools, true_tools, top_k=5):
    """Simple top-k recall assuming candidate_tools are aligned with logits."""
    output_probs = torch.sigmoid(output_logits).cpu().detach().numpy()
    recalls = []
    for i, probs in enumerate(output_probs):
        top_k_indices = np.argsort(probs)[::-1][:top_k]
        predicted = set(candidate_tools[i][j] for j in top_k_indices)
        actual = set(true_tools[i])
        if not actual:
            continue
        recall = len(predicted & actual) / len(actual)
        recalls.append(recall)
    return np.mean(recalls) if recalls else 0.0