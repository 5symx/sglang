import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertPredictorSingleLabel(nn.Module):
    def __init__(self, embed_dim=2048, num_experts=128, hidden_dim_huge=2048, hidden_dim=1024):
        super().__init__()
        hidden_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim_huge)
        self.fc2 = nn.Linear(hidden_dim_huge, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_experts)

    def forward(self, embedding):
        x = F.relu(self.fc1(embedding))
        x = F.relu(self.fc2(x))
        logits = self.out(x)  # (batch_size, num_experts)
        return logits



class ExpertPredictorMultiLabelSimple(nn.Module):
    def __init__(self, embed_dim, num_experts, hidden_dim=512):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_experts)

    def forward(self, embedding):
        """
        Args:
            embedding (Tensor): shape (batch_size, embed_dim)
        Returns:
            logits (Tensor): raw scores (batch_size, num_experts)
            probs (Tensor): sigmoid probabilities (batch_size, num_experts)
        """
        x = F.relu(self.fc1(embedding))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        probs = torch.sigmoid(logits)  # independent probability per expert
        return logits, probs


class ExpertPredictor(nn.Module):
    def __init__(self, embed_dim, num_experts, expert_id_dim=0, hidden_dim=512):
        """
        Args:
            embed_dim (int): Dimension of token embeddings.
            num_experts (int): Total number of experts in the MoE.
            expert_id_dim (int): Dimension of expert ID encoding (0 if not used).
            hidden_dim (int): Hidden layer size for MLP.
        """
        super().__init__()

        input_dim = embed_dim + expert_id_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_experts)

    def forward(self, embedding, expert_id=None, topk=2):
        """
        Args:
            embedding (Tensor): shape (batch_size, embed_dim)
            expert_id (Tensor): optional one-hot or embedding of expert ID, shape (batch_size, expert_id_dim)
            topk (int): number of experts to select
        Returns:
            probs (Tensor): probability distribution over experts
            topk_indices (Tensor): indices of top-k predicted experts
        """
        if expert_id is not None:
            x = torch.cat([embedding, expert_id], dim=-1)
        else:
            x = embedding

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(probs, k=topk, dim=-1)
        return probs, topk_indices


class ExpertPredictorMultiLabel(nn.Module):
    def __init__(self, embed_dim, num_experts, expert_id_dim=0, hidden_dim=512):
        super().__init__()

        input_dim = embed_dim + expert_id_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_experts)

    def forward(self, embedding, expert_id=None):
        if expert_id is not None:
            x = torch.cat([embedding, expert_id], dim=-1)
        else:
            x = embedding
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)  # raw scores
        return logits


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    embed_dim = 256
    num_experts = 16
    expert_id_dim = 8   # optional one-hot encoding

    model = ExpertPredictor(embed_dim, num_experts, expert_id_dim)

    # Dummy input
    embedding = torch.randn(batch_size, embed_dim)
    expert_id = torch.randn(batch_size, expert_id_dim)

    probs, topk_indices = model(embedding, expert_id, topk=2)
    print("Top-k expert indices:", topk_indices)
