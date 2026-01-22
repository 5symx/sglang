import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sglang.srt.predictor.expert_predictor import ExpertPredictor
import sys
# Assume we already have training data:
# embeddings: shape (N, embed_dim)
# expert_ids: shape (N, expert_id_dim) [optional]
# labels: shape (N,) with integer expert indices (ground truth top-1 expert)


def topk_accuracy(probs, labels, k=2):
    """
    Compute top-k accuracy.
    
    Args:
        probs (Tensor): shape (batch_size, num_experts), predicted probabilities.
        labels (Tensor): shape (batch_size,), ground truth expert indices.
        k (int): number of top experts to consider.
    
    Returns:
        accuracy (float): fraction of samples where true label is in top-k predictions.
    """
    # Get top-k indices
    topk_indices = torch.topk(probs, k=k, dim=-1).indices  # shape (batch_size, k)
    
    # Compare labels with top-k predictions
    correct = (topk_indices == labels.unsqueeze(1)).any(dim=1).float()
    
    return correct.mean().item()


N = 1000
embed_dim = 256
num_experts = 16
expert_id_dim = 8

embeddings = torch.randn(N, embed_dim)
expert_ids = torch.randn(N, expert_id_dim)
labels = torch.randint(0, num_experts, (N,))

dataset = TensorDataset(embeddings, expert_ids, labels)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for batch in dataloader: 
    emb, eid, lbl = batch 
    print("Embedding shape:", emb.shape) # (batch_size, embed_dim) 
    print("Expert ID shape:", eid.shape) # (batch_size, expert_id_dim) 
    print("Labels shape:", lbl.shape) # (batch_size,) 
    break
sys.exit(0)

# Model, loss, optimizer
model = ExpertPredictor(embed_dim, num_experts, expert_id_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Training loop
num_epochs = 5
# top1
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
    
#     for batch in dataloader:
#         emb, eid, lbl = batch
        
#         optimizer.zero_grad()
#         probs, _ = model(emb, eid, topk=2)
        
#         loss = criterion(probs, lbl)
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item() * emb.size(0)
#         _, predicted = probs.max(1)
#         correct += (predicted == lbl).sum().item()
#         total += lbl.size(0)
    
#     avg_loss = total_loss / total
#     acc = correct / total
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

# topk
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_top1 = 0
    correct_topk = 0
    total = 0
    
    for emb, eid, lbl in dataloader:
        optimizer.zero_grad()
        probs, _ = model(emb, eid, topk=2)
        
        loss = criterion(probs, lbl)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * emb.size(0)
        
        # Top-1 accuracy
        _, predicted = probs.max(1)
        correct_top1 += (predicted == lbl).sum().item()
        
        # Top-k accuracy
        correct_topk += (torch.topk(probs, k=2, dim=-1).indices == lbl.unsqueeze(1)).any(dim=1).sum().item()
        
        total += lbl.size(0)
    
    avg_loss = total_loss / total
    acc_top1 = correct_top1 / total
    acc_topk = correct_topk / total
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, "
          f"Top-1 Acc: {acc_top1:.4f}, Top-2 Acc: {acc_topk:.4f}")
