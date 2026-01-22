import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sglang.srt.predictor.expert_predictor import ExpertPredictorMultiLabel
# Example synthetic dataset
N = 1000
embed_dim = 256
num_experts = 16
expert_id_dim = 8

embeddings = torch.randn(N, embed_dim)
expert_ids = torch.randn(N, expert_id_dim)

# Labels: binary vectors (multi-label ground truth)
labels = torch.zeros(N, num_experts)
for i in range(N):
    # randomly activate 2 experts per sample
    active = torch.randint(0, num_experts, (2,))
    labels[i, active] = 1.0

dataset = TensorDataset(embeddings, expert_ids, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, loss, optimizer
model = ExpertPredictorMultiLabel(embed_dim, num_experts, expert_id_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total = 0
    correct_topk = 0
    
    for emb, eid, lbl in dataloader:
        optimizer.zero_grad()
        logits = model(emb, eid)
        loss = criterion(logits, lbl)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * emb.size(0)
        total += emb.size(0)
        
        # Evaluate top-k accuracy (e.g., k=2)
        probs = torch.sigmoid(logits)
        topk_indices = torch.topk(probs, k=2, dim=-1).indices
        # Check if any of the true experts are in predicted top-k
        correct_topk += (lbl.gather(1, topk_indices).sum(dim=1) > 0).sum().item()
    
    avg_loss = total_loss / total
    acc_topk = correct_topk / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Top-2 Acc: {acc_topk:.4f}")
