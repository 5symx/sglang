# predict topk by using label of expert count across layers.

from sglang.srt.predictor.expert_predictor import ExpertPredictorSingleLabel

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import os
import json
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

# Parameters
N = 1000
embed_dim = 2048
num_experts = 128
k=8



def load_data(json_path):
    if os.path.exists(json_path):
        print("Loading dataset from JSON...")
        with open(json_path, "r") as f:
            data = json.load(f)

        embeddings = torch.tensor([sample["embedding"] for sample in data], dtype=torch.float32)
        labels = torch.tensor([sample["label"] for sample in data], dtype=torch.float32)

    else:
        print("JSON file not found, generating synthetic data...")
        embeddings = torch.randn(N, embed_dim)
        labels = torch.zeros(N, num_experts)
        for i in range(N):
            active = torch.randint(0, num_experts, (2,))  # randomly activate 2 experts
            labels[i, active] = 1.0

    # Wrap into TensorDataset
    dataset = TensorDataset(embeddings, labels)

    

    # Suppose dataset is your TensorDataset
    train_size = int(0.8 * len(dataset))   # 80% train
    test_size = len(dataset) - train_size  # 20% test
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Inspect one batch
    for emb, lbl in train_loader:
        print("Embedding shape:", emb.shape)   # (batch_size, embed_dim)
        print("Label shape:", lbl.shape)       # (batch_size, num_experts)
        break

    return train_loader,test_loader



def ndcg_at_k(pred_scores, true_relevance, k=10):
    """
    pred_scores: (batch_size, num_experts) predicted scores
    true_relevance: (batch_size, num_experts) true counts or relevance
    """
    # Get top-k predicted indices
    topk_pred = torch.topk(pred_scores, k=k, dim=-1).indices
    
    # Gather true relevance for predicted order
    rel = true_relevance.gather(1, topk_pred)
    
    # Compute DCG
    discounts = torch.log2(torch.arange(k, device=pred_scores.device) + 2).float()
    dcg = (rel / discounts).sum(dim=1)
    
    # Compute IDCG (ideal ranking)
    ideal_rel = torch.topk(true_relevance, k=k, dim=-1).values
    idcg = (ideal_rel / discounts).sum(dim=1)
    
    # Avoid division by zero
    ndcg = (dcg / (idcg + 1e-8)).mean().item()
    return ndcg


def evaluate(model, dataloader, criterion, k=10):
    model.eval()
    total_loss, total, correct_topk = 0, 0, 0
    ndcg_score = 0
    
    with torch.no_grad():
        for emb, lbl in dataloader:
            logits = model(emb)

            lbl_probs = lbl / lbl.sum(dim=1, keepdim=True)
            log_probs = F.log_softmax(logits, dim=1) # normalize predictions 
            loss = criterion(log_probs, lbl_probs)

            # loss = criterion(logits, lbl)
            total_loss += loss.item() * emb.size(0)
            total += emb.size(0)
            
            # Top-k accuracy
            topk_pred = torch.topk(logits, k=k, dim=-1).indices
            topk_true = torch.topk(lbl, k=k, dim=-1).indices
            match = (topk_pred.unsqueeze(2) == topk_true.unsqueeze(1))
            correct_topk += match.any(dim=2).any(dim=1).sum().item()
            
            # NDCG
            ndcg_score += ndcg_at_k(logits, lbl, k=k) * emb.size(0)
    
    avg_loss = total_loss / total
    acc_topk = correct_topk / total
    ndcg_score /= total
    return avg_loss, acc_topk, ndcg_score



def training(train_loader, test_loader, num_epochs = 5):
    # Model, loss, optimizer
    model = ExpertPredictorSingleLabel(embed_dim, num_experts)
    # criterion = nn.BCEWithLogitsLoss()  # multi-label loss # norm 0-1
    # criterion = nn.MSELoss() # raw 0-count
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.KLDivLoss(reduction="batchmean") 
    

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total = 0
        correct_topk = 0
        
        for emb, lbl in train_loader:
            optimizer.zero_grad()
            logits = model(emb)

            lbl_probs = lbl / lbl.sum(dim=1, keepdim=True)
            log_probs = F.log_softmax(logits, dim=1) # normalize predictions 
            loss = criterion(log_probs, lbl_probs)

            # loss = criterion(logits, lbl) # for criterion = nn.MSELoss()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * emb.size(0)
            total += emb.size(0)

            # lbl shape: (batch_size, 128)
            topk_pred  = torch.topk(logits, k=k, dim=-1).indices  # (batch_size, 5)
            topk_true = torch.topk(lbl, k=k, dim=-1).indices

            # Gather the true labels at the predicted indices
            match = (topk_pred.unsqueeze(2) == topk_true.unsqueeze(1))
            correct_topk += match.any(dim=2).any(dim=1).sum().item()

            
            # # Evaluate top-k accuracy (e.g., k=2)
            # topk_indices = torch.topk(logits, k=5, dim=-1).indices
            # # Check if any of the true experts are in predicted top-k
            # # correct_topk += (lbl.gather(1, topk_indices).sum(dim=1) > 0).sum().item()
            # correct_topk += (topk_indices == lbl.unsqueeze(1)).any(dim=1).sum().item()
        
        avg_loss = total_loss / total
        acc_topk = correct_topk / total
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Top-{k} Acc: {acc_topk:.4f}")
        
        ndcg_score = 0
        with torch.no_grad():
            for emb, lbl in train_loader:
                logits = model(emb)
                ndcg_score += ndcg_at_k(logits, lbl, k=k) * emb.size(0)
        ndcg_score /= total
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Top-{k} Acc: {acc_topk:.4f}, NDCG@{k}: {ndcg_score:.4f}")

        test_loss, test_acc, test_ndcg = evaluate(model, test_loader, criterion, k=k)
        print(f"Epoch {epoch+1}/{num_epochs}, " \
            f"Train Loss: {avg_loss:.4f}, Train Top-{k} Acc: {acc_topk:.4f}, Train NDCG@{k}: {ndcg_score:.4f}, " \
            f"Test Loss: {test_loss:.4f}, Test Top-{k} Acc: {test_acc:.4f}, Test NDCG@{k}: {test_ndcg:.4f}")

if __name__ == "__main__":
    # json_path = "moe_dataset.json"
    json_path = "/home/ymx/sglang/scripts/my_test/output-122.json"
    train_loader,test_loader = load_data(json_path=json_path)
    training(train_loader,test_loader, num_epochs=30)