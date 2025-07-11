# === Imports ===
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from collections import Counter
from tqdm import tqdm

# === Load SNF matrix ===
base_dir = r"C:\UTM Degree\y4s2\Bio Modelling Simulation_Dr Azurah\Project"
fused = pd.read_csv(os.path.join(base_dir, "fused_similarity_matrix_removeNan.csv"), index_col=0)
fused.index = fused.index.str.strip().str.replace("-", ".")
fused.columns = fused.columns.str.strip().str.replace("-", ".")

# === Load labels ===
labels_df = pd.read_csv(os.path.join(base_dir, "aligned_labels.csv"))
labels_df['sampleID'] = labels_df['sampleID'].str.strip().str.replace("-", ".")
desired_order = ['Luminal A', 'Luminal B', 'HER2-enriched', 'Basal-like']
labels_df = labels_df[labels_df['label'].isin(desired_order)]
labels_df = labels_df.set_index("sampleID").loc[fused.index]
labels_df['label'] = pd.Categorical(labels_df['label'], categories=desired_order, ordered=True)
label_names = np.array(desired_order)
y = labels_df['label'].cat.codes.to_numpy()

# === Load reduced feature data ===
cnv = pd.read_csv(os.path.join(base_dir, "autoencoder_removeNan/DNA_cnv_reduced.csv"), index_col=0)
meth = pd.read_csv(os.path.join(base_dir, "autoencoder_removeNan/DNA_methylation_reduced.csv"), index_col=0)
mrna = pd.read_csv(os.path.join(base_dir, "autoencoder_removeNan/DNA_mrna_reduced.csv"), index_col=0)
for df in [cnv, meth, mrna]:
    df.index = df.index.str.strip().str.replace("-", ".")

# === Align and concatenate real features ===
x_df = pd.concat([cnv, meth, mrna], axis=1).loc[fused.index]
x = torch.tensor(x_df.values, dtype=torch.float)

# === Threshold weak edges in SNF matrix ===
threshold = 0.6
adj_matrix = fused.values.copy()
adj_matrix[adj_matrix < threshold] = 0
edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)
edge_weight = torch.tensor(adj_matrix[adj_matrix != 0], dtype=torch.float)

# === Split into train/test
train_idx, test_idx = train_test_split(np.arange(len(y)), stratify=y, test_size=0.2, random_state=42)

# === Class distribution checks
print("\n[BEFORE SPLIT] Class counts:")
print(pd.Series(y).value_counts().sort_index().rename(index=lambda i: label_names[i]))

print("\n[TRAIN SPLIT]")
train_counts = Counter(y[train_idx])
for i, label in enumerate(label_names):
    print(f"{label} ({i}): {train_counts.get(i, 0)}")

print("\n[TEST SPLIT]")
test_counts = Counter(y[test_idx])
for i, label in enumerate(label_names):
    print(f"{label} ({i}): {test_counts.get(i, 0)}")

# === Compute class weights for imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2, 3]), y=y)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# === Create graph data object
y_tensor = torch.tensor(y, dtype=torch.long)
data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y_tensor)

# === Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
class_weights = class_weights.to(device)

# === GCN Model ===
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

# === GraphSAGE (GNN) Model ===
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# === Train function ===
def train_model(model, data, train_idx, optimizer, criterion, epochs=100):
    model.train()
    for epoch in tqdm(range(epochs), desc="Training"):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()
    return model

# === Evaluate function ===
def evaluate_model(model, data, test_idx, model_name="Model"):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    y_true = data.y[test_idx].cpu().numpy()
    y_pred = pred[test_idx].cpu().numpy()

    print(f"\n[{model_name} CLASSIFICATION REPORT]")
    print(classification_report(y_true, y_pred, target_names=label_names))

    # Save predictions
    results_df = pd.DataFrame({
        "sampleID": labels_df.index[test_idx],
        "true_label": label_names[y_true],
        "predicted_label": label_names[y_pred]
    })
    results_df.to_csv(f"{model_name.lower()}_results.csv", index=False)
    print(f"📁 Saved to: {model_name.lower()}_results.csv")

# === Train & Evaluate GCN ===
print("\n🚀 Training GCN...")
model_gcn = GCN(in_channels=data.num_features, hidden_channels=64, out_channels=len(label_names)).to(device)
optimizer_gcn = torch.optim.Adam(model_gcn.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
model_gcn = train_model(model_gcn, data, train_idx, optimizer_gcn, criterion, epochs=100)
evaluate_model(model_gcn, data, test_idx, model_name="GCN")

# === Train & Evaluate GraphSAGE ===
print("\n🚀 Training GraphSAGE...")
model_sage = GraphSAGE(in_channels=data.num_features, hidden_channels=64, out_channels=len(label_names)).to(device)
optimizer_sage = torch.optim.Adam(model_sage.parameters(), lr=0.01)
model_sage = train_model(model_sage, data, train_idx, optimizer_sage, criterion, epochs=100)
evaluate_model(model_sage, data, test_idx, model_name="GNN")
