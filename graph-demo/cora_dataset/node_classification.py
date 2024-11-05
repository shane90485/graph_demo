# %%
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.nn import GraphConv

# 載入 Cora 資料集
dataset = dgl.data.CoraGraphDataset()
graph = dataset[0]
print("節點特徵維度:", graph.ndata["feat"].shape)
print("類別數:", dataset.num_classes)


# %%
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super(GCN, self).__init__()
        # 第一層 Graph Convolution
        self.conv1 = GraphConv(in_feats, hidden_feats)
        # 第二層 Graph Convolution
        self.conv2 = GraphConv(hidden_feats, num_classes)

    def forward(self, g, inputs):
        # 第一層卷積 + ReLU 激活
        x = self.conv1(g, inputs)
        x = torch.relu(x)
        # 第二層卷積（輸出分類 output）
        x = self.conv2(g, x)
        return x


# 選擇 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 GCN 模型
epochs = 200
in_feats = graph.ndata["feat"].shape[1]
hidden_feats = 16
num_classes = dataset.num_classes
model = GCN(in_feats, hidden_feats, num_classes).to(device)
graph = graph.to(device)

features = graph.ndata["feat"]
labels = graph.ndata["label"]
train_mask = graph.ndata["train_mask"]  # 訓練集遮罩
val_mask = graph.ndata["val_mask"]  # 評估集遮罩
test_mask = graph.ndata["test_mask"]  # 測試集遮罩


# %%
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()

    # 前向傳遞
    output = model(graph, features)

    # 計算訓練損失
    train_loss = loss_fn(output[train_mask], labels[train_mask])
    val_loss = loss_fn(output[val_mask], labels[val_mask])

    # 反向傳播
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # 計算訓練和測試的準確率
    train_acc = (output[train_mask].argmax(dim=1) == labels[train_mask]).float().mean()
    val_acc = (output[val_mask].argmax(dim=1) == labels[val_mask]).float().mean()

    print(
        f"Epoch {epoch}/{epochs}, Loss: {train_loss.item():.4f}, "
        f"Train Acc: {train_acc.item():.4f}, "
        f"Val Loss: {val_loss.item():.4f}, "
        f"Val Acc: {val_acc.item():.4f}"
    )

# %%
# 評估模型
model.eval()
with torch.no_grad():
    output = model(graph, features)
    test_acc = (output[test_mask].argmax(dim=1) == labels[test_mask]).float().mean()
    print(f"Test Accuracy: {test_acc.item():.4f}")
