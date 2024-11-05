# %%
import dgl
import dgl.data
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.nn import SAGEConv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# 載入 Cora 資料集
dataset = dgl.data.CoraGraphDataset()
graph = dataset[0]

# 打印資料集信息
print("節點數量:", graph.num_nodes())
print("邊數量:", graph.num_edges())

# 從圖中抽取所有的正樣本
u, v = graph.edges()

# 將一部分邊移除作為訓練集，並保留剩下的邊作為評估集
u_train, u_test, v_train, v_test = train_test_split(
    u, v, test_size=0.3, random_state=42
)

# 將部分邊從圖中刪除以構建訓練圖
train_graph = dgl.remove_edges(graph, torch.arange(len(u_test)))

# 構建負樣本
neg_u = np.random.randint(0, graph.num_nodes(), len(u_test))
neg_v = np.random.randint(0, graph.num_nodes(), len(u_test))

# %%


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super(GraphSAGE, self).__init__()
        # 兩層 GraphSAGE
        self.sage1 = SAGEConv(in_feats, hidden_feats, aggregator_type="mean")
        self.sage2 = SAGEConv(hidden_feats, hidden_feats, aggregator_type="mean")

    def forward(self, g, x):
        # 第一層 GraphSAGE + ReLU
        h = self.sage1(g, x)
        h = torch.relu(h)
        # 第二層 GraphSAGE
        h = self.sage2(g, h)
        return h


class LinkPredictor(nn.Module):
    def __init__(self, hidden_feats):
        super(LinkPredictor, self).__init__()
        self.W = nn.Linear(hidden_feats, hidden_feats)

    def forward(self, h, u, v):
        # 計算節點 u 和 v 的嵌入向量
        h_u = h[u]
        h_v = h[v]
        # 餘弦相似度作為邊的分數
        score = torch.sum(self.W(h_u) * h_v, dim=1)
        return score


epochs = 200
in_feats = graph.ndata["feat"].shape[1]
hidden_feats = 16
model = GraphSAGE(in_feats, hidden_feats)
predictor = LinkPredictor(hidden_feats)
graph = graph


optimizer = optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=0.01)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(epochs):
    model.train()

    # 前向傳播，得到所有節點的嵌入
    h = model(train_graph, train_graph.ndata["feat"])

    # 正樣本分數
    pos_score = predictor(h, u_train, v_train)
    # 負樣本分數
    neg_score = predictor(h, neg_u, neg_v)

    # 標籤：正樣本為 1，負樣本為 0
    pos_label = torch.ones_like(pos_score)
    neg_label = torch.zeros_like(neg_score)

    # 計算損失
    loss = loss_fn(pos_score, pos_label) + loss_fn(neg_score, neg_label)

    # 反向傳播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 將分數和標籤合併
    with torch.no_grad():
        scores = torch.cat([pos_score, neg_score]).cpu().numpy()
        labels = torch.cat([pos_label, neg_label]).cpu().numpy()
        # 計算 Accuracy
        auc = roc_auc_score(labels, scores)

    print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {auc:.4f}")


# %%
# 評估模型

model.eval()
with torch.no_grad():
    h = model(graph, graph.ndata["feat"])
    pos_score = predictor(h, u_test, v_test)
    neg_score = predictor(h, neg_u, neg_v)

    # 將分數和標籤合併
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = (
        torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        .cpu()
        .numpy()
    )

    # 計算 AUC
    auc = roc_auc_score(labels, scores)
    print(f"AUC: {auc:.4f}")


# %%
