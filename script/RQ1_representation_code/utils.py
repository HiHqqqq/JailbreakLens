import torch
import torch.nn as nn
class LRProbe(torch.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 1, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def pred(self, x):
        p=self(x)
        return p.round(),p
    
    def from_data(acts, labels, lr=0.001, weight_decay=0.1, epochs=1000, device='cpu'):
        acts, labels = acts.to(device), labels.to(device)
        probe = LRProbe(acts.shape[-1]).to(device)
        
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.BCELoss()(probe(acts), labels)
            loss.backward()
            opt.step()
        
        return probe
    def direction(self):
        return self.net[0].weight.data[0]

import torch
import torch.nn as nn

class MeanCenterProbe:
    def __init__(self,device):
        self.center_class_0 = None
        self.center_class_1 = None
        self.device=device

    def get_direction(self, X, y):
        """Compute the centers of each class"""
        X=X.cpu()
        y=y.cpu()
        self.center_class_0 = X[y == 0].mean(dim=0)
        self.center_class_1 = X[y == 1].mean(dim=0)
        self.direction_vector = (self.center_class_1 - self.center_class_0)
        self.direction_vector /= torch.norm(self.direction_vector)  # Normalize


    def pred(self, X):
        X=X.cpu()
        # Compute the distance to each class center
        dist_to_class_0 = torch.norm(X - self.center_class_0, dim=1)
        dist_to_class_1 = torch.norm(X - self.center_class_1, dim=1)

        projections = ((X - self.center_class_0) @ self.direction_vector) / (self.direction_vector @ self.direction_vector)
        projected_points = self.center_class_0 + projections[:, None] * self.direction_vector
        
        # Calculate distances from the projected points to the class centers
        dist_to_class_0 = torch.norm(projected_points - self.center_class_0, dim=1)
        dist_to_class_1 = torch.norm(projected_points - self.center_class_1, dim=1)
        
        # Convert distances to probabilities
        sum_distances = dist_to_class_0 + dist_to_class_1
        proba_class_0 = 1 - (dist_to_class_0 / sum_distances)
        proba_class_1 = 1 - (dist_to_class_1 / sum_distances)
        
        # Predict class labels based on which distance is smaller
        predictions = torch.where(dist_to_class_0 < dist_to_class_1, 0, 1)
        return predictions.to(self.device),proba_class_1.to(self.device)#, proba_class_1

    def direction(self):
        return self.direction_vector.data
   

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
class PCAProbe:
    def __init__(self,device='cuda', n_components=1):
        self.n_components = n_components
        self.device=device
        self.direction_vector = None
        self.threshold = 0
        self.class_labels = None

    def get_direction(self, X, y):
        X=X.cpu()
        y=y.cpu()
        """Use PCA to find the direction vector and determine the threshold"""
        X_np = X.numpy()  # Convert torch tensor to numpy array
        pca = PCA(n_components=self.n_components)
        pca.fit(X_np)
        self.direction_vector = torch.tensor(pca.components_[0])  # Convert direction back to torch tensor
        
        # Project training data onto the direction vector
        projections = X.matmul(self.direction_vector)
        
        # Calculate median projections for each class
        median_projection_class_0 = projections[y == 0].median().item()
        median_projection_class_1 = projections[y == 1].median().item()
        self.median_projection_class_0 =median_projection_class_0 
        self.median_projection_class_1 =median_projection_class_1 
        
        # Determine the threshold for classification
        self.threshold = (median_projection_class_0 + median_projection_class_1) / 2
        
        # Determine which class is on the positive side of the threshold
        if median_projection_class_0 > median_projection_class_1:
            self.class_labels = (0, 1)
        else:
            self.class_labels = (1, 0)

    def pred(self, X):
        X=X.cpu()
        """Predict binary labels using the direction vector and threshold"""
        if self.direction is None:
            raise ValueError("Direction vector not set. Call get_direction first.")
        
        # Project X onto the direction vector
        projections = X.matmul(self.direction_vector)
        
        # Predict class labels based on the threshold
        predictions = torch.where(projections > self.threshold, self.class_labels[0], self.class_labels[1])

        denominator = self.median_projection_class_1 - self.median_projection_class_0
        if denominator == 0:
            logits = torch.zeros_like(projections)
        else:
            logits = (projections - self.median_projection_class_0) / denominator
            logits = logits.clamp(0, 1)  # 限制 logits 在 0-1 之间
        
        return predictions.to(self.device),logits.to(self.device)

    def direction(self):
        return self.direction_vector.data


# class ClusterMeanProbe(torch.nn.Module):
#     def __init__(self, d_in):
#         """
#         ClusterMean探针模型初始化
#         Args:
#             d_in: 输入表征的维度
#         """
#         super().__init__()
#         self.centroid_A = torch.zeros(d_in)  # 初始化 group A 的中心点
#         self.centroid_B = torch.zeros(d_in)  # 初始化 group B 的中心点
#         self.direction_vector=torch.zeros(d_in) 

#     def forward(self, x):
#         dist_to_A = torch.norm(x - self.centroid_A.to(x.device), dim=1)
#         # 计算输入 x 到 group B 中心点的距离
#         dist_to_B = torch.norm(x - self.centroid_B.to(x.device), dim=1)

#         projections = ((x - self.centroid_A.to(x.device)) @ self.direction_vector) / (self.direction_vector @ self.direction_vector)
#         projected_points = self.centroid_A.to(x.device) + projections[:, None] * self.direction_vector
        
#         # Calculate distances from the projected points to the class centers
#         dist_to_A = torch.norm(projected_points - self.centroid_A.to(x.device), dim=1)
#         dist_to_B = torch.norm(projected_points - self.centroid_B.to(x.device), dim=1)
        
#         # Convert distances to probabilities
#         sum_distances = dist_to_A + dist_to_B
#         proba_A = 1 - (dist_to_A / sum_distances)
#         proba_B = 1 - (dist_to_B / sum_distances)
        
#         # Predict class labels based on which distance is smaller
#         predictions = torch.where(dist_to_A < dist_to_B, 0, 1)
#         return predictions,proba_A  #, proba_B
        
#         # #### old
#         # dist_to_A = torch.norm(x - self.centroid_A.to(x.device), dim=1)
#         # # 计算输入 x 到 group B 中心点的距离
#         # dist_to_B = torch.norm(x - self.centroid_B.to(x.device), dim=1)
#         # logits = dist_to_A / (dist_to_A + dist_to_B + 1e-9)

#         # # 比较距离，距离更近的归为哪个 group
#         # return (dist_to_B < dist_to_A).float(),logits
#     def pred(self, x):
     
#         preds,logits = self(x)
#         return preds,logits
#         # if logit:
#         #     return preds, None  # ClusterMean 不需要logit值，返回 None 作为占位符
#         # else:
#         #     return preds

#     @classmethod
#     def from_data(cls, acts, labels, device='cpu'):
#         acts, labels = acts.to(device), labels.to(device)
#         probe = cls(acts.shape[-1]).to(device)  # 创建探针模型实例

#         # 计算 Group A 和 Group B 的中心点
#         centroid_A = torch.mean(acts[labels == 0], dim=0)
#         centroid_B = torch.mean(acts[labels == 1], dim=0)

#         # 更新探针模型中的中心点
#         probe.centroid_A = centroid_A
#         probe.centroid_B = centroid_B
#         direction_vector = (centroid_A - centroid_B)
#         probe.direction_vector /= torch.norm(direction_vector).to('cpu')
#         probe.direction_vector=probe.direction_vector.to(device)

#         return probe

#     def direction(self):
#         return self.direction_vector.data

# # 示例用法
# if __name__ == "__main__":
#     # 创建随机数据，假设每个输入表征的维度为 4096
#     representations_A = torch.randn(100, 4096)  # 100个属于group A的表征
#     representations_B = torch.randn(120, 4096)  # 120个属于group B的表征
#     input_data = torch.randn(10, 4096)          # 10个要进行分类的表征

#     # 标签 0 表示 group A，1 表示 group B
#     acts = torch.cat([representations_A, representations_B], dim=0)
#     labels = torch.cat([torch.zeros(100), torch.ones(120)], dim=0)

#     # 创建 ClusterMean 探针模型，并从数据拟合
#     cluster_probe = ClusterMeanProbe.from_data(acts, labels)

#     # 计算方向向量
#     direction_vector = cluster_probe.direction()
#     print("Direction vector from Group A to Group B:", direction_vector)

#     # 使用探针模型进行分类
#     output = cluster_probe(input_data)
#     print("Classification result (0: Group A, 1: Group B):", output)

from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch,random
import numpy as np
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
