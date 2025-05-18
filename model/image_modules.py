import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init
import torch.nn.functional as F
import open_clip
from transformers import AutoImageProcessor, ViTModel

class ImageDecoder(nn.Module):
    def __init__(
        self, out_dim, input_dim, 
        policy='mean', gcn=False, num_sample=10
    ):
        super().__init__()
        self.gcn = gcn
        self.policy=policy
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.num_sample = num_sample
        self.weight = nn.Parameter(torch.FloatTensor(
            self.out_dim, 
            self.input_dim if self.gcn else 2*self.input_dim
        ))
        init.xavier_uniform_(self.weight)

    def forward(self, x, Adj):
        neigh_feats = self.aggregate(x, Adj)
        if not self.gcn:
            combined = torch.cat([x, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.T)).T
        combined = F.normalize(combined,2,1) #计算每一行的L2范数
        return combined
    def aggregate(self,x, Adj):
        adj=Variable(Adj).to(Adj.device)
        if not self.gcn:
            n=len(adj)
            adj = adj-torch.eye(n).to(adj.device)
        if self.policy=='mean':
            num_neigh = adj.sum(1, keepdim=True)
            mask = adj.div(num_neigh)
            to_feats = mask.mm(x)
        elif self.policy=='max':
            indexs = [i.nonzero() for i in adj==1]
            to_feats = []
            for feat in [x[i.squeeze()] for i in indexs]:
                if len(feat.size()) == 1:
                    to_feats.append(feat.view(1, -1))
                else:
                    to_feats.append(torch.max(feat,0)[0].view(1, -1))
            to_feats = torch.cat(to_feats, 0)
        return to_feats
       

class ImageEncoder_QuiltNet(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='ViT-B-32-quickgelu'):
        super().__init__()
        self.model_name=model_name
        if model_name == 'ViT-B-32-quickgelu':
            self.model, _, _ = open_clip.create_model_and_transforms(model_name, 
                                                                pretrained='./model/QuiltNet-B-32/open_clip_pytorch_model.bin')

    def forward(self, x):
        if self.model_name == 'ViT-B-32-quickgelu':
            return self.model(x)



class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x