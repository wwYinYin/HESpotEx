import torch
from torch import nn
import torch.nn.functional as F

from model.image_modules import ImageDecoder, ProjectionHead, ImageEncoder_QuiltNet

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

        
class Image_Model(nn.Module):
    def __init__(
        self,
        out_embedding,
        latent_dim=512,
        method='global',
        stage='train'
    ):
        super().__init__()
        self.method=method
        self.image_encoder = ImageEncoder_QuiltNet(model_name='ViT-B-32-quickgelu')
        self.image_embedding=512
        self.stage=stage

        self.image_projection = ProjectionHead(embedding_dim=self.image_embedding,projection_dim=latent_dim) #aka the input dim, 2048 for resnet50
        self.image_projection2 = ProjectionHead(embedding_dim=latent_dim,projection_dim=out_embedding)

        self.image_decoder=nn.ModuleList([ImageDecoder(latent_dim,latent_dim) for _ in range(4)])
        self.jknet=nn.Sequential(
            nn.LSTM(latent_dim,out_embedding,2),
            SelectItem(0),
        )
        self.latent_head = nn.Sequential(
            nn.Linear(self.image_embedding, 1024),
            # nn.Linear(72, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, latent_dim)
        )
        self.gene_head = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, out_embedding)
        )

    def forward(self, batch):
        if self.stage == 'inference':
            image_features = self.image_encoder(batch["image"])[0]
        
            adj = batch["adj"]
            nuclei= batch["nuclei_number"]
            image_embeddings = self.latent_head(image_features)
            linear_out = self.gene_head(image_embeddings)

            jk=[]
            for layer in self.image_decoder:
                out=layer(image_embeddings,adj)
                jk.append(out.unsqueeze(0))
                # image_embeddings=layer(image_embeddings,adj)
                # jk.append(image_embeddings.unsqueeze(0))
            out=torch.cat(jk,0)
            out=self.jknet(out).mean(0)
            # out = linear_out
            out = out+linear_out
            out=out*nuclei

            return out, image_features
        else:
            # Getting Image and spot Features
            if self.method == "phikon":
                image_features = batch["image"].squeeze(0)
                # image_features = self.image_encoder(batch["image"].squeeze(0))
            else:
                image_features = self.image_encoder(batch["image"].squeeze(0))[0]
            # image_features = batch["image"].squeeze(0)  
              
            spot_features = batch["expression"].squeeze(0)
            spot_latent_features = batch["latent"].squeeze(0)
            adj = batch["adj"].squeeze(0)
            nuclei= batch["nuclei_number"].squeeze(0)
            # print(image_features.shape)
            # Getting Image and Spot Embeddings (with same dimension) 
            # image_embeddings = self.image_projection(image_features)
            # image_embeddings=self.relu(image_embeddings)
            # linear_out = self.image_projection2(image_embeddings)
            image_embeddings = self.latent_head(image_features)
            
            linear_out = self.gene_head(image_embeddings)

            jk=[]
            for layer in self.image_decoder:
                out=layer(image_embeddings,adj)
                jk.append(out.unsqueeze(0))
                # image_embeddings=layer(image_embeddings,adj)
                # jk.append(image_embeddings.unsqueeze(0))
            out=torch.cat(jk,0)
            out=self.jknet(out).mean(0)
            # out = linear_out
            out = out+linear_out

            out=out*nuclei
            # out = F.softmax(out, dim=-1)
            # Calculating the Loss
            # logits = (spot_latent_features @ image_embeddings.T) / self.temperature
            # images_similarity = image_embeddings @ image_embeddings.T
            # spots_similarity = spot_latent_features @ spot_latent_features.T
            # targets = F.softmax(
            #     (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
            # )
            # spots_loss = cross_entropy(logits, targets, reduction='none')
            # images_loss = cross_entropy(logits.T, targets.T, reduction='none')
            # latent_simloss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
            # latent_simloss = latent_simloss.mean()
            latent_simloss=1 - PCC(spot_latent_features, image_embeddings).nanmean()
            # latent_simloss=F.mse_loss(spot_latent_features, image_embeddings)

            PCC_loss = 1 - PCC(spot_features, out, axis=1).nanmean()
            # spot_features = F.softmax(spot_features, dim=-1)
        
            mse_loss = F.mse_loss(out, spot_features)
            # mse_loss = 1 - PCC(spot_features, out).nanmean()

            # predict_counts = out.mean(1).unsqueeze(0)
            # gt_counts = spot_features.mean(1).unsqueeze(0)
            # counts_loss = F.mse_loss(predict_counts, gt_counts)
            # final_loss=latent_simloss + mse_loss

            return latent_simloss, mse_loss, PCC_loss, out

class Image_Model_ST1K(nn.Module):
    def __init__(
        self,
        out_embedding,
        latent_dim=512,
        method='global',
        stage='train'
    ):
        super().__init__()
        self.method=method
        self.image_encoder = ImageEncoder_QuiltNet(model_name='ViT-B-32-quickgelu')
        self.image_embedding=512
        self.stage=stage

        self.image_projection = ProjectionHead(embedding_dim=self.image_embedding,projection_dim=latent_dim) #aka the input dim, 2048 for resnet50
        self.image_projection2 = ProjectionHead(embedding_dim=latent_dim,projection_dim=out_embedding)

        self.image_decoder=nn.ModuleList([ImageDecoder(latent_dim,latent_dim) for _ in range(4)])
        self.jknet=nn.Sequential(
            nn.LSTM(latent_dim,out_embedding,2),
            SelectItem(0),
        )
        self.latent_head = nn.Sequential(
            nn.Linear(self.image_embedding, 1024),
            # nn.Linear(72, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, latent_dim)
        )
        self.gene_head = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, out_embedding)
        )

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"].squeeze(0))[0] 
            
        spot_features = batch["expression"].squeeze(0)
        spot_latent_features = batch["latent"].squeeze(0)
        adj = batch["adj"].squeeze(0)
        image_embeddings = self.latent_head(image_features)
        
        linear_out = self.gene_head(image_embeddings)

        jk=[]
        for layer in self.image_decoder:
            out=layer(image_embeddings,adj)
            jk.append(out.unsqueeze(0))
            # image_embeddings=layer(image_embeddings,adj)
            # jk.append(image_embeddings.unsqueeze(0))
        out=torch.cat(jk,0)
        out=self.jknet(out).mean(0)
        # out = linear_out
        out = out+linear_out
        latent_simloss=1 - PCC(spot_latent_features, image_embeddings).nanmean()

        PCC_loss = 1 - PCC(spot_features, out, axis=1).nanmean()
        # spot_features = F.softmax(spot_features, dim=-1)
    
        mse_loss = F.mse_loss(out, spot_features)

        return latent_simloss, mse_loss, PCC_loss, out

def pcc_distances(v1, v2):
    if v1.shape[1] != v2.shape[1]:
        raise ValueError("The two matrices v1 and v2 must have equal dimensions; two slice data must have the same genes")

    n = v1.shape[1]
    sums = torch.outer(v1.sum(1), v2.sum(1))
    stds = torch.outer(v1.std(1), v2.std(1))
    correlation = (torch.matmul(v1, v2.T) - sums / n) / stds / n
    distances = 1 - correlation
    return distances

def kl_divergence_backend(X, Y):
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    X = X / torch.sum(X, axis=1, keepdims=True)
    Y = Y / torch.sum(Y, axis=1, keepdims=True)
    log_X = torch.log(X)
    log_Y = torch.log(Y)
    X_log_X = torch.einsum('ij,ij->i', X, log_X)
    X_log_X = torch.reshape(X_log_X, (1, X_log_X.shape[0]))
    D = X_log_X.T - torch.matmul(X, log_Y.T)
    return D

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
def PCC(targets,preds,axis=0):
    func = Pearsonr

    if axis == 1:
        c_tensor = torch.zeros(targets.shape[1])
        for i in range(targets.shape[1]):
            r = func(targets[:,i], preds[:,i])
            c_tensor[i]=r
    else:
        c_tensor = torch.zeros(targets.shape[0])
        for i in range(targets.shape[0]):
            r = func(targets[i,:], preds[i,:])
            c_tensor[i]=r
    return c_tensor

def Pearsonr(im1,im2):
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = torch.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = torch.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    pearsonr=sigma12/(sigma1*sigma2)
    #print(pearsonr)
    pearsonr=torch.where(torch.isnan(pearsonr), torch.full_like(pearsonr, 0), pearsonr)
    return pearsonr
