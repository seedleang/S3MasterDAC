import numpy as np
import torch
import torch.nn as nn

from utils import PositionalEncoding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

# //////////////////////////////////////////////////////////////////////////////////////////////// <attention types> ////

class AttentionLayer(nn.Module):
    # Une couche d'attention à réutiliser partout comme nn.Module
    def __init__(self,dimbed,dimlat):
        super(AttentionLayer,self).__init__()
        self.dimbed = dimbed
        self.soft = nn.Softmax(dim=1)
        self.q = nn.Linear(dimbed,dimlat,bias=False) 
        self.v = nn.Linear(dimbed,dimlat,bias=False)
        self.k = nn.Linear(dimbed,dimlat,bias=False) 
        self.gtheta = nn.Linear(dimlat,dimbed) 
        self.norm = nn.LayerNorm(BATCH_SIZE,dimbed)
        
    def forward(self,x):
        #print('x',x.shape)
        xtilde = self.norm(x)
        #print('xtilde',xtilde.shape)
        q = self.q(xtilde.transpose(1,2))
        v = self.v(xtilde.transpose(1,2))
        k = self.k(xtilde.transpose(1,2))
        #print('q ',q.shape,' v ',v.shape,' k ',k.shape)
        e = torch.bmm(q.transpose(0,1), k.permute(1,2,0))
        #print('e ',e.shape)
        a = self.soft(1/np.sqrt(self.dimbed)*e) 
        #print('a ',a.shape)
        ftheta = torch.bmm(a,v.transpose(0,1)) 
        #print('f ',ftheta.shape)
        return self.gtheta(ftheta).permute(1,2,0)
        
        
class SelfAttention(nn.Module):
    # Cas de base.
    def __init__(self,dimbed,dimlat,dimout,nL):
        super(SelfAttention,self).__init__()
        l = []
        # Concaténation de nL sous-couches d'attention
        for i in range(nL):
            l.append(AttentionLayer(dimbed,dimlat))
            l.append(nn.ReLU())
            l.append(nn.Dropout())
        self.attention = nn.Sequential(*l) 
        self.classifier = nn.Linear(dimbed,dimout)
        self.sgma = nn.Sigmoid()
    
    def forward(self,x):
        focus = self.attention(x.transpose(1,2))
        # Classification sur la moyenne des attentions.
        focus = torch.mean(focus,axis=0)
        return self.sgma(self.classifier(focus.transpose(0,1)))
        

class SelfAttentionPE(nn.Module):
    # Ajout des encodages positionnels. Voir Notebook pour une analyse.
    def __init__(self,dimbed,dimlat,dimout,nL):
        super(SelfAttentionPE,self).__init__()
        l = []
        for i in range(nL):
            l.append(AttentionLayer(dimbed,dimlat))
            l.append(nn.ReLU())
            l.append(nn.Dropout())
        self.attention = nn.Sequential(*l) 
        self.classifier = nn.Linear(dimbed,dimout)
        self.sgma = nn.Sigmoid()
        self.pe = PositionalEncoding(d_model=dimbed,max_len=3000).to(device)
        
    def forward(self,x):         
        x = self.pe(x.transpose(0,1)).transpose(0,1)
        focus = self.attention(x.transpose(1,2))
        # Classification sur une moyenne.
        focus = torch.mean(focus,axis=0)
        return self.sgma(self.classifier(focus.transpose(0,1)))
    

class SelfAttentionCLS(nn.Module):
    # Ajout d'un token artificiel qui va subir toutes les transformations...
    def __init__(self,dimbed,dimlat,dimout,nL):
        super(SelfAttentionCLS,self).__init__()
        l = []
        for i in range(nL):
            l.append(AttentionLayer(dimbed,dimlat))
            l.append(nn.ReLU())
            l.append(nn.Dropout())
        self.attention = nn.Sequential(*l) 
        self.classifier = nn.Linear(dimbed,dimout)
        self.sgma = nn.Sigmoid()
        self.cls = torch.nn.Parameter(torch.ones(dimbed))
    
    def forward(self,x):
        x = torch.cat((self.cls.unsqueeze(0).repeat((1,BATCH_SIZE,1)),x))
        focus = self.attention(x.transpose(1,2))
        # Classification sur ce seul token !
        cls_learned = focus[0,:,:]
        return self.sgma(self.classifier(cls_learned.transpose(0,1)))


class SelfAttentionPE_CLS(nn.Module):
    # Est-il utile de combiner les deux ?
    def __init__(self,dimbed,dimlat,dimout,nL):
        super(SelfAttentionPE_CLS,self).__init__()
        l = []
        for i in range(nL):
            l.append(AttentionLayer(dimbed,dimlat))
            l.append(nn.ReLU())
            l.append(nn.Dropout())
        self.attention = nn.Sequential(*l) 
        self.classifier = nn.Linear(dimbed,dimout)
        self.sgma = nn.Sigmoid()
        self.cls = torch.nn.Parameter(torch.ones(dimbed))
        self.pe = PositionalEncoding(d_model=dimbed,max_len=3000).to(device)
        
    def forward(self,x): 
        x = torch.cat((self.cls.unsqueeze(0).repeat((1,BATCH_SIZE,1)),x))
        # Encodages positionnels 
        x = self.pe(x.transpose(0,1)).transpose(0,1)
        focus = self.attention(x.transpose(1,2))
        # ET classification sur un seul token
        cls_learned = focus[0,:,:]
        return self.sgma(self.classifier(cls_learned.transpose(0,1)))
    
# /////////////////////////////////////////////////////////////////////////////////////////////// </attention types> ////