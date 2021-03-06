import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# //////////////////////////////////////////////////////////////////////////////////////////////// <attention types> ////

class NaiveAttention(nn.Module):
    # Attention naïve, représentation des phrases par une valeur moyenne
    def __init__(self,dimbed, dimout):
        super(NaiveAttention,self).__init__()
        self.classifier = nn.Linear(dimbed,dimout)
        
    def forward(self,x):
        # x[0] = batch au padding masqué, x[1] = masque 
        x_avg = torch.sum(x[0],axis=0) / torch.sum(x[1],axis=0)
        return self.classifier(x_avg)
    
    
class SimpleAttention(nn.Module):
    # Apprentissage d'une query globale
    def __init__(self,dimbed,dimout):
        super(SimpleAttention,self).__init__()
        self.soft = nn.Softmax(dim=0)
        # La query q est un vecteur unique, partagé,
        # qui servira à calculer les scores d'attention 
        self.q = nn.Linear(dimbed,1,bias=False)
        self.classifier = nn.Linear(dimbed,dimout)
        
    def forward(self,x):
        e = self.q(x[0]) 
        # Remplacement des 0 par -infini, 
        # pour que le padding et les OOV aient un impact nul post-softmax
        e = torch.where(e==0,torch.zeros_like(e)-np.inf,e)
        a = self.soft(e)
        # Somme des valeurs des mots pondérées par leurs scores d'attention 
        c = torch.sum(a*x[0],axis=0) 
        # Retourner aussi les attentions permettra de visualiser l'entropie.
        return self.classifier(c), a 
    
    
class FurtherAttention(nn.Module):
    def __init__(self,dimbed,dimout,dive=True):
        super(FurtherAttention,self).__init__()
        self.soft = nn.Softmax(dim=0)
        self.qlearner = nn.Linear(dimbed,dimbed) 
        self.vlearner = nn.Linear(dimbed,dimbed) 
        self.classifier = nn.Linear(dimbed,dimout)
        # Paramétrage du plongement
        self.dive = dive
        self.activation = nn.ReLU() # inutilisée ? voir l'effet ?
        
    def forward(self,x):
        # Apprentissage de la query sur les phrases masquées
        x_avg = torch.sum(x[0],axis=0) / torch.sum(x[1],axis=0)
        q = self.qlearner(x_avg)
        if self.dive: # Faut-il apprendre des plongements de valeurs ?
            v = self.vlearner(x[0])
        # Produit entre la query et les clés
        e = torch.bmm(x[0].transpose(0,1), q.unsqueeze(2))
        e = torch.where(e==0,torch.zeros_like(e)-np.inf,e)
        a = self.soft(e.transpose(0,1)) 
        
        if self.dive: c = torch.sum(a*v,axis=0) 
        else : c = torch.sum(a*x[0],axis=0) 
        return self.activation(self.classifier(c)), a
    
    
class LSTMAttention(nn.Module):
    def __init__(self,dimbed,dimout):
        super(LSTMAttention,self).__init__()
        self.soft = nn.Softmax(dim=0)
        self.qlearner = nn.Linear(dimbed,dimbed) 
        self.lstm = nn.LSTM(dimbed,dimbed)
        self.classifier = nn.Linear(dimbed,dimout)
        
    def forward(self,x):
        x_avg = torch.sum(x[0],axis=0) / torch.sum(x[1],axis=0)
        q = self.qlearner(x_avg)
        # Apprentissage des plongements de valeurs grâce à une LSTM
        ht, ot = self.lstm(x[0])
        e = torch.bmm(x[0].transpose(0,1), q.unsqueeze(2))
        e = torch.where(e==0,torch.zeros_like(e)-np.inf,e)
        a = self.soft(e.transpose(0,1)) 
        c = torch.sum(a*ht,axis=0)
        return self.classifier(c), a
    
class BILSTMAttention(nn.Module):
    def __init__(self,dimbed,dimout):
        super(BILSTMAttention,self).__init__()
        self.soft = nn.Softmax(dim=0)
        self.qlearner = nn.Linear(dimbed,dimbed) 
        self.lstm = nn.LSTM(dimbed,dimbed,bidirectional=True)
        self.classifier = nn.Linear(dimbed,dimout)
        
    def forward(self,x):
        x_avg = torch.sum(x[0],axis=0) / torch.sum(x[1],axis=0)
        q = self.qlearner(x_avg)
        # Apprentissage des plongements de valeurs grâce à une LSTM
        ht, ot = self.lstm(x[0])
        e = torch.bmm(x[0].transpose(0,1), q.unsqueeze(2))
        e = torch.where(e==0,torch.zeros_like(e)-np.inf,e)
        a = self.soft(e.transpose(0,1)) 
        c = torch.sum(a*ht,axis=0)
        return self.classifier(c), a
    
# /////////////////////////////////////////////////////////////////////////////////////////////// </attention types> ////