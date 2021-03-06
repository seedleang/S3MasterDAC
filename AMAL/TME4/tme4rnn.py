import torch
import torch.nn as nn

# ////////////////////////////////////////////////////////////////////////////////////// <checkpointing on the model> ////

class State(object) :
    def __init__(self, model, optim, linconv=None):
        self.model = model
        self.optim = optim
        self.linconv = linconv # generation task
        self.epoch, self.iteration = 0, 0
        
# ///////////////////////////////////////////////////////////////////////////////////// </checkpointing on the model> ////

# //////////////////////////////////////////////////////////////////////////////////////// <recurrent neural network> ////

class RNN(nn.Module):
    def __init__(self, diminput, dimlatent, dimoutput):
        super(RNN, self).__init__()
        self.diminput = diminput
        self.dimlatent = dimlatent
        self.dimoutput = dimoutput
        # Three types of layers
        # (at the entrance, inside the network, on exiting it)
        self.Lininp = nn.Linear(diminput,dimlatent)
        self.Linlat = nn.Linear(dimlatent,dimlatent)
        self.Linout = nn.Linear(dimlatent,dimoutput)
        self.tanh = nn.Tanh()
        
    def oneStep(self,seqsAtT,h):    
        # :seqsAtT: dim = sbatch×diminput // got this from a batch of sequences sampled at time t
        # :h:       dim = sbatch×dimlatent // this is the previous hidden state predicted and kept in memory
        return self.tanh(self.Lininp(seqsAtT.view(-1,self.diminput))+self.Linlat(h))
       
    def forward(self,seqs,h):
        # seqs' size is lenseq×sbatch×diminput. Starting from arbitrary state h, 
        # return a sequence of estimated hidden states of size lenseq×sbatch×dimlatent on input=seqs
        allhidden = [h]
        for timestamp in seqs:
            h = self.oneStep(timestamp,allhidden[-1])
            allhidden += [h]
        return torch.stack(allhidden,axis=0)    
    
    def decode(self,h):
        # You can output this anytime (depends on the chosen architecture // many-to-many, many-to-one, (...)) 
        return self.Linout(h)
    
# /////////////////////////////////////////////////////////////////////////////////////// </recurrent neural network> ////