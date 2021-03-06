import torch
import torch.nn as nn

# ///////////////////////////////////////////////////////////////////////////////////// <checkpointing on the models> ////

class State(object) :
    def __init__(self, model, optim, linconv=None):
        self.model = model
        self.optim = optim
        self.linconv = linconv # for the generation task
        self.epoch, self.iteration = 0, 0
        
# //////////////////////////////////////////////////////////////////////////////////// </checkpointing on the models> ////

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

# //////////////////////////////////////////////////////////////////////////////////// <long short-term memory cells> ////

class LSTM(nn.Module):
    def __init__(self, diminput, dimlatent, dimoutput):
        super(LSTM, self).__init__()
        self.diminput = diminput
        self.dimlatent = dimlatent
        self.dimoutput = dimoutput
        self.tanh = nn.Tanh()
        self.sgma = nn.Sigmoid()
        # The gate
        self.Gateinput = torch.nn.Linear(diminput+dimlatent,dimlatent)
        self.Gateforget = torch.nn.Linear(diminput+dimlatent,dimlatent)
        self.Gateinner = torch.nn.Linear(diminput+dimlatent,dimlatent)
        self.Gateoutput = torch.nn.Linear(diminput+dimlatent,dimlatent)
        self.Linout = nn.Linear(dimlatent,dimoutput)

    def oneStep(self,seqsAtT,h,C):
        if len(seqsAtT.shape)<len(h.shape):seqsAtT = seqsAtT.unsqueeze(0)
        merge = torch.cat((h,seqsAtT),1)
        keep = self.sgma(self.Gateinput(merge))
        forget = self.sgma(self.Gateforget(merge))
        out = self.sgma(self.Gateoutput(merge))
        Ct = forget * C + keep * self.tanh(self.Gateinner(merge))
        return out * self.tanh(Ct), Ct
    
    def forward(self,seqs,h,C):
        # seqs' size is lenseq×sbatch×diminput. Starting from arbitrary state h, 
        # return the hidden states' predicted sequence of size lenseq×sbatch×dimlatent on input=seqs
        allhidden = [h]
        for timestamp in seqs:
            # C is new to this game
            h,C = self.oneStep(timestamp,allhidden[-1],C)
            allhidden += [h]
        return torch.stack(allhidden,axis=0),C
    
    def decode(self,h):
        # You can output this anytime (depends on the chosen architecture // many-to-many, many-to-one, (...)) 
        return self.Linout(h)
        
# /////////////////////////////////////////////////////////////////////////////////// </long short-term memory cells> ////
    
# /////////////////////////////////////////////////////////////////////////////////////////// <gated recurrent units> ////

class GRU(nn.Module):
    def __init__(self, diminput, dimlatent, dimoutput):
        super(GRU, self).__init__()
        self.diminput = diminput
        self.dimlatent = dimlatent
        self.dimoutput = dimoutput
        self.tanh = nn.Tanh()
        self.sgma = nn.Sigmoid()
        self.Gatez = torch.nn.Linear(diminput+dimlatent,dimlatent)
        self.Gater = torch.nn.Linear(diminput+dimlatent,dimlatent)
        self.Gateinner = torch.nn.Linear(diminput+dimlatent,dimlatent)
        self.Linout = nn.Linear(dimlatent,dimoutput)

    def oneStep(self, seqsAtT, h):
        if len(seqsAtT.shape)<len(h.shape):seqsAtT = seqsAtT.unsqueeze(0)
        merge = torch.cat((h,seqsAtT),1)
        zt = self.sgma(self.Gatez(merge))
        rt = self.sgma(self.Gater(merge))
        return (1-zt)*h + zt * self.tanh(self.Gateinner(torch.cat((h*rt, seqsAtT), 1)))
    
    def forward(self,seqs,h):
        # seqs' size is lenseq×sbatch×diminput. Starting from arbitrary state h, 
        # return the hidden states' predicted sequence of size lenseq×sbatch×dimlatent on input=seqs
        allhidden = [h]
        for timestamp in seqs:
            h = self.oneStep(timestamp,allhidden[-1])
            allhidden += [h]
        return torch.stack(allhidden,axis=0)    
    
    def decode(self,h):
        # You can output this anytime (depends on the chosen architecture // many-to-many, many-to-one, (...)) 
        return self.Linout(h)
        
# ////////////////////////////////////////////////////////////////////////////////////////// </gated recurrent units> ////