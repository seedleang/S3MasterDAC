from tme10dataset import *
from tme10models import *

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# //////////////////////////////////////////////////////////////////////////////////////////////////// <state saving> ////

class State(object) :
    def __init__(self, model, optim, linconv=None):
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0, 0
        
# /////////////////////////////////////////////////////////////////////////////////////////////////// </state saving> ////


# /////////////////////////////////////////////////////////////////////////////////////////////////// <training loop> ////

epochs = 42

def trainAttention(model=SelfAttention,dimbed=50,dimlat=40,dimout=2,nL=3,maxiter=epochs,lr=0.01,verbose=True):
    writer = SummaryWriter("runs")
    
    if model == SelfAttention :
        print("\n///////////////////////////// Self-attention classifier ////////////////////////////\n")
        name = "selfattention.pch"
        etiq = "/SelfA"
    elif model == SelfAttentionPE :
        print("\n//////////////////////////// Adding positional encodings ///////////////////////////\n")
        name = "selfattentionpe.pch"
        etiq = "/SelfAPE"
    elif model == SelfAttentionCLS :
        print("\n///////////////////////////// CLS-token classification /////////////////////////////\n")
        name = "selfattentioncls.pch"
        etiq = "/SelfACLS"
    elif model == SelfAttentionPE_CLS :
        print("\n////////////////////////// CLS with positional encodings ///////////////////////////\n")
        name = "selfattentionclspe.pch"
        etiq = "/SelfACLSPE"
        
    # Creating a checkpointed model
    savepath = Path(name)
    if savepath.is_file():
        print("Restarting from previous state.")
        with savepath.open("rb") as fp :
            state = torch.load(fp)
    else:
        lin = model(dimbed,dimlat,dimout,nL).to(device)
        optim = torch.optim.Adam(params=lin.parameters(),lr=lr)
        # optim = torch.optim.SGD(params=lin.parameters(),lr=lr)
        state = State(lin,optim)
    
    loss = nn.CrossEntropyLoss()
    
    # Training the model
    for epoch in tqdm(range(state.epoch,maxiter)):
        state.model = state.model.train()
        losstrain = 0
        accytrain = 0
        divtrain = 0
        for x, y in train_loader:
            state.optim.zero_grad()
            y = y.to(device)
            preds = state.model(x)
    
            ltrain = loss(preds,y.long())
            ltrain.backward()
            state.optim.step()
            state.iteration += 1
            
            acctr = sum((preds.argmax(1) == y)).item() / y.shape[0]
            losstrain += ltrain
            accytrain += acctr
            divtrain += 1
            
        state.model = state.model.eval()
        losstest = 0
        accytest = 0
        divtest = 0
        for x, y in test_loader:
            with torch.no_grad():
                y = y.to(device)
                preds = state.model(x)
                ltest = loss(preds,y.long()) 
                accts = sum((preds.argmax(1) == y)).item() / y.shape[0]  
            losstest += ltest
            accytest += accts
            divtest += 1        
        
        # Saving the loss
        writer.add_scalars('Attention/Loss'+etiq,{'train':losstrain/divtrain,'test':losstest/divtest},epoch)
        writer.add_scalars('Attention/Accuracy'+etiq,{'train':accytrain/divtrain,'test':accytest/divtest},epoch)
        if verbose:
            print('\nLOSS: \t\ttrain',(losstrain/divtrain).item(),'\t\ttest',(losstest/divtest).item())
            print('\nACCURACY: \t\ttrain',accytrain/divtrain,'\t\ttest',accytest/divtest)
        
        # Saving the current state after each epoch
        with savepath.open ("wb") as fp:
            state.epoch = epoch+1
            torch.save(state, fp)
            
    print("\n\n\033[1mDone.\033[0m\n")
    writer.flush()
    writer.close()
	
# ////////////////////////////////////////////////////////////////////////////////////////////////// </training loop> ////