from tme9dataset import *
from tme9models import *

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

epochs = 100

def trainAttention(model=NaiveAttention,dimbed=50,dimout=2,maxiter=epochs,epsilon=0.01,reg=None,verbose=True):
    # Le paramètre "reg" permet de choisir de régulariser ou non avec un critère entropique.
    writer = SummaryWriter("runs")
    
    if model == NaiveAttention:
        print("\n//////////////////// Attention-based LinNet : naive baseline /////////////////\n")
        name = "attentionbase.pch"
        etiq = "/Base_SGD"
    elif model == SimpleAttention:
        print("\n///////////////// Attention-based LinNet : basic implementation //////////////\n")
        name = "attentionclassic.pch"
        etiq = "/Classic_SGD"
    elif model == FurtherAttention:
        print("\n///////////////// Attention-based LinNet : further improvements //////////////\n")
        name = "attentionfurther.pch"
        etiq = "/Further_SGD_regul"
    elif model == LSTMAttention:
        print("\n//////////////////// Attention-based LinNet : adding an LSTM /////////////////\n")
        name = "attentionlstm.pch"
        etiq = "/LSTM_SGD"
    elif model == BILSTMAttention:
        print("\n//////////////////// Attention-based LinNet : adding an BiLSTM /////////////////\n")
        name = "attentionbilstm.pch"
        etiq = "/BiLSTM_SGD"
    # Creating a checkpointed model
    savepath = Path(name)
    if savepath.is_file():
        print("Restarting from previous state.")
        with savepath.open("rb") as fp :
            state = torch.load(fp)
    else:
        lin = model(dimbed,dimout).to(device)
        optim = torch.optim.SGD(params=lin.parameters(),lr=epsilon)
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
            if model == NaiveAttention: preds = state.model(x)
            else: preds, attns = state.model(x)  
            if model != NaiveAttention:
                entropytrain = Categorical(probs = attns.squeeze(2).t()).entropy()
            penalty = reg * torch.sum(entropytrain) if reg else 0
            ltrain = loss(preds,y.long()) + penalty
            
            ltrain.backward()
            state.optim.step()
            state.iteration += 1
            acctr = sum((preds.argmax(1) == y)).item() / y.shape[0]
            losstrain += ltrain
            accytrain += acctr
            divtrain += 1
            
        #if model != NaiveAttention:
        #    entropytrain = Categorical(probs = attns.squeeze(2).t()).entropy()
            
        state.model = state.model.eval()
        losstest = 0
        accytest = 0
        divtest = 0
        for x, y in test_loader:
            with torch.no_grad():
                y = y.to(device)
                if model == NaiveAttention :
                    preds = state.model(x)
                else :
                    preds, attns = state.model(x)                
                ltest = loss(preds,y.long()) 
                accts = sum((preds.argmax(1) == y)).item() / y.shape[0]  
            losstest += ltest
            accytest += accts
            divtest += 1
            
        # Saving the loss
        writer.add_scalars('Attention/Loss'+etiq,{'train':losstrain/divtrain,'test':losstest/divtest},epoch)
        writer.add_scalars('Attention/Accuracy'+etiq,{'train':accytrain/divtrain,'test':accytest/divtest},epoch)
        
        if model != NaiveAttention :
            entropytest = Categorical(probs = attns.squeeze(2).t()).entropy()
            writer.add_histogram('Attention/EntropyTest'+etiq,entropytest,epoch)
            writer.add_histogram('Attention/EntropyTrain'+etiq,entropytrain,epoch)
        
        if verbose:
            print('\nLOSS: \t\ttrain',(losstrain/divtrain).item(),'\t\ttest',(losstest/divtest).item())
            print('ACCURACY: \ttrain',accytrain/divtrain,'\t\ttest',accytest/divtest)
        
        # Saving the current state after each epoch
        with savepath.open ("wb") as fp:
            state.epoch = epoch+1
            torch.save(state, fp)
            
    print("\n\n\033[1mDone.\033[0m\n")
    writer.flush()
    writer.close()

# ////////////////////////////////////////////////////////////////////////////////////////////////// </training loop> ////