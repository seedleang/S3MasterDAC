import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm.auto import tqdm

from tme5trumpdataset import *
from tme5generation import *
from tme5models import *



# Implementing a suitable loss function that overrides all padding characters
def maskedCrossEntropy(output, target, padcar=PAD_IX):
    pureloss = F.cross_entropy(output,target,reduction='none')
    mask0 = torch.zeros_like(pureloss)
    mask1 = torch.ones_like(pureloss)
    mask = torch.where(target==padcar,mask0,mask1)
    fineloss = torch.sum(pureloss*mask)/torch.sum(mask)
    return fineloss






# //////////////////////////////////////////////////////////////////////////////////////////////// <let's talk Trump> ////

def trainModel(modeltype,etiq,dimbed=60,latent=len(lettre2id),maxlen=200,maxiter=250,epsilon=0.001,sbatch=64):
    writer = SummaryWriter("runs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ls = torch.nn.LogSoftmax(dim=1)
    embedding = nn.Embedding(len(id2lettre),dimbed,padding_idx=0).to(device)
    
    C=False
    if modeltype == LSTM: C=True
    # Making the train set (no test set), feeding the DataLoader
    print("\nGenerating the sequences...")
    trainspeech = DataLoader(handleSpeech(listen('./data/trump_full_speech.txt'),maxlen=maxlen),shuffle=True,collate_fn=collate_fn,batch_size=sbatch,drop_last=True)

    print("\n/////////////////////////////// "+etiq+"-based generation using an embedding ///////////////////////////////\n")

    # Creating a checkpointed model
    savepath = Path("generate"+etiq+str(dimbed)+str(maxlen)+".pch")
    if savepath.is_file():
        print("Restarting from previous state.")
        with savepath.open("rb") as fp :
            state = torch.load(fp)
    else:
        model = modeltype(dimbed,latent,len(lettre2id)+1).float()
        model = model.to(device)
        optim = torch.optim.Adam(params=model.parameters(),lr=epsilon)
        state = State(model,optim)

    # Training the model
    for epoch in tqdm(range(state.epoch,maxiter)):
        for x in trainspeech:
            x = x.to(device)
            state.optim.zero_grad()
            # Embedding
            bed = embedding(x)
            
            ltrain = 0
            ht = torch.zeros(sbatch,latent,requires_grad=True).float().to(device)
            # Handling the LSTM case where C has to be defined 
            if C: Ct = torch.zeros(sbatch,latent,requires_grad=True).float().to(device)
            for t in range(0,len(x)-1):
                # Considering each letter in the input sequence, 
                # predicting a distribution for the next one, computing the loss
                if C: ht,Ct = state.model.oneStep(bed[t],ht,Ct) # (LSTM case)
                else: ht = state.model.oneStep(bed[t],ht)
                ltrain += maskedCrossEntropy(ls(state.model.decode(ht)),x[t+1].long())
            ltrain /= (len(x)-1)
            ltrain.backward()
            state.optim.step()
            state.iteration += 1

        # Saving the loss
        writer.add_scalars('Loss/'+etiq+'/Generate/Embedding'+str(dimbed)+'/Length'+str(maxlen),{'train':ltrain},epoch)
        # Saving the current state after each epoch
        with savepath.open ("wb") as fp:
            state.epoch = epoch+1
            torch.save(state, fp)
            
    # Generating stuff
    generate(model=state.model, decoder=state.model.decode, ls=ls, latent=latent, embedding=embedding, device=device, maxlen=maxlen, C=C)
    generateBeam(model=state.model, decoder=state.model.decode, ls=ls, latent=latent, embedding=embedding, device=device, maxlen=maxlen, C=C)
    generateBeam(model=state.model, decoder=state.model.decode, ls=ls, latent=latent, embedding=embedding, device=device, maxlen=maxlen, C=C, kNucleus=7)
    generateBeam(model=state.model, decoder=state.model.decode, ls=ls, latent=latent, embedding=embedding, device=device, maxlen=maxlen, C=C, kNucleus=3)


    print("\n\n\033[1mDone.\033[0m\n")
    writer.flush()
    writer.close()

# /////////////////////////////////////////////////////////////////////////////////////////////// </let's talk Trump> ////