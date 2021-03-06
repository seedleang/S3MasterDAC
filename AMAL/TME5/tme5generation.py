from tme5trumpdataset import *
import torch
from torch.distributions import Categorical

def handlingInit(x,function,latent,embedding,C,device):
    # Just a shorthand for code readability
    bed = embedding(x.to(device))
    h = torch.zeros(1,latent).float().to(device)
    if C:
        Ct = torch.zeros(1,latent).float().to(device)
        ht,Ct = function(bed,h,Ct)
        return ht,Ct
    ht = function(bed,h)
    return ht,None

s = ["[appl", ". ","The world ","A lot ","We w","$","That's not ","I am ","Our ","Hi","It's","The U","These"]

def generate(model, decoder, ls, latent, embedding, device, eos=EOS_IX, start=s, maxlen=100, C=False):
    print("\033[1mTrying to generate new sentences\033[0m")
    
    print("\n...from known init strings (argmax)\n")
    for i in range(len(start)):
        # Full forward on each init string
        x = string2code(start[i])
        ht,Ct = handlingInit(x,model.forward,latent,embedding,C,device)
        ht = ht[-1]
        distr = ls(decoder(ht))
        out = [distr.argmax(1)]  
        t = 1
        while out[-1] != eos and t<maxlen: 
            if C: ht,Ct = model.oneStep(embedding(out[-1]),ht,Ct)
            else: ht = model.oneStep(embedding(out[-1]),ht)
            distr = ls(decoder(ht))
            out.append(distr.argmax(1))
            t += 1
        out = torch.stack(out).squeeze(1)
        init = [i.item() for i in x]
        code = [c.item() for c in out]
        print(code2string(init)+"\033[92m"+code2string(code)+"\033[0m")
        
    print("\n...from empty strings (full sampling from the predicted distributions)\n")
    for _ in range(10):
        ht = torch.zeros(1,latent).float().to(device)
        if C: Ct = torch.zeros(1,latent).float().to(device)
        out = [torch.tensor([0]).to(device)]
        t = 0
        while out[-1] != eos and t<maxlen: 
            if C: ht,Ct = model.oneStep(embedding(out[-1]),ht,Ct)
            else: ht = model.oneStep(embedding(out[-1]),ht)
            distr = Categorical(logits=ls(decoder(ht)))
            out.append(distr.sample())
            t += 1
        out = torch.stack(out).squeeze(1)
        code = [c.item() for c in out]
        print("\033[92m"+code2string(code)+"\033[0m")
        
    print("\n...from empty strings again (sampling the first letter + argmax)\n")
    for _ in range(10):
        ht,Ct = handlingInit(torch.tensor([0]),model.oneStep,latent,embedding,C,device) 
        distr = Categorical(logits=ls(decoder(ht)))
        out = [distr.sample()]
        t = 1
        while out[-1] != eos and t<maxlen: 
            if C: ht,Ct = model.oneStep(embedding(out[-1]),ht,Ct)
            else: ht = model.oneStep(embedding(out[-1]),ht)
            distr = ls(decoder(ht))
            out.append(distr.argmax(1))
            t += 1
        out = torch.stack(out).squeeze(1)
        code = [c.item() for c in out]
        print("\033[92m"+code2string(code)+"\033[0m")

        
def generateBeam(model, decoder, ls, latent, embedding, device, eos=EOS_IX, start=s, k=7, maxlen=100, C=False, kNucleus=None):
    print("\n\n\033[1mGeneration with a beam search (k="+str(k)+")\033[0m\n")
    nucleaire = None
    color = '\033[96m'
    if kNucleus: 
        nucleaire = pNucleus(decoder, kNucleus)
        color = '\033[36m'
        print("Adding a nucleus sampler (range="+str(kNucleus)+")\n")

    for i in range(len(start)):
        possibilities = []
        x = string2code(start[i])
        # Keeping a list version of each init string in memory
        init = x.reshape(-1).cpu().tolist()
        # The model's first hidden state estimate on the init string
        ht,Ct = handlingInit(x,model.forward,latent,embedding,C,device) 
        distr = ls(decoder(ht.squeeze(1))) 
        
        score = sum([distr[i][c] for i, c in enumerate(init)]).cpu().item() # pas 0 à cause du log
        possibilities = [(init + [c], score + p) for c, p in enumerate(distr[-1])]
        possibilities = sorted(possibilities, key=lambda x: x[1])[-k:]

        for _ in range(maxlen-1):
            possibilities = beamStep(possibilities, model, decoder, ls, latent, embedding, device, eos, k, C, nucleaire)
          
        choice = possibilities[0][0]
        until = np.where(np.array(choice) == eos)[0]
        if len(until): until = until[0]
        else: until=maxlen
        init = code2string(init)
        print(init+color+code2string(choice)[len(init):until+1]+"\033[0m")
            

def beamStep(existingposs, model, decoder, ls, latent, embedding, device, eos, k, C, nucleus):
    extend = []
    for seq, score in existingposs:
        x = torch.LongTensor(seq).reshape(-1,1)
        ht,Ct = handlingInit(x,model.forward,latent,embedding,C,device)
        ht = ht[-1]
        if nucleus: distr = nucleus(ht)
        else: distr = ls(decoder(ht)) 
        extend += [(seq + [c], score + p) for c, p in enumerate(distr.squeeze(0))]
    return sorted(extend, key=lambda x: x[1])[-k:]   
                
                
def pNucleus(decoder, k: int):
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties
        Si k==vocab_size alors p_nucleus ne change pas la distribution
        h (torch.Tensor): l'état à décoder (shape == (1, hidden_size))
        """
        p = torch.softmax(decoder(h).squeeze(0),dim=0)
        topk = p.topk(k)
        refactor = p.detach().clone()
        refactor[topk.indices] = 0
        final = (p-refactor) / torch.sum(topk.values)
        return final
    return compute