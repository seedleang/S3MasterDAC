import torch
from torch.utils.data import Dataset
import string
import unicodedata

# /////////////////////////////////////////////////////////////////////////////////////////////// <d. trump speeches> ////

LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]= ''
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

def listen(path):
    data = []
    f = open(path, "r")
    for s in f:
        data.extend(string2code(normalize(s)))
    return torch.stack(data)

class handleSpeech(Dataset):
    def __init__(self,data,limit):
        self.data = data[:-(data.shape[0]%limit)].reshape(-1,limit).float()
        self.lngt = len(self.data)
        
    def __getitem__(self, index):
        return self.data[index], index
            
    def __len__(self):
        return self.lngt
    
# ////////////////////////////////////////////////////////////////////////////////////////////// </d. trump speeches> ////
