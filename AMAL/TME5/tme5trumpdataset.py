import string
import unicodedata
import numpy as np
from typing import List
import torch
from torch.utils.data import Dataset

"""Ajout d'un caractère de fin + possibilité de longueurs différentes par rapport à la dernière fois"""

# /////////////////////////////////////////////////////////////////////////////////////////////// <d. trump speeches> ////

PAD_IX = 0
EOS_IX = 1
LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
id2lettre = dict(zip(range(2, len(LETTRES)+2), LETTRES))
id2lettre[PAD_IX] = '' # NULL CHARACTER
id2lettre[EOS_IX] = '|' # END OF STRING
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    return torch.tensor([lettre2id[c] for c in normalize(s)]).long()

def code2string(t):
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

def listen(path):
    data = ""
    f = open(path, "r")
    for s in f:
        data+=normalize(s)
    return data
    
def collate_fn(samples: List[List[int]]):
    until = np.max([len(sentence) for sentence in samples])
    res = []
    eos = torch.tensor([EOS_IX], dtype=torch.long)
    for sentence in samples:
        pads = torch.full((until-len(sentence),), PAD_IX, dtype=torch.long)
        res.append(torch.cat((sentence, pads, eos), 0))
    return torch.stack(res).t()

class handleSpeech(Dataset):
    def __init__(self, text: str, *, maxsent=None, maxlen=None):
        sentences = [sent.strip() for sent in text.split(".")]
        if maxlen == None:
            maxlen = len(text)
        self.data = [string2code(sent+".") for sent in sentences if len(sent)>1 and len(sent)<maxlen]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i] 
    
# /////////////////////////////////////////////////////////////////////////////////////////////// <d. trump speeches> ////