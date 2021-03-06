from tp9 import *
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ///////////////////////////////////////////////////////////////////////////////////////////////// <GloVE embedding> ////

word2id, embedding, foldertext_train, foldertext_test = get_imdb_data()
id2word = {v: k for k, v in word2id.items()}

# //////////////////////////////////////////////////////////////////////////////////////////////// </GloVE embedding> ////

# //////////////////////////////////////////////////////////////////////////////////////////////// <dataset handling> ////

PAD_IX = -1
torchemb = torch.FloatTensor(embedding).to(device)

def padding(batch):
    texts = [torch.tensor(t[0]) for t in batch]
    labels = [t[1] for t in batch]
    x = torch.nn.utils.rnn.pad_sequence(texts,padding_value=PAD_IX)
    # Chaque mot est encodé par un entier : c'est un index dans la matrice "embedding"
    x = torchemb[x] 
    return x, torch.tensor(labels)

# Les batchs ont dû être fortement réduits pour ne pas remplir totalement la mémoire du GPU.
BATCH_SIZE = 16
train_loader = DataLoader(foldertext_train, BATCH_SIZE,collate_fn=padding,shuffle=True,drop_last=True)
test_loader = DataLoader(foldertext_test, BATCH_SIZE,collate_fn=padding,shuffle=True,drop_last=True)

# /////////////////////////////////////////////////////////////////////////////////////////////// </dataset handling> ////