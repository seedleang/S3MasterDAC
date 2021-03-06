import torch
from torch.utils.data import Dataset
import numpy as np
import random
import csv

# /////////////////////////////////////////////////////////////////////////////////////////////// <city temperatures> ////

def fill_na(mat):
    # Filling the missing values in the data matrix
    ix,iy = np.where(np.isnan(mat))
    for i,j in zip(ix,iy):
        # The rest of the sequence might be unknown
        # -> make it stationary
        if np.isnan(mat[i+1,j]):
            mat[i,j]=mat[i-1,j]
        # When it's known - when there is only one tick missing, 
        # we compute an average to fill in the blanks
        else:
            mat[i,j]=(mat[i-1,j]+mat[i+1,j])/2.
    return mat


def read_temps(path):
    # Extracting the data from the CSV file
    data = []
    with open(path, "rt") as fp:
        reader = csv.reader(fp, delimiter=',')
        next(reader)
        for row in reader:
            if not row[1].replace(".","").isdigit():
                continue
            data.append([float(x) if x != "" else float('nan') for x in row[1:]])
    return torch.tensor(fill_na(np.array(data)), dtype=torch.float)
    
    
"""Pour créer l'ensemble de train, la méthode retenue consiste à 
récupérer aléatoirement 10 000 séquences (fonction makeSets)
*après avoir généré toutes les possibilités* (fonction makeSeqs).
Note : on a conscience que ç'aurait été inenvisageable sur un dataset plus grand."""


def makeSeqs(lenseq,step,monoville): 
    # The full training set will consist of all the sequences of length lenseq, 
    # separated by a "stride" parameter "step" // THEY OVERLAP
    tempstrain = read_temps('data/tempAMAL_train.csv')
    # Total recording time over the train set
    traintime = tempstrain.shape[0]
    
    # The full testing set will consist of all the sequences of length lenseq with NO OVERLAPPING
    tempstest = read_temps('data/tempAMAL_test.csv')
    # Total recording time over the test set
    testtime = tempstest.shape[0]
    
    trainX = []
    trainY = []
    testX = []
    testY = []
    
    # Case 1: collecting data from a single city
    if monoville:
        for i in range(0,traintime-lenseq,step):
            trainX.append(tempstrain[i:i+lenseq,monoville])
            trainY.append(monoville)
        for i in range(0,testtime-lenseq,lenseq):
            testX.append(tempstest[i:i+lenseq,monoville])
            testY.append(monoville)
        return trainX,trainY,testX,testY
    
    # Case 2: collecting data from all 10 cities and labelling the sequences accordingly
    for citynum in range(10,20):
        for i in range(0,traintime-lenseq,step):
            trainX.append(tempstrain[i:i+lenseq,citynum])
            trainY.append(citynum-10)
        for i in range(0,testtime-lenseq,lenseq):
            testX.append(tempstest[i:i+lenseq,citynum])
            testY.append(citynum-10)
    return trainX,trainY,testX,testY


def makeSets(lenseq,step,traincontent=10000,monoville=None): 
    trainX,trainY,testX,testY = makeSeqs(lenseq,step,monoville)
    
    # Random selection - sampling a maximum of 10 000 sequences from the training set
    idx = np.sort(random.sample(range(len(trainX)-1),min(len(trainX)-1,traincontent)))
    trainX = [trainX[i] for i in idx]
    trainX = torch.stack(trainX)
    mean = trainX.mean()
    std = trainX.std()
    # Normalisation
    trainX -= mean
    trainX /= std
    
    # Keeping the computed test set as is.
    testX = torch.stack(testX)
    # Normalisation
    testX -= mean
    testX /= std
    
    # Matching labels
    trainY = [trainY[i] for i in idx]
    trainY = torch.from_numpy(np.array(trainY))
    testY = torch.from_numpy(np.array(testY))
    return trainX,trainY,testX,testY,mean,std


class handleWeather(Dataset) :
    # Dataset object to be used in the DataLoader
    def __init__(self, X, Y):
        self.data = X
        self.data = self.data.view(self.data.shape[0],-1)
        self.etiq = Y
        self.lngt = len(self.etiq)

    def __getitem__(self, index):
        return self.data[index], self.etiq[index]
    
    def __len__(self) :
        return self.lngt
    
# ////////////////////////////////////////////////////////////////////////////////////////////// </city temperatures> ////

