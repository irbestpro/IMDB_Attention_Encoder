'''
    Written By: Mehdi Touyserkani - Aug 2024.
    https://ir-bestpro.com.
    https://www.linkedin.com/in/bestpro-group/
    https://github.com/irbestpro/
    ir_bestpro@yahoo.com
    BESTPRO SOFTWARE ENGINEERING GROUP
    
'''

import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.kdim = 128
        self.MHSA = nn.MultiheadAttention(embed_dim = self.kdim , num_heads = 1 , kdim = self.kdim , vdim = self.kdim , batch_first = True) # Multi Head Self Attention Layer

        self.Linear1 = nn.Sequential(nn.Linear(768 , 1024)  , nn.LeakyReLU())
        self.Linear2 = nn.Sequential(nn.Linear(1024 , 2048) , nn.LeakyReLU())
        self.Linear3 = nn.Sequential(nn.Linear(2048 , 4096) , nn.LeakyReLU())
        self.Linear4 = nn.Sequential(nn.Linear(4096 , 768)  , nn.LeakyReLU())
        self.Linear5 = nn.Sequential(nn.Linear(768 , 768)  , nn.Tanh())

    def forward(self , X):
        X_temp = X.reshape(X.size(0) , 6 , -1)
        mhsa1 = self.MHSA(query = X_temp , key = X_temp , value = torch.randn((X.size(0) , 6 , self.kdim)))[0].reshape(X.size(0) , -1)
        out = self.Linear1(mhsa1)
        out = self.Linear2(out)
        out = self.Linear3(out)
        out = self.Linear4(out)
        out = self.Linear5(out)
        temp_out = torch.matmul(X , self.Linear5[0].weight.T).reshape(X.size(0) , 6 , -1)
        mhsa2 = self.MHSA(query = temp_out , key = temp_out , value = temp_out)[0].reshape(X.size(0) , -1)
        return mhsa1 , out , mhsa2 # Multi-Head-Self-Attention layer1 / Reconstructed vector / Weights of Network in last layer / Multi-Head-Self-Attention layer2

class Dense_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.FC = nn.Sequential(nn.Linear(768 , 1024) , nn.LeakyReLU() , nn.Linear(1024 , 2)) # classify input sequences in 2 classes.
        
    def forward(self , X):
        return self.FC(X).squeeze(1)


        