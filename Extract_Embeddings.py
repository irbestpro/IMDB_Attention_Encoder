'''
    Written By: Mehdi Touyserkani - Aug 2024.
    https://ir-bestpro.com.
    https://www.linkedin.com/in/bestpro-group/
    https://github.com/irbestpro/
    ir_bestpro@yahoo.com
    BESTPRO SOFTWARE ENGINEERING GROUP

'''

import pandas as pd
import numpy as np
import torch
import string

from transformers import AutoTokenizer , BertModel
from Model.Params import Params

if __name__ == '__main__':

    Train_Data = pd.read_csv('./IMDB_Dataset/Train.csv') # load IMDB Dataset (TrainSet)
    Test_Data = pd.read_csv('./IMDB_Dataset/Test.csv') # load IMDB Dataset (TestSet)

    Train_Data['text'] = Train_Data['text'].map(lambda x: x.translate(str.maketrans('', '', string.punctuation))) # remove the punctuations from texts
    Test_Data['text'] = Test_Data['text'].map(lambda x: x.translate(str.maketrans('', '', string.punctuation))) # remove the punctuations from texts

    Train_Data = Train_Data.iloc[: , 1:].to_numpy() # convert to numpy array
    Test_Data = Test_Data.iloc[: , 1:].to_numpy() # convert to numpy array

    Threshold = 4000
    Train_Samples = np.concatenate((Train_Data[(Train_Data[:,1] == 1) , :][0:Threshold] , # Split 8000 samples as Train Data
                                    Train_Data[(Train_Data[:,1] == 0) , :][0:Threshold]))
    
    Test_Samples = np.concatenate((Test_Data[(Test_Data[:,1] == 1) , :][-Threshold:] , # Split 8000 samples as Test Data
                                   Test_Data[(Test_Data[:,1] == 0) , :][-Threshold:]))
    
    all_Data = np.concatenate((Train_Samples , Test_Samples)) # concatenation of Train and Test samples

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') # Bert Auto Tokenizer
    embedding_Layer = BertModel.from_pretrained("bert-base-uncased", attn_implementation="sdpa") # Bert Transformer
    embedded_vectors = [] # Text, labels and Tokens at Input 
    Texts = []
    labels = []
    tokens = []

    [(Texts.append(item[0]) , labels.append (item[1])) for item in all_Data]
    labels = np.array(labels)
    [(print(f'Tokenizing Text # {idx}'),tokens.append(tokenizer(txt, padding="max_length", max_length = Params.MAX_SEQUENCE_LENGHT ,truncation = True , return_tensors='pt'))) for idx,txt in enumerate(Texts)] # Bert auto tokenizer

    with torch.no_grad():
        [embedded_vectors.append(embedding_Layer(item['input_ids'],attention_mask=item['attention_mask']).last_hidden_state.mean(dim=1).squeeze(0).detach().numpy()) for item in tokens] # using the Bert transformer for creating the embedded vectors of input sequences

    pd.DataFrame(embedded_vectors).to_csv("./Embedded/Vectors.csv") # save the embedded vectors
    pd.DataFrame(labels).to_csv("./Embedded/Labels.csv") # save the related labels
