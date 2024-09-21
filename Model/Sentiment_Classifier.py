'''
    Written By: Mehdi Touyserkani - Aug 2024.
    https://ir-bestpro.com.
    https://www.linkedin.com/in/bestpro-group/
    https://github.com/irbestpro/
    ir_bestpro@yahoo.com
    BESTPRO SOFTWARE ENGINEERING GROUP

'''

import torch
import torch.optim
import pickle
from torch import nn
import matplotlib.pyplot as plt

from Model.Attention_Encoder import (AutoEncoder , Dense_Network)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import get_scheduler
from Model.Params import Params


class Classifier():
    
    def __init__(self):
        self.embedding_words = []
        self.labels = []
        self.train_acc = []
        self.test_acc = []
    
    #_________using the BERT Transformer(Embedding) as pre-processing phase_________

    def Preprocessing(self , embedded_vector , labels):
        self.embedding_words = torch.tensor(embedded_vector).unsqueeze(1) # casting the embedded words to tensor array
        self.labels = torch.tensor(labels).squeeze(1) # casting the labels to tensor array 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.embedding_words, self.labels, test_size = 0.25, random_state = 42) # Split Train and Test samples to 75-25
    
    #__________handle the model context manager______________

    def __enter__(self):
        return self
    
    #__________Save the models in binary files_______________

    def __exit__(self, exc_type, exc_val, exc_tb):
        
        pickle.dump(self.encoder_model , open('./encoder_model.nila' , 'wb'))
        pickle.dump(self.dense_model , open('./dense_model.nila' , 'wb'))

        plt.plot(range(0,len(self.train_acc)), self.train_acc , label = "Training Phase")
        plt.plot(range(0,len(self.train_acc)), self.test_acc , label = "Test Phase")
        plt.title('Accuracy score of IMDB Benchmark sentiment classification')
        plt.legend()
        plt.show()

    #________start the classification________________________

    def Classifiy(self):

        self.encoder_model = AutoEncoder() # EncoderModel object
        self.dense_model = Dense_Network() # DenseModel object
        encoder_Loss = nn.MSELoss() # Encoder loss function
        dense_Loss = nn.CrossEntropyLoss() # Classifier loss function

        encoder_optimizer = torch.optim.Adam(params = self.encoder_model.parameters() , lr = Params.LEARNING_RATE , weight_decay = Params.WEIGHT_DECAY) # Adam optimizer
        Dense_Optimizer = torch.optim.Adam(params = self.dense_model.parameters() , lr = Params.LEARNING_RATE , weight_decay = Params.WEIGHT_DECAY) # Adam optimizer
        scheduler = get_scheduler(name="linear" , optimizer = Dense_Optimizer , num_training_steps = Params.EPOCHS * len(self.y_train) , num_warmup_steps = 30) # Warm-up the learning rate by scheduler

        #_______________TRAIN PHASE____________________

        self.dense_model.train()
        self.encoder_model.train()
        for epoch in range(0 , Params.EPOCHS):
            encoder_temp_loss = 0
            dense_temp_loss = 0
            temp_acc = 0
            steps = 0
            for batch in range(0 , self.X_train.shape[0] , Params.BATCH_SIZE):
                X = torch.as_tensor(self.X_train[batch : batch +  Params.BATCH_SIZE])
                Y = torch.as_tensor(self.y_train[batch : batch +  Params.BATCH_SIZE]).type(torch.LongTensor)

                X , encoder_out , weights = self.encoder_model(X) # encoded sequences
                dense_out = self.dense_model(weights) # predicted data   

                encode_loss = encoder_Loss(encoder_out, X) # encoder loss value
                encoder_temp_loss += encode_loss.item()
                loss = dense_Loss(dense_out , Y) # densemodel loss value
                dense_temp_loss += loss.item()
                
                predicted = torch.argmax(dense_out.data , dim = 1) # predicted labels
                temp_acc += accuracy_score(Y , predicted)
                
                steps +=1
                self.encoder_model.zero_grad()
                self.dense_model.zero_grad()
                (encode_loss + loss).backward() # aggregated backward process

                encoder_optimizer.step() # encoder optimizer new step
                Dense_Optimizer.step() # dense optimizer new step
                scheduler.step()

        #_______________TEST PHASE____________________

            self.dense_model.eval()
            self.encoder_model.eval()
            with torch.no_grad():
                _ , _ , encoded = self.encoder_model(torch.as_tensor(self.X_test)) # Encode input sequence
                output = self.dense_model(encoded) # get the densemodel outputs
                predicted = torch.argmax(output.data , dim = 1) # get the final labels
                test_accuracy = accuracy_score(torch.as_tensor(self.y_test) , predicted) # final model accuracy

            self.train_acc.append(temp_acc / steps * 100)
            self.test_acc.append(test_accuracy * 100)
            print(f'{epoch = } , Encoder Loss value : {encoder_temp_loss / steps} , Dense Loss value : {dense_temp_loss / steps} , Train Acc : {self.train_acc[-1]}% , Test acc : {self.test_acc[-1]}%')

        