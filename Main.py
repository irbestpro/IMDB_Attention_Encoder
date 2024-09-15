'''
    Written By: Mehdi Touyserkani - Aug 2024.
    https://ir-bestpro.com.
    https://www.linkedin.com/in/bestpro-group/
    https://github.com/irbestpro/
    ir_bestpro@yahoo.com
    BESTPRO SOFTWARE ENGINEERING GROUP

'''

import numpy as np
import pandas as pd
from Model.Sentiment_Classifier import Classifier

if __name__ == '__main__' :

    #______LOAD THE EMBEDDED VECTORS_____________

    Embedd_Data = pd.read_csv('./Embedded/Vectors.csv').to_numpy(dtype = np.float32)[: , 1:] # load embedded sequence dataset
    Labels = pd.read_csv('./Embedded/Labels.csv').to_numpy(dtype = np.float32)[ : , 1:] # load labels

    #__________MODEL CONTEXT MANAGER_____________

    with Classifier() as model:
        model.Preprocessing(Embedd_Data , Labels)
        model.Classifiy()
    
