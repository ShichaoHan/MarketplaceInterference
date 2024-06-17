import pandas as pd 
import tensorflow_ranking as tfr 
from scipy.special import kl_div
import numpy as np 

## Modifying the tensor for 3d input 
class MyModel_multiple(Model):
    def __init__(self, k, num_treats,predictY=False):
        super(MyModel_multiple, self).__init__()
        self.k = k
        self.num_treats = num_treats
        self.groupNames = ['A'] + ['B' + str(i+1) for i in range(self.num_treats)]
        
        self.baseline_logit = Dense( 1, activation = "linear")
        self.logit_dense_layer = {} 
        for g in self.groupNames:
            self.logit_dense_layer[g] = Dense(1, activation = "linear")
        self.common_hidden = Dense(3, activation = "linear")
        self.softmax =tf.keras.activations.softmax
        
        self.predictY = predictY
        self.d5 = Dense(self.k, activation = "relu")
        self.doutcome = Dense(1)
        
        self.d_onehot = tf.keras.layers.Lambda(lambda x: tf.one_hot(tf.argmax(x, axis=-1), k))
        
    
    def call(self, inputs):

        split_structure =  [4] + [1] * self.num_treats + [1]
        splitted_elements = tf.split(inputs, split_structure, axis=2)
        x1 = splitted_elements[0]
        ypredicts = splitted_elements[-1]
        
        
        
        
        ## Step 1: Reshape the input 
        
        reshape_x1 = tf.reshape(x1, (-1, x1.shape[-1]))
        
        ## Step 2: a common hidden layer
        x1_common_hidden = self.common_hidden(reshape_x1)
        
        #x1_common_hidden_3d = tf.reshape(x1_common_hidden, [x1.shape[0],self.k, x1_common_hidden.shape[1]])
        x1_common_hidden_3d = tf.reshape(x1_common_hidden,[-1, k, 3])
        
        ## Baseline logit
        x1_final = self.baseline_logit(x1_common_hidden_3d)
        
        # Get the first element of the second dimension
        # first_element = x1_final[:, 0:1, :]

        # Subtract the first element from every element of the second dimension
        # x1_final = x1_final - first_element

        
        ## Step 3: logit model
        for i in range(self.num_treats):
            w_g = splitted_elements[i + 1]
            xg_hidden = self.logit_dense_layer['B'+str(i+1)](x1_common_hidden_3d)
            x1_final = tf.add(tf.multiply(w_g, xg_hidden), x1_final)
            
        ## Step 4: Softmax
        reshaped_data = tf.squeeze(x1_final, axis=-1)
        softmax_p =  self.softmax(reshaped_data)

        # for g in self.groupNames:
        #     if g != 'A':
        #         w_g = self.treatment_matrix_dict[g]
        #         x = tf.math.add(tf.multiply(w_g, self.logit_dense_layer[g](x1)), x)
        #x = tf.add(tf.multiply(self.treatment_matrix_dict['B1'],  x2_hidden), x1_hidden)

        # x = tf.mul( w, x1_hidden)

        y1 = self.d_onehot(softmax_p)
        
        if self.predictY:
            x5 = self.d5(x1)
            y2 = self.doutcome(tf.multiply(softmax_p, x5))
        else:
            
            ypredicts = splitted_elements[-1]
            ypredicts = tf.squeeze(ypredicts, axis=-1)
            
            y2 = tf.reduce_sum(tf.multiply(softmax_p, ypredicts), axis = 1 )
            y2 = tf.expand_dims(y2, axis=-1)

            
        # ## One hot vector  
        # y1 = softmax_p
        # x5 = self.d5(ypredicts)
        # print("softmax shape", y1.shape)
        
        # ## Outcome 
    
        # y2 = self.doutcome(tf.multiply(y1, x5))
        


        res = tf.concat([softmax_p,y2,softmax_p], axis=1)
        return res

# Define custom loss function
def custom_loss(y_true, y_pred):

    y1_true, y2_true = tf.split(y_true, [K, 1], axis=1)
    _, y2_pred, y1_pred = tf.split(y_pred, [K, 1, K], axis=1)


    loss1 = tf.keras.losses.CategoricalCrossentropy()(y1_true, y1_pred)
    loss2 = tf.keras.losses.MeanSquaredError()(y2_true, y2_pred)
    return loss1 + loss2 
    
    
    
    