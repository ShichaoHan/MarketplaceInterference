import pandas as pd 
import numpy as np 

import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

import random
import time 


class MyModel_random():
    
    def predict(self, inputs, verbose=1):

        X_goodbads, X_utility, W_matrix, exposure_matrix = np.split(inputs, [1,2,3], axis=2)
        K = X_utility.shape[1]
        logit = np.ones_like(X_utility)
        logit = np.squeeze(logit, axis=-1)
        softmax_p = np.ones_like(logit) / K
        ypredicts = np.squeeze(X_utility, axis=-1)

        exposure = np.squeeze(exposure_matrix, axis=-1)

        y2 = np.sum(exposure * ypredicts, axis = 1, keepdims=True)
        res = np.concatenate([softmax_p, logit, y2, ypredicts], axis=1)
        return res

class MyModel_true():
    def __init__(self, k, promo):
        self.k = k
        self.promo = promo
        
    
    def predict(self, inputs, verbose=1):

        X_goodbads, X_utility, W_matrix, exposure_matrix = np.split(inputs, [1,2,3], axis=2)
        logit = self.promo * W_matrix * X_goodbads + X_utility
        logit = np.squeeze(logit, axis=-1)
        softmax_p =  np.exp(logit) / np.sum(np.exp(logit), axis=1, keepdims=True)

        ypredicts = np.squeeze(X_utility, axis=-1)

        exposure = np.squeeze(exposure_matrix, axis=-1)

        y2 = np.sum(exposure * ypredicts, axis = 1, keepdims=True)
        res = np.concatenate([softmax_p, logit, y2, ypredicts], axis=1)
        return res


class MyModel_embeddings(Model):
    def __init__(self, k, d, num_treats):
        super(MyModel_embeddings, self).__init__()
        self.k = k
        self.d = d
        self.num_treats = num_treats
        self.baseline_layer_1 = Dense(10, activation = "relu")
        self.baseline_layer_2 = Dense(10, activation = "relu")
        self.baseline_layer_logit = Dense(1, activation = "linear")
        
        self.uplift_layer_1 = {} 
        self.uplift_layer_2 = {} 
        self.uplift_layer_logit = {} 
        for g in range(num_treats):
            self.uplift_layer_1[g] = Dense(10, activation = "relu")
            self.uplift_layer_2[g] = Dense(10, activation = "relu")
            self.uplift_layer_logit[g] = Dense(1, activation = "linear")
        self.softmax = tf.keras.activations.softmax

        self.outcome_layer_1 = Dense(10, activation = "relu")
        self.outcome_layer_2 = Dense(10, activation = "relu")
        self.outcome_logit = Dense(1, activation = "linear")
        
    
    def call(self, inputs):

        split_structure =  [2 * self.d + 1] + [1] * self.num_treats + [1]
        splitted_elements = tf.split(inputs, split_structure, axis=2)
        x = splitted_elements[0]
        exposure = tf.squeeze(splitted_elements[-1], axis=-1)

        
        ## Step 2: Score 
        ### Baseline logit
        baseline = self.baseline_layer_1(x)
        #baseline = self.baseline_layer_2(baseline)
        logit = self.baseline_layer_logit(baseline)
        
        ### Uplift
        for g in range(self.num_treats):
            w_g = splitted_elements[g + 1]
            uplift = self.uplift_layer_1[g](x)
            #uplift = self.uplift_layer_2[g](uplift)
            uplift = self.uplift_layer_logit[g](uplift)
            logit = tf.add(tf.multiply(w_g, uplift), logit)
            
        ## Step 3: Softmax
        logit = tf.squeeze(logit, axis=-1)
        softmax_p =  tf.keras.activations.softmax(logit, axis=-1)

        ## Outcome 
        ypredicts = self.outcome_layer_1(x)
        #ypredicts = self.outcome_layer_2(ypredicts)
        ypredicts = self.outcome_logit(ypredicts)
        ypredicts = tf.squeeze(ypredicts, axis=-1)

        y2 = tf.reduce_sum(tf.multiply(exposure, ypredicts), axis = 1, keepdims=True)
        res = tf.concat([softmax_p, logit, y2, ypredicts], axis=1)
        return res


class MyModel_multiple(Model):
    def __init__(self, k, num_treats):
        super(MyModel_multiple, self).__init__()
        self.k = k
        self.num_treats = num_treats
        self.groupNames = ['A'] + ['B' + str(i+1) for i in range(self.num_treats)]
        self.baseline_logit = Dense(1, activation = "linear")
        self.outcome = Dense(1, activation = "linear")
        self.logit_dense_layer = {} 
        for g in self.groupNames:
            self.logit_dense_layer[g] = Dense(1, activation = "linear")
        self.softmax = tf.keras.activations.softmax
        
    
    def call(self, inputs):

        split_structure =  [2] + [1] * self.num_treats + [1]
        splitted_elements = tf.split(inputs, split_structure, axis=2)
        x1 = splitted_elements[0]
        exposure = tf.squeeze(splitted_elements[self.num_treats + 1], axis=-1)
        _, K, dim_x = x1.shape
        
        
        ## Step 1: Reshape the input 
        reshape_x1 = tf.reshape(x1, (-1, dim_x))
        
        ## Step 2: Score 
        ### Baseline logit
        x1_final = self.baseline_logit(x1)
        
        ### Uplift
        for i in range(self.num_treats):
            w_g = splitted_elements[i + 1]
            xg_hidden = self.logit_dense_layer['B'+str(i+1)](x1)
            x1_final = tf.add(tf.multiply(w_g, xg_hidden), x1_final)
            
        ## Step 3: Softmax
        logit = tf.reshape(x1_final, (-1, self.k))
        softmax_p =  self.softmax(logit, axis=-1)

        ## Outcome 
        ypredicts = self.outcome(x1)
        ypredicts = tf.squeeze(ypredicts, axis=-1)

        y2 = tf.reduce_sum(tf.multiply(exposure, ypredicts), axis = 1, keepdims=True)
        res = tf.concat([softmax_p, logit, y2, ypredicts], axis=1)
        return res


# Define custom loss function
def custom_loss(y_true, y_pred):
    K = y_true.shape[1] - 1
    y1_true, y2_true = tf.split(y_true, [K, 1], axis=1)
    _, y1_logit_pred, y2_pred, _= tf.split(y_pred, [K, K, 1, K], axis=1)
    loss1 = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y1_true, y1_logit_pred)
    loss2 = tf.keras.losses.MeanSquaredError()(y2_true, y2_pred)
    return loss1 + loss2