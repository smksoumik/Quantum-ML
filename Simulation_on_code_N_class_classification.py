# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 09:55:54 2020

@author: Soumik Adhikary
"""

import qiskit
qiskit.__qiskit_version__
from qiskit import IBMQ
IBMQ.load_account()
from qiskit import *
import math
import cmath
import matplotlib.pyplot as plt
import tensorflow as tf
import numdifftools as nd
import numpy as np
import pickle
import pandas as pd
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.visualization import plot_bloch_multivector

#import_dataset
mydataset = pd.read_excel('fisher_iris_input.xlsx')

#class_formation (2class_setosa_virginica in Iris)
data=np.array(mydataset)
x_setosa = data[0:50]
x_virginica = data[50:100]
x_versicolor = data[100:150]

#train_test_sets_formation
x_test_setosa = x_setosa[40:50]
x_train_setosa = x_setosa[0:40]

x_test_virginica = x_virginica[40:50]
x_train_virginica = x_virginica[0:40]

x_test_versicolor = x_versicolor[40:50]
x_train_versicolor = x_versicolor[0:40]

train_sets = np.array([x_train_setosa,x_train_versicolor,x_train_virginica])
test_sets =  np.array([x_test_setosa,x_test_versicolor,x_test_virginica])

#tensorflow_variable_and_parameter_initialisation
def get_parameter(number_of_classes,num_param,var):
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    param=tf.compat.v1.get_variable(name="weight", dtype=tf.float64,shape=[number_of_classes,num_param,],initializer=tf.compat.v1.initializers.truncated_normal(mean=0.0,stddev=var))
    init = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session()
    sess.run(init)
    return param.eval(sess)

#classical_pre_processing(dressing)
def to_single_number(w,x):
    pre_processed_vector = x*w
    return np.sum(pre_processed_vector)

#forward_pass_and_probability_calculation
def forward_pass(pre_processed_vector, alpha_1, alpha_2, alpha_3):
    pi = 3.14
    
    proj_0 = np.array([[1],[0]])
    proj_1 = np.array([[0],[1]])
    
    state_initial = np.array([[1],[0]])
    hadamard = np.array([[(1/ math.sqrt(2)),(1/ math.sqrt(2))],[(1/ math.sqrt(2)),-(1/ math.sqrt(2))]])
    encoder = math.cos(2*pi*pre_processed_vector)*np.array([[1,0],[0,1]]) + complex(0,math.sin(2*pi*pre_processed_vector))*np.array([[1,0],[0,-1]])
    
    U_z_1 = math.cos(2*pi*alpha_1)*np.array([[1,0],[0,1]]) + complex(0,math.sin(2*pi*alpha_1))*np.array([[1,0],[0,-1]])
    U_y = math.cos(2*pi*alpha_2)*np.array([[1,0],[0,1]]) + complex(0,math.sin(2*pi*alpha_2))*np.array([[0,complex(0,-1)],[complex(0,1),0]])
    U_z_3 = math.cos(2*pi*alpha_3)*np.array([[1,0],[0,1]]) + complex(0,math.sin(2*pi*alpha_3))*np.array([[1,0],[0,-1]])
    SU_2 = np.matmul(U_z_1,np.matmul(U_y,U_z_3))
    
    state_hadamard = np.matmul(hadamard,state_initial)
    state_encoded = np.matmul(encoder,state_hadamard)
    state_encoded_rotation = np.matmul(SU_2,state_encoded)
    
    prob_0 = np.matmul(proj_0.transpose(),state_encoded_rotation)*np.conjugate(np.matmul(proj_0.transpose(),state_encoded_rotation))
    prob_1 = np.matmul(proj_1.transpose(),state_encoded_rotation)*np.conjugate(np.matmul(proj_1.transpose(),state_encoded_rotation))
    
    return prob_0[0][0].real, prob_1[0][0].real

#loss_calculation
def loss_function_linear(param, x_train, number_of_classes):
    loss = 0
    param_mod = np.reshape(param, (number_of_classes,x_train[1].shape[1]+3))
    for i in range(number_of_classes):
        w=param_mod[i][0:x_train[i].shape[1]]
        alpha_1 = param_mod[i][x_train[i].shape[1]]
        alpha_2 = param_mod[i][x_train[i].shape[1]+1]
        alpha_3 = param_mod[i][x_train[i].shape[1]+2]
        for k in range(number_of_classes):
            if(k == i):
                for j in range(len(x_train[k])):
                    pre_processed_vector = to_single_number(w,x_train[k][j])
                    prob_0, prob_1 = forward_pass(pre_processed_vector, alpha_1, alpha_2, alpha_3)
                    loss += (1-prob_0)
            else:
                for j in range(len(x_train[k])):
                    pre_processed_vector = to_single_number(w,x_train[k][j])
                    prob_0, prob_1 = forward_pass(pre_processed_vector, alpha_1, alpha_2, alpha_3)
                    loss += (1-prob_1)
        #print("loss",loss)
    return loss

#accuracy_calculation
def accuracy(param, x_train, x_test, number_of_classes):
    probability_train_accuracy = []
    probability_test_accuracy = []
    train_accuracy_count = 0
    test_accuracy_count = 0
    for k in range(number_of_classes):
        for j in range(len(x_train[k])):
            for i in range(number_of_classes):
                w=param[i][0:x_train[i].shape[1]]
                alpha_1 = param[i][x_train[i].shape[1]]
                alpha_2 = param[i][x_train[i].shape[1]+1]
                alpha_3 = param[i][x_train[i].shape[1]+2]
                pre_processed_vector = to_single_number(w,x_train[k][j])
                prob_0, prob_1 = forward_pass(pre_processed_vector, alpha_1, alpha_2, alpha_3)
                probability_train_accuracy.append(prob_0)
            if(probability_train_accuracy[k] == max(probability_train_accuracy)):
               train_accuracy_count += 1
            probability_train_accuracy = []
    
    training_samples = 0
    for i in range(number_of_classes):
        training_samples += len(x_train[k])
    
    train_accuracy = train_accuracy_count/training_samples
    
    for k in range(number_of_classes):
        for j in range(len(x_test[k])):
            for i in range(number_of_classes):
                w=param[i][0:x_train[i].shape[1]]
                alpha_1 = param[i][x_train[i].shape[1]]
                alpha_2 = param[i][x_train[i].shape[1]+1]
                alpha_3 = param[i][x_train[i].shape[1]+2]
                pre_processed_vector = to_single_number(w,x_test[k][j])
                prob_0, prob_1 = forward_pass(pre_processed_vector, alpha_1, alpha_2, alpha_3)
                probability_test_accuracy.append(prob_0)
            if(probability_test_accuracy[k] == max(probability_test_accuracy)):
               test_accuracy_count += 1
            probability_test_accuracy = []
    
    test_samples = 0
    for i in range(number_of_classes):
        test_samples += len(x_test[k])
    
    
    test_accuracy = test_accuracy_count/test_samples
        
    return train_accuracy, test_accuracy        
      
import tqdm
minimization_iteration = 5
param = get_parameter(3,x_train_setosa.shape[1]+3,0.1)
print("param shape",param.shape)
losses_train=[]
losses_test=[]
accuracies_train=[]
accuracies_test=[]
grad=nd.Gradient(loss_function_linear)
m=np.random.random()
v=np.random.random()
number_of_classes = 3
for t in tqdm.tqdm(range(1,minimization_iteration)):
    print(t)
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon= 1e-8
    print(data.shape)
    g = grad(param, train_sets,3)
    g=np.reshape(g,(number_of_classes,param.shape[1]))
    m = beta_1 * m + (1 - beta_1) * g
    v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
    m_hat = m / (1 - np.power(beta_1, t))
    v_hat = v / (1 - np.power(beta_2, t))
    param = param - (1e-2) * m_hat / (np.sqrt(v_hat) + epsilon)
    if(t%1==0):
        train_acc,test_acc=accuracy(param, train_sets, test_sets,3)
        info_train=loss_function_linear(param, train_sets,3)
        info_test=loss_function_linear(param, test_sets,3)
        print("Train Loss : "+str(info_train))
        print("Train Accuracy : "+str(train_acc)) 
        print("Test Loss : "+str(info_test))
        print("Test Accuracy : "+str(test_acc)) 
        losses_train.append(info_train)
        accuracies_train.append(train_acc)
        losses_test.append(info_test)
        accuracies_test.append(test_acc)



losses_train=np.array(losses_train)
accuracies_train=np.array(accuracies_train)
losses_test=np.array(losses_test)
accuracies_test=np.array(accuracies_test)
data=[losses_train,losses_test,accuracies_train,accuracies_test,param]
#print("--Training Finished--")
#print("--Dumping necessary files--")
#with open("wbc_adam.pickle",'wb') as pickle_file:
#	pickle.dump(data,pickle_file,protocol=pickle.HIGHEST_PROTOCOL)
#
#print("--Dumping Done--")


plt.plot(losses_train,label='Training accuracy')
plt.plot(losses_test,label='Testing accuracy')
plt.xlabel("Adam Iterations")
plt.ylabel("Training Loss")
plt.legend(loc='best')
plt.grid()
plt.suptitle("Learning Curve for IRIS")
plt.savefig("Loss_IRIS.png")
plt.show()      
    












































