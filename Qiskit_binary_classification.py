# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 10:31:51 2020

@author: Soumik Adhikary
"""

import qiskit
qiskit.__qiskit_version__
from qiskit import IBMQ
IBMQ.load_account()
from qiskit import *
import matplotlib.pyplot as plt
import tensorflow as tf
import numdifftools as nd
import numpy as np
import pickle
import pandas as pd
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.visualization import plot_bloch_multivector


#for k in tf.__dict__.keys():
#    print(k)
#print(tf.__version__)






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

#tensorflow_variable_and_parameter_initialisation
def get_parameter(num_param,var):
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    param=tf.compat.v1.get_variable(name="weight", dtype=tf.float64,shape=[num_param,],initializer=tf.compat.v1.initializers.truncated_normal(mean=0.0,stddev=var))
    init = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session()
    sess.run(init)
    return param.eval(sess)

#classical_pre_processing(dressing)
def to_single_number(w,x):
    pre_processed_vector = x*w
    return np.sum(pre_processed_vector)

#qiskit_circuit
def forward_pass(pre_processed_vector, alpha_1, alpha_2, alpha_3):
    pi = 3.14
    quantum_register = QuantumRegister(1)
    classical_register = ClassicalRegister(1)

    circuit = QuantumCircuit(quantum_register, classical_register)

    repetitions = 10000
    count_0 = []
    count_1 = []
    
    
    m_simulator = Aer.get_backend('qasm_simulator')

    circuit.h(quantum_register[0])
    circuit.rz(2*pi*pre_processed_vector, quantum_register[0])
    circuit.rz(2*pi*alpha_1, quantum_register[0])
    circuit.ry(2*pi*alpha_2, quantum_register[0])
    circuit.rz(2*pi*alpha_3, quantum_register[0])
    circuit.measure(quantum_register, classical_register)
    execute(circuit, backend = m_simulator, shots=repetitions)
    result_m = execute(circuit, backend = m_simulator, shots=repetitions).result()
    count_0.append(result_m.get_counts(circuit)['0'])
    count_1.append(result_m.get_counts(circuit)['1'])
    
    prob_0 = count_0[0]/repetitions
    prob_1 = count_1[0]/repetitions
    
    return prob_0, prob_1


#loss_calculation
def loss_function(param, x_train_A, x_train_B):
    w=param[0:x_train_A.shape[1]] 
    alpha_1 = param[x_train_A.shape[1]]  
    alpha_2 = param[x_train_A.shape[1]+1]  
    alpha_3 = param[x_train_A.shape[1]+2]    
    loss_A = 0
    loss_B = 0
    for i in range(len(x_train_A)):
        pre_processed_vector_A = to_single_number(w,x_train_A[i])
        pre_processed_vector_B = to_single_number(w,x_train_B[i])
        prob_A_0, prob_A_1 = forward_pass(pre_processed_vector_A, alpha_1, alpha_2, alpha_3)
        prob_B_0, prob_B_1 = forward_pass(pre_processed_vector_B, alpha_1, alpha_2, alpha_3)
        loss_A += (1-prob_A_0)
        loss_B += (1-prob_B_1)
    return loss_A+loss_B

#accuracy_calculation
def accuracy(param, x_train_A,x_train_B,x_test_A,x_test_B):
    w=param[0:x_train_A.shape[1]] 
    alpha_1 = param[x_train_A.shape[1]]  
    alpha_2 = param[x_train_A.shape[1]+1]  
    alpha_3 = param[x_train_A.shape[1]+2]
    train_count_A = 0
    train_count_B = 0
    test_count_A = 0
    test_count_B = 0
    for i in range(len(x_train_A)):
        pre_processed_vector_A = to_single_number(w,x_train_A[i])
        pre_processed_vector_B = to_single_number(w,x_train_B[i])
        prob_A_0_train, prob_A_1_train = forward_pass(pre_processed_vector_A, alpha_1, alpha_2, alpha_3)
        prob_B_0_train, prob_B_1_train = forward_pass(pre_processed_vector_B, alpha_1, alpha_2, alpha_3)
        if (prob_A_0_train > prob_A_1_train):
            train_count_A += 1
        if (prob_B_1_train > prob_B_0_train):
            train_count_B += 1
    train_accuracy = (train_count_A+train_count_B)/(len(x_train_A)+len(x_train_B))
    for i in range(len(x_test_A)):
        pre_processed_vector_A = to_single_number(w,x_test_A[i])
        pre_processed_vector_B = to_single_number(w,x_test_B[i])
        prob_A_0_test, prob_A_1_test = forward_pass(pre_processed_vector_A, alpha_1, alpha_2, alpha_3)
        prob_B_0_test, prob_B_1_test = forward_pass(pre_processed_vector_B, alpha_1, alpha_2, alpha_3)
        if (prob_A_0_test > prob_A_1_test):
            test_count_A += 1
        if (prob_B_1_test > prob_B_0_test):
            test_count_B += 1
    test_accuracy = (test_count_A+test_count_B)/(len(x_test_A)+len(x_test_B))
    return train_accuracy, test_accuracy


    
import tqdm
param = get_parameter(x_train_setosa.shape[1]+3,0.1)
losses_train=[]
losses_test=[]
accuracies_train=[]
accuracies_test=[]
grad=nd.Gradient(loss_function)
m=np.random.random()
v=np.random.random()
for t in tqdm.tqdm(range(1,10)):
    print(t)
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon= 1e-8
    g = grad(param,x_train_setosa,x_train_virginica)
    m = beta_1 * m + (1 - beta_1) * g
    v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
    m_hat = m / (1 - np.power(beta_1, t))
    v_hat = v / (1 - np.power(beta_2, t))
    param = param - (1e-2) * m_hat / (np.sqrt(v_hat) + epsilon)
    if(t%1==0):
        train_acc,test_acc=accuracy(param, x_train_setosa, x_train_virginica, x_test_setosa, x_test_virginica)
        info_train=loss_function(param, x_train_setosa, x_train_virginica)
        info_test=loss_function(param, x_test_setosa, x_test_virginica)
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


plt.plot(losses_train,label='Training loss')
plt.plot(losses_test,label='Testing loss')
plt.xlabel("Adam Iterations")
plt.ylabel("Loss")
plt.legend(loc='best')
plt.grid()
plt.suptitle("Learning Curve for IRIS")
plt.savefig("Loss_IRIS.png")
plt.show()       
    
    
plt.plot(accuracies_train,label='Training accuracy')
plt.plot(accuracies_test,label='Testing accuracy')
plt.xlabel("Adam Iterations")
plt.ylabel("Accuracies")
plt.legend(loc='best')
plt.grid()
plt.suptitle("Learning Curve for IRIS")
plt.savefig("Accuracy_IRIS.png")
plt.show()    
    
    
   
#def generic_function(k,g):
#    return k*k+g*g
#    
#k = get_parameter(1,0.2) 
#g = get_parameter(1,0.2) 
#generic_grad = nd.Gradient(generic_function)()
#print(k,g)
#print(generic_function(k,g))
#print(generic_grad)
    
#generic_array = nd.array([2,1,3,4,5,3,6,4]) 
#    
#generic_array[np.random.choice(6)]
    

#circuit.draw(output = 'mpl')

#plot_histogram(result_m.get_counts(circuit))










#quantum_register = QuantumRegister(1)
#classical_register = ClassicalRegister(1)
#
#circuit = QuantumCircuit(quantum_register, classical_register)
#
#m_simulator = Aer.get_backend('qasm_simulator')
#
#count_0 = []
#count_1 = []
#
#for i in range(4):
#    circuit.h(quantum_register[0])
#    circuit.measure(quantum_register, classical_register)
#    execute(circuit, backend = m_simulator, shots = 100000)
#    result_m = execute(circuit, backend = m_simulator).result()
#    count_0.append(result_m.get_counts(circuit)['0'])
#    print(result_m.get_counts(circuit))
#    
#    
#
#
#print(count_0)
#count_0 = result_m.get_counts(circuit)['0']
#count_1 = result_m.get_counts(circuit)['1']
#
#probability_0 = count_0/1024
#probability_1 = count_1/1024




















#s_simulator = Aer.get_backend('statevector_simulator')
#execute(circuit, backend = s_simulator)
#result_s = execute(circuit, backend = s_simulator).result()
#print(result_s.get_statevector())
#plot_bloch_multivector(result_s.get_statevector())

































