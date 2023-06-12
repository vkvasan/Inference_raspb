import torch
from torch import nn, optim
from torch.nn import functional as F
import argparse
import datetime
import os
from copy import deepcopy
from tqdm import tqdm 
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
#import wandb
import torchvision.utils as vutils
from statsmodels.tsa.seasonal import STL
from scipy.signal import medfilt2d


from qkeras import QDense, QActivation, QConv1D, QConv2D
from qkeras.quantizers import quantized_bits, quantized_relu, smooth_sigmoid
from qkeras.utils import model_save_quantized_weights


def DataBatch(data, label, text, batchsize, shuffle=True):
    
    n = data.shape[0]
    if shuffle:
        index = np.random.permutation(n)
    else:
        index = np.arange(n)
    for i in range(int(np.ceil(n/batchsize))):
        inds = index[i*batchsize : min(n,(i+1)*batchsize)]
        yield data[inds], label[inds], text[inds]


def sample_noise(data, rate):
    a = int( data.shape[0] )
    b = int( data.shape[1] )
    c = int( data.shape[2] )
    #print( data.shape[0])
    noise = 0.1 * np.random.rand(a, b, c)
   
    idx1 = np.random.choice(data.shape[1], int(rate*data.shape[1]))
    noise_data = deepcopy(data)
    noise_data[:,idx1] = data[:,idx1] + noise[:,idx1]

    idx3 = np.random.choice(range(1,data.shape[1]), int(rate*(data.shape[1]-1)))
    noise_data[:,idx3] = noise_data[:,idx3-1]
    
    return noise_data, np.concatenate((idx1,idx3))


import tensorflow as tf

z = 10
epochs = 5001
bs = 16
seq_len = 3000
logdir = '/log'
model_path = '/checkpoint'
run_tag = ''
train = 1
lam = 0.5 
rate = 0.01
batchSize = 16 

# load dataset
high_vi = tf.convert_to_tensor(np.load('./data/train/high_vi_%d.npy'%seq_len))
print( high_vi.dtype)
high_w = tf.convert_to_tensor(np.load('./data/train/high_w_%d.npy'%seq_len))
print( high_w.dtype)

high_a = tf.convert_to_tensor(np.load('./data/train/high_a_%d.npy'%seq_len))
print( high_a.dtype)

low_vi = tf.convert_to_tensor(np.load('./data/test/low_vi_%d.npy'%seq_len))
low_w = tf.convert_to_tensor(np.load('./data/test/low_w_%d.npy'%seq_len))
low_a = tf.convert_to_tensor(np.load('./data/test/low_a_%d.npy'%seq_len))

seq_len, vi_dim, imu_w_dim, imu_a_dim = high_vi.shape[1], high_vi.shape[2], high_w.shape[2], high_a.shape[2]


high_vi = high_vi.numpy()
high_w = high_w.numpy()
high_a = high_a.numpy()


from tensorflow.keras.layers import Input, Dense,Embedding,Dropout,Activation,Conv1D,ReLU,Conv2D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam, SGD

in_dim = vi_dim+imu_w_dim+imu_a_dim
z_dim = z
out_dim=vi_dim

from tensorflow.keras.layers import Input, Conv2D, Activation, Reshape, ZeroPadding2D
from tensorflow.keras.models import Model

inputs = Input(shape=(seq_len, in_dim), name='input_c')
print(inputs.shape)

# Reshape input to 2D
x = Reshape((seq_len, in_dim, 1))(inputs)

x = ZeroPadding2D(padding=((6, 6), (0, 0)))(x)

x = Conv2D(filters=128, kernel_size=(7, in_dim), padding='valid')(x)
print(x.shape)
x = Activation('relu')(x)

x = Conv2D(filters=256, kernel_size=(5, 1), padding='valid')(x)
x = Activation('relu')(x)
print(x.shape)

x = Conv2D(filters=128, kernel_size=(3, 1), padding='valid')(x)
x = Activation('relu')(x)
print(x.shape)

x = Conv2D(filters=out_dim, kernel_size=(1, 1), padding='valid')(x)
print(x.shape)

# Reshape output back to 2D
out = Reshape((seq_len, out_dim))(x)
print(out.shape)
# Crop output to match the desired shape
cropped_out = out[:, :seq_len, :]

print(cropped_out.shape)
model = Model(inputs, cropped_out)


model = Model(inputs, out)

model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

from tensorflow.keras.losses import mean_squared_error
mse = tf.keras.losses.MeanSquaredError()

testgenerator_x = tf.concat([low_vi, low_w, low_a],axis=-1)
testgenerator_y = low_vi
#print(testgenerator_x.shape,testgenerator_y.shape)


single_sample_x = testgenerator_x[0,:,:]
single_sample_y = testgenerator_y[0,:,:]
#print(single_sample_x.shape,single_sample_y.shape)

import keras

model = keras.models.load_model('my_model.h5')

import time
import numpy

input_data_list = []

num_runs = 5
inference_times = []

a = numpy.zeros((1,3000,13))

#print( a.shape)

for i in range(num_runs):
    a[0,:,:] = testgenerator_x[i,:,:]
    start_time = time.time()    
    output = model.predict(a)
    end_time = time.time()
    inference_time_no_quant = end_time - start_time
    inference_times.append(inference_time_no_quant)

# Calculate average inference time
average_inference_time = sum(inference_times) / len(inference_times)

print("Average Inference Time without quantization: {:.2f} ms".format(average_inference_time * 1000))



import tensorflow as tf


input_shape = (1,3000,13)
# Load the trained Keras mod
# Convert the Keras model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply post-training quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = tf.lite.RepresentativeDataset(
    lambda: [[np.random.rand(*input_shape).astype(np.float32)]])  # Provide representative dataset

# Set the quantization type to int8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Set the input and output type to int8
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.int8

#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Convert the model to TensorFlow Lite format
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
input_data = input_data.reshape(input_shape)


import time

interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

num_runs = 10
inference_times = []
for _ in range(num_runs):
    start_time = time.time()
    
    # Run inference
    interpreter.invoke()

    end_time = time.time()
    inference_time = end_time - start_time
    inference_times.append(inference_time)

# Calculate average inference time
average_inference_time = sum(inference_times) / len(inference_times)

# Print the average inference time
print("Average Inference Time after quantization: {:.2f} ms".format(average_inference_time * 1000))
# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
