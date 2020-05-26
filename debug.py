import os
import uuid
import shutil
from glob import glob
from datetime import date

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm, trange

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

data_train = np.reshape(x_train, [-1,28,28,1])
labels_train = y_train
data_val = data_train[45000:]
labels_val = labels_train[45000:]
data_train = data_train[:45000]
labels_train = data_train[:45000]

def batch_generator(mode="train", batch_size=100):
    """Generator for the data batches.
    
    Args:
        mode (str): Mode in ['train', 'val'] that decides which data set the generator
            samples from (default: 'train').
        batch_size (int): The size of the batches (default: 100).
        
    Yields:
        np.array: Data batch.
    """
    print('testing',flush=True)
    assert mode in ["train", "val"], "The mode should be in {train, val}."
    if mode=="train":
        images = data_train.copy()
        labels = labels_train.copy()
    elif mode=="val":
        images = data_val.copy()
        labels = labels_val.copy()
    
    while True:
        indices = np.random.permutation(np.arange(len(images)))
        images = images[indices]
        labels = labels[indices]
        time_series=True
        if time_series:
            print('yes')
            for i, image in enumerate(images):
                start_image = image
                end_image = images[np.random.choice(np.where(labels == (labels[i] + 1) % 10)[0])]
                interpolation = interpolate_arrays(start_image, end_image, batch_size)
                yield interpolation + np.random.normal(scale=0.01, size=interpolation.shape)
        else:
            for i in range(len(images)//batch_size):
                yield images[i*batch_size:(i+1)*batch_size]

def main():
    """Main method to build a model, train it and evaluate it.
    
    Args:
        latent_dim (int): Dimensionality of the SOM-VAE's latent space.
        som_dim (list): Dimensionality of the SOM.
        learning_rate (float): Learning rate for the training.
        decay_factor (float): Factor for the learning rate decay.
        alpha (float): Weight for the commitment loss.
        beta (float): Weight for the SOM loss.
        gamma (float): Weight for the transition probability loss.
        tau (float): Weight for the smoothness loss.
        modelpath (path): Path for the model checkpoints.
        save_model (bool): Indicates if the model should be saved after training and evaluation.
        
    Returns:
        dict: Results of the evaluation (NMI, Purity, MSE).
    """
    # Dimensions for MNIST-like data
    input_length = 28              #update for brains
    input_channels = 28            #update for brains
    print('ty')
    # get data 
    
    print('testing',flush=True)
    mode="train"
    batch_size=100
    #assert mode in ["train", "val"], "The mode should be in {train, val}."
    if mode=="train":
        images = data_train.copy()
        labels = labels_train.copy()
    elif mode=="val":
        images = data_val.copy()
        labels = labels_val.copy()
    
    while True:
        indices = np.random.permutation(np.arange(len(images)))
        images = images[indices]
        labels = labels[indices]
        time_series=True
        if time_series:
            print('yes')
            for i, image in enumerate(images):
                start_image = image
                end_image = images[np.random.choice(np.where(labels == (labels[i] + 1) % 10)[0])]
                interpolation = interpolate_arrays(start_image, end_image, batch_size)
                yield interpolation + np.random.normal(scale=0.01, size=interpolation.shape)
        else:
            for i in range(len(images)//batch_size):
                yield images[i*batch_size:(i+1)*batch_size] 





    #data_generator = get_data_generator(True)

    # build model
    #model = SOMVAE(inputs=x, latent_dim=latent_dim, som_dim=som_dim, learning_rate=lr_val, decay_factor=decay_factor,
    #        input_length=input_length, inputcd so_channels=input_channels, alpha=alpha, beta=beta, gamma=gamma,
    #        tau=tau, mnist=mnist)
    
    #x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28, 1])
    #lr_val = tf.compat.v1.placeholder_with_default(learning_rate, [])

    #train_model(model, x, lr_val, generator=data_generator)

    #result = evaluate_model(model, x)

    #if not save_model:
    #    shutil.rmtree(os.path.dirname(modelpath))


if __name__ == '__main__':
    
    tmp = main()