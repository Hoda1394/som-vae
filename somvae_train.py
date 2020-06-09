"""
Script to train the SOM-VAE model as described in https://arxiv.org/abs/1806.02199
Copyright (c) 2018
Author: Vincent Fortuin
Institution: Biomedical Informatics group, ETH Zurich
License: MIT License

If you want to optimize the hyperparameters using labwatch, you have to install labwatch and SMAC
and comment in the commented out lines.
"""

import os
import uuid
import shutil
from glob import glob
from datetime import date

import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
from tqdm import tqdm, trange
import sacred
from sacred.stflow import LogFileWriter

# from labwatch.assistant import LabAssistant
# from labwatch.optimizers.random_search import RandomSearch
# from labwatch.optimizers.smac_wrapper import SMAC
# from labwatch.optimizers.bayesian_optimization import BayesianOptimization
# from labwatch import hyperparameters as hyper

from somvae_model import SOMVAE
from utils import *

ex = sacred.Experiment("hyperopt")
ex.observers.append(sacred.observers.FileStorageObserver.create("../sacred_runs"))
ex.captured_out_filter = sacred.utils.apply_backspaces_and_linefeeds

# ex.observers.append(sacred.observers.MongoObserver.create(db_name="somvae_hyperopt"))
# assistant = LabAssistant(ex, "somvae_hyperopt", optimizer=SMAC, url="localhost:{}".format(db_port))

@ex.config
def ex_config():
    """Sacred configuration for the experiment.
    
    Params:
        num_epochs (int): Number of training epochs.
        patience (int): Patience for the early stopping.
        batch_size (int): Batch size for the training.
        latent_dim (int): Dimensionality of the SOM-VAE's latent space.
        som_dim (list): Dimensionality of the self-organizing map.
        learning_rate (float): Learning rate for the optimization.
        alpha (float): Weight for the commitment loss.
        beta (float): Weight for the SOM loss.
        gamma (float): Weight for the transition probability loss.
        tau (float): Weight for the smoothness loss.
        decay_factor (float): Factor for the learning rate decay.
        name (string): Name of the experiment.
        ex_name (string): Unique name of this particular run.
        logdir (path): Directory for the experiment logs.
        modelpath (path): Path for the model checkpoints.
        interactive (bool): Indicator if there should be an interactive progress bar for the training.
        data_set (string): Data set for the training.
        save_model (bool): Indicator if the model checkpoints should be kept after training and evaluation.
        time_series (bool): Indicator if the model should be trained on linearly interpolated
            MNIST time series.
        mnist (bool): Indicator if the model is trained on MNIST-like data.
    """
    num_epochs = 20
    patience = 100
    batch_size = 32
    latent_dim = 64
    som_dim = [8,8]
    learning_rate = 0.0005
    alpha = 1.0
    beta = 0.9
    gamma = 1.8
    tau = 1.4
    decay_factor = 0.9
    name = ex.get_experiment_info()["name"]
    ex_name = "{}_{}_{}-{}_{}_{}".format(name, latent_dim, som_dim[0], som_dim[1], str(date.today()), uuid.uuid4().hex[:5])
    logdir = "../logs/{}".format(ex_name)
    modelpath = "../models/{}/{}.ckpt".format(ex_name, ex_name)
    interactive = True
    data_set = "MNIST_data"
    save_model = False
    time_series = True
    mnist = True


# @assistant.search_space
# def search_space():
#     num_epochs = 20
#     patience = 20
#     batch_size = 32
#     latent_dim = hyper.UniformInt(lower=64, upper=256, log_scale=True)
#     som_dim = [8,8]
#     learning_rate = hyper.UniformFloat(lower=0.0001, upper=0.01, log_scale=True)
#     alpha = hyper.UniformFloat(lower=0., upper=2.)
#     beta = hyper.UniformFloat(lower=0., upper=2.)
#     gamma = hyper.UniformFloat(lower=0., upper=2.)
#     tau = hyper.UniformFloat(lower=0., upper=2.)
#     decay_factor = hyper.UniformFloat(lower=0.8, upper=1.)
#     interactive = False

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

data_train = np.reshape(x_train, [-1,28,28,1])
labels_train = y_train
data_val = data_train[45000:].astype(np.float32)
labels_val = labels_train[45000:]
data_train = data_train[:45000].astype(np.float32)
labels_train = data_train[:45000]

#@ex.capture
def get_data_generator(time_series):
    """Creates a data generator for the training.
    
    Args:
        time_series (bool): Indicates whether or not we want interpolated MNIST time series or just
            normal MNIST batches.
    
    Returns:
        generator: Data generator for the batches."""

    def batch_generator(mode="train", batch_size=100):
        """Generator for the data batches.
        
        Args:
            mode (str): Mode in ['train', 'val'] that decides which data set the generator
                samples from (default: 'train').
            batch_size (int): The size of the batches (default: 100).
            
        Yields:
            np.array: Data batch.
        """

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
            if time_series:
                for i, image in enumerate(images):
                    start_image = image
                    end_image = images[np.random.choice(np.where(labels == (labels[i] + 1) % 10)[0])]
                    interpolation = interpolate_arrays(start_image, end_image, batch_size)
                    yield interpolation + np.random.normal(scale=0.01, size=interpolation.shape)
            else:
                for i in range(len(images)//batch_size):
                    yield images[i*batch_size:(i+1)*batch_size]

    return batch_generator


@ex.capture
def train_model(model, lr_val, num_epochs, patience, batch_size, logdir,
        modelpath, learning_rate, interactive, generator):
    """Trains the SOM-VAE model.
    
    Args:
        model (SOM-VAE): SOM-VAE model to train.
        x (tf.Tensor): Input tensor or placeholder.
        lr_val (tf.Tensor): Placeholder for the learning rate value.
        num_epochs (int): Number of epochs to train.
        patience (int): Patience parameter for the early stopping.
        batch_size (int): Batch size for the training generator.
        logdir (path): Directory for saving the logs.
        modelpath (path): Path for saving the model checkpoints.
        learning_rate (float): Learning rate for the optimization.
        interactive (bool): Indicator if we want to have an interactive
            progress bar for training.
        generator (generator): Generator for the data batches.
    """
    train_gen = generator("train", batch_size)
    val_gen = generator("val", batch_size)

    #saver = tf.compat.v1.train.Saver(keep_checkpoint_every_n_hours=2.)    #could be upgraded
    #summaries = tf.compat.v1.summary.merge_all()                          #could be upgraded

    #with tf.compat.v1.Session() as sess:
    #sess.run(tf.compat.v1.global_variables_initializer())
    
    #with LogFileWriter(ex):                                                          #Sacred
    #    train_writer = tf.compat.v1.summary.FileWriter(logdir+"/train", sess.graph)  #could be upgraded to TFv.2
    #    test_writer = tf.compat.v1.summary.FileWriter(logdir+"/test", sess.graph)    #could be upgraded to TFv.2
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
    
    # Initialize
    num_batches = len(data_train)//batch_size
    patience_count = 0
    step = 0
    test_losses = []
    writer = tf.summary.create_file_writer("../models/test")

    def train_step(inputs,epoch,batch):
        with tf.GradientTape() as tape:
            model.call(inputs=inputs)
            train_loss = model.loss()
            #print("Epoch {}, batch {}, loss {}".format(epoch,batch,train_loss))

        grads = tape.gradient(train_loss,model.trainable_variables)
        #lr_decay = tf.compat.v1.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_factor, staircase=True)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return train_loss
        
    @tf.function
    def call_train_step(inputs,epoch,batch):
        print('hey')
        train_loss = train_step(inputs,epoch,batch)
        return train_loss

    print("Training...")
    try:
        if interactive:
            pbar = tqdm(total=num_epochs*(num_batches)) 

        for epoch in range(num_epochs):
            batch_val = next(val_gen)
            model.call(inputs=batch_val)
            test_losses.append(model.loss())

            with writer.as_default():
                    tf.summary.scalar("test loss", test_losses[-1], step=step)
                    writer.flush() 

            #if test_losses[-1] == min(test_losses):
            #    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
            #    checkpoint.save(modelpath)
            #    patience_count = 0
            #else:
            #    patience_count += 1
            #if patience_count >= patience:
            #    break

            for i in range(num_batches):
                step += 1
                batch_train = next(train_gen)

                train_loss = call_train_step(batch_train,epoch,i)
                break

                if i%100 == 0:
                    with writer.as_default():
                        tf.summary.scalar("train loss", train_loss, step=step)
                        writer.flush()

                #train_step_SOMVAE.run(feed_dict={x: batch_data, lr_val:learning_rate})
                #train_step_prob.run(feed_dict={x: batch_data, lr_val:cd ``})

                if interactive:
                    pbar.set_postfix(epoch=epoch, train_loss=train_loss.numpy(), test_loss=test_losses[-1].numpy(), refresh=False)
                    pbar.update(1)
            break

    except KeyboardInterrupt:
        pass
    finally:
        #model.save('../models/model.h5')
        #tmp = tf.keras.models.load_model('../models/model.h5')
        if interactive:
            pbar.close()

@ex.capture
def evaluate_model(model,x, modelpath, batch_size):
    """Evaluates the performance of the trained model in terms of normalized
    mutual information, purity and mean squared error.
    
    Args:
        model (SOM-VAE): Trained SOM-VAE model to evaluate.
        x (tf.Tensor): Input tensor or placeholder.
        modelpath (path): Path from which to restore the model.
        batch_size (int): Batch size for the evaluation.
        
    Returns:
        dict: Dictionary of evaluation results (NMI, Purity, MSE).
    """
    num_batches = len(data_val)//batch_size

    test_k_all = []
    test_rec_all = []
    test_mse_all = []
    print("Evaluation...")
    for i in range(num_batches):
        batch_data = data_val[i*batch_size:(i+1)*batch_size]
        model.call(inputs=batch_data)
        test_k_all.extend(model.k)
        test_rec = model.reconstruction_q
        test_rec_all.extend(test_rec)
        test_mse_all.append(mean_squared_error(tf.reshape(test_rec,[-1]),tf.reshape(batch_data,[-1]) ))

    test_nmi = compute_NMI(test_k_all, labels_val[:len(test_k_all)])
    test_purity = compute_purity(test_k_all, labels_val[:len(test_k_all)])
    test_mse = np.mean(test_mse_all)

    results = {}
    results["NMI"] = test_nmi
    results["Purity"] = test_purity
    results["MSE"] = test_mse

    return results
 

@ex.automain
def main(latent_dim, som_dim, learning_rate, decay_factor, alpha, beta, gamma, tau, modelpath, save_model, mnist):
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
    input_duration = 32

    # get data 
    data_generator = get_data_generator(True)

    x=1
    lr_val = tf.compat.v1.placeholder_with_default(learning_rate, [])

    # build model
    model = SOMVAE(latent_dim=latent_dim, som_dim=som_dim, learning_rate=lr_val, decay_factor=decay_factor,
                input_length=input_length, input_channels=input_channels, batch_size=input_duration,alpha=alpha, beta=beta, gamma=gamma,
                tau=tau, mnist=mnist)

    train_model(model,lr_val, generator=data_generator)

    result = evaluate_model(model,x=x)

    #if not save_model:
    #    shutil.rmtree(os.path.dirname(modelpath))
    print(result)
    return result


