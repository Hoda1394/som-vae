import os
import uuid
import shutil
import time
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

from somvae_model import SOMVAE
from utils import *

ex = sacred.Experiment("hyperopt")
ex.observers.append(sacred.observers.FileStorageObserver.create("../sacred_runs"))
ex.captured_out_filter = sacred.utils.apply_backspaces_and_linefeeds

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
    modelpath = "./models/{}/{}.ckpt".format(ex_name, ex_name)
    interactive = True
    data_set = "MNIST_data"
    save_model = False
    time_series = True
    mnist = False


@ex.capture
def train_model(model, num_epochs, patience, batch_size, logdir,
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

    learning_decay = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_rate=0.9, decay_steps=1000,staircase=True, name='Exp_decay')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_decay)

    learning_decay = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate*100, decay_rate=0.9, decay_steps=1000,staircase=True, name='Exp_decay')
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=learning_decay)

    # Initialize
    num_batches = len(list(generator.as_numpy_iterator()))
    patience_count = 0
    step = 0
    test_losses = []
    writer = tf.summary.create_file_writer("./models/test")
        
    def train_step(inputs):
        with tf.GradientTape() as tape:
            model.call(inputs=inputs)
            train_loss = model.loss()
        grads = tape.gradient(train_loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return train_loss
    
    def train_step_prob(inputs):
        with tf.GradientTape() as tape:
            model.transition_probabilities = model.get_transition_probabilities()
            train_loss_prob = model.loss_probabilities()
        grads = tape.gradient(train_loss_prob,model.raw_probabilities)
        optimizer2.apply_gradients(zip([grads], [model.raw_probabilities]))
        return train_loss_prob

    @tf.function
    def call_train_step(inputs):
        loss = train_step(inputs)
        loss_prob = train_step_prob(inputs)
        return loss

    print("Training...")
    try:
        if interactive:
            pbar = tqdm(total=num_epochs*(num_batches)) 

        for epoch in range(num_epochs):
            #batch_val = next(val_gen)
            #model.call(inputs=batch_val)
            #test_losses.append(model.loss())
            test_losses.append(tf.constant(0))    # TODO : add validation dataset

            with writer.as_default():
                tf.summary.scalar("test loss", test_losses[-1], step=step)
                writer.flush() 

            if test_losses[-1] == min(test_losses):
                checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
                checkpoint.save(modelpath)
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= patience:
                break
        
            for batch_train in generator:
                step += 1
                train_loss= call_train_step(batch_train)

                if step%100 == 0:
                    with writer.as_default():
                        tf.summary.scalar("train loss", train_loss, step=step)
                        writer.flush()

                if interactive:
                    pbar.set_postfix(epoch=epoch, train_loss=train_loss.numpy(), test_loss=test_losses[-1].numpy(), refresh=False)
                    pbar.update(1)
  
    except KeyboardInterrupt:
        pass
    finally:
        #model.encoder_.save('./models/model_encoder.h5')
        #model.decoder_.save('./models/model_decoder.h5')
        #model.embeddings.

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
    input_length = 2              #update for brains
    input_channels = 65890        #update for brains
    input_duration = 2

    print('Preparing TF records')
    data_pattern="/om4/group/gablab/data/datalad/openneuro/ds000224/derivatives/surface_pipeline/sub-MSC01/processed_restingstate_timecourses/ses-func*/cifti/sub-MSC01_ses-func*_task-rest_bold_32k_fsLR_2.dtseries.nii"
    tf_folder="/om/user/abizeul/tfrecords_ds000224_rest"

    #write_cifti_tfrecords(data_pattern=data_pattern,tfrecords_folder=tf_folder,size_shard=10)

    print("Loading data")
    dataset = get_dataset(tfrecords_folder=tf_folder,batch_size=2,epoch_size=2)

    # build model
    model = SOMVAE(latent_dim=latent_dim, som_dim=som_dim,input_length=input_length,
                input_channels=input_channels, batch_size=input_duration, alpha=alpha, 
                beta=beta, gamma=gamma, tau=tau, mnist=False)

    train_model(model,generator=dataset)

    #result = evaluate_model(model,x=1)

    #if not save_model:
    #    shutil.rmtree(os.path.dirname(modelpath))
    #return result
    return 1

