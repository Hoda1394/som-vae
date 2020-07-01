# som-vae
Adaptation of the SOM-VAE code for fMRI generation purposes in Tensorflow v2. \
Code adapted from https://github.com/ratschlab/SOM-VAE 

As the initial code, scripts use the sacred librairy to monitor run parameters

## Create training environment

To create the conda environment necessary to train the som-vae

    conda env create -f environment.yml

## Prepare tfrecords

To create tfrecords that will be used for training the som-vae.
Data must be 2D Nifti files with shape [time_steps, sample_size]

    python3 somvae_train.py with prepare=True tf_folder='my_folder' data_pattern='my_pattern'

## Train som-vae with tfrecords

To train the som-vae with tfrecords using a certain batch size:

    python3 somvae_train.py with prepare=False tf_folder='my_folder' batch_size=2
