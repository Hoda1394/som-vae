"""
SOM-VAE model as described in https://arxiv.org/abs/1806.02199
Copyright (c) 2018
Author: Vincent Fortuin
Institution: Biomedical Informatics group, ETH Zurich
License: MIT License
"""

import functools
import numpy as np
import tensorflow as tf

loss_mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)


#def weight_variable(shape, name):
#    """Creates a TensorFlow Variable with a given shape and name and truncated normal initialization."""
#    initial = tf.random.truncated_normal(shape, stddev=0.1)
#    return tf.Variable(initial,name=name)


#def bias_variable(shape, name):
#    """Creates a TensorFlow Variable with a given shape and name and constant initialization."""
#    initial = tf.constant(0.1, shape=shape)
#    return tf.Variable(initial, name=name)


def conv2d(x, shape, name, strides=(1,1)):
    """Creates a 2D convolutional layer with weight and bias variables.
    
    Args:
        x (tf.Tensor): Input tensor.
        shape (list): Shape of the weight matrix.
        name (str): Name of the layer.
        strides (list): Strides for the convolution (default: [1,1,1,1]).
    Returns:
        tf.Tensor: The convolution defined by the weight matrix and the biases with the given strides.
    """
    #weight = weight_variable(shape, "{}_W".format(name))
    #bias = bias_variable([shape[-1]], "{}_b".format(name))
    #return tf.nn.conv2d(input=x, filters=weight, strides=strides, padding='SAME', name=name) + bias
    return tf.keras.layers.Conv2D(filters=shape[-1], kernel_size=(shape[0],shape[0]),strides=strides, padding="same",name=name,
        use_bias=True,kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0,stddev=0.1),bias_initializer=tf.constant_initializer(value=0.1))(x)


#def conv2d_transposed(x, shape, outshape, name, strides=(1,1)):
#    """Creates a transposed convolutional layer simimar to conv2d.
    
#    Args:
#        x (tf.Tensor): Input tensor.
#        shape (list): Shape of the weight matrix.
#        name (str): Name of the layer.
#        strides (list): Strides for the convolution (default: [1,1,1,1]).
#    Returns:
#        tf.Tensor: The transposed convolution defined by the weight matrix and the biases with the given strides.
#    """
#    weight = weight_variable(shape, "{}_W".format(name))
#    bias = bias_variable([shape[-2]], "{}_b".format(name))
#    return tf.nn.conv2d_transpose(x, weight, output_shape=outshape, strides=strides, padding='SAME', name=name) + bias
#    return tf.keras.layers.Conv2DTranspose(filters=shape[-1], kernel_size=(shape[0],shape[0]),strides=strides, padding="same",name=name,
#        use_bias=True,kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0,stddev=0.1),bias_initializer=tf.constant_initializer(value=0.1))(x)


def max_pool_2x2(x):
    """Creates a 2x2 max-pooling layer."""
    return tf.nn.max_pool2d(input=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#def conv1d(x, shape, name, stride=1):
#    """Creates a 1D convolutional layer with weight and bias variables.
    
#    Args:
#        x (tf.Tensor): Input tensor.
#        shape (list): Shape of the weight matrix.
#        name (str): Name of the layer.
#        stride (int): Stride for the convolution (default: 1).
#    Returns:
#        tf.Tensor: The convolution defined by the weight matrix and the biases with the given stride.
#    """
#    weight = weight_variable(shape, "{}_W".format(name))
#    bias = bias_variable([shape[-1]], "{}_b".format(name))
#    return tf.nn.conv1d(input=x, filters=weight, stride=stride, padding='SAME', name=name) + bias


#def max_pool_2x1(x):
#    """Creates a 2x1 max-pooling layer."""
#    return tf.compat.v1.layers.max_pooling1d(x, pool_size=2, strides=2, padding='SAME')


def lazy_scope(function):
    """Creates a decorator for methods that makes their return values load lazily.
    
    A method with this decorator will only compute the return value once when called
    for the first time. Afterwards, the value will be cached as an object attribute.
    Inspired by: https://danijar.com/structuring-your-tensorflow-models
    
    Args:
        function (func): Function to be decorated.
        
    Returns:
        decorator: Decorator for the function.
    """
    attribute = "_cache_" + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.compat.v1.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class SOMVAE(tf.keras.Model):
    """Class for the SOM-VAE model as described in https://arxiv.org/abs/1806.02199"""

    def __init__(self, latent_dim=64, som_dim=[8,8], learning_rate=1e-4, decay_factor=0.95, decay_steps=1000,
            input_length=28, input_channels=28, batch_size=32, alpha=1., beta=1., gamma=1., tau=1., mnist=True):
        """Initialization method for the SOM-VAE model object.
        
        Args:
            inputs (tf.Tensor): The input tensor for the model.
            latent_dim (int): The dimensionality of the latent embeddings (default: 64).
            som_dim (list): The dimensionality of the self-organizing map (default: [8,8]).
            learning_rate (float): The learning rate for the optimization (default: 1e-4).
            decay_factor (float): The factor for the learning rate decay (default: 0.95).
            decay_steps (int): The number of optimization steps before every learning rate
                decay (default: 1000).
            input_length (int): The length of the input data points (default: 28).
            input_channels (int): The number of channels of the input data points (default: 28).
            alpha (float): The weight for the commitment loss (default: 1.).
            beta (float): The weight for the SOM loss (default: 1.).
            gamma (float): The weight for the transition probability loss (default: 1.).
            tau (float): The weight for the smoothness loss (default: 1.).
            mnist (bool): Flag that tells the model if we are training in MNIST-like data (default: True).
        """
        super(SOMVAE, self).__init__()
        self.inputs = tf.Variable(tf.zeros(shape=[batch_size, input_length, input_channels, 1],dtype=tf.dtypes.int32),shape=[32,28, 28, 1])
        self.latent_dim = latent_dim
        self.som_dim = som_dim
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        self.input_length = input_length
        self.input_channels = input_channels
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.mnist = mnist
        
        # Static
        self.encoder_ = self.get_encoder()
        self.decoder_ = self.get_decoder()

        #Dynamic
        self.embeddings = self.get_embeddings() 
        self.transition_probabilities = self.get_transition_probabilities()
        self.global_step = self.get_global_step()

    #@lazy_scope
    def get_embeddings(self):
        """Creates variable for the SOM embeddings."""
        #embeddings = tf.compat.v1.get_variable("embeddings", self.som_dim+[self.latent_dim],initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.05))
        embeddings=tf.Variable(tf.initializers.TruncatedNormal(mean=0., stddev=0.05)(shape=self.som_dim+[self.latent_dim], dtype=tf.float32))
        #tf.compat.v1.summary.tensor_summary("embeddings", embeddings)
        return embeddings

    #@lazy_scope
    def get_transition_probabilities(self):
        """Creates tensor for the transition probabilities."""
        probabilities_raw = tf.Variable(tf.zeros(self.som_dim+self.som_dim), name="probabilities_raw")
        probabilities_positive = tf.exp(probabilities_raw)
        probabilities_summed = tf.reduce_sum(input_tensor=probabilities_positive, axis=[-1,-2], keepdims=True)
        probabilities_normalized = probabilities_positive / probabilities_summed
        return probabilities_normalized

    #@lazy_scope
    def get_global_step(self):
        """Creates global_step variable for the optimization."""
        global_step = tf.Variable(0, trainable=False, name="global_step")
        return global_step

    #@lazy_scope
    def get_batch_size(self):
        """Reads the batch size from the input tensor."""
        batch_size = tf.shape(input=self.inputs)[0]
        return batch_size

    def get_encoder(self):
        if not self.mnist:
            h_0 = tf.keras.layers.Input(shape=[None,self.input_length, self.input_channels, 1], name='input')
            h_1 = tf.keras.layers.Dense(256, activation="dense")(h_0)
            h_2 = tf.keras.layers.Dense(128, activation="dense")(h_1)
            z_e = tf.keras.layers.Dense(self.latent_dim, activation="relu")(h_2)

        else:
            h_0 = tf.keras.layers.Input(shape=[self.input_length, self.input_channels,1], name='input')
            h_conv1 = tf.nn.relu(conv2d(h_0, [4,4,1,256], "conv1"))
            h_pool1 = max_pool_2x2(h_conv1)
            h_conv2 = tf.nn.relu(conv2d(h_pool1, [4,4,256,256], "conv2"))
            h_pool2 = max_pool_2x2(h_conv2)
            #flat_size = 7*7*256
            #h_flat = tf.reshape(h_pool2, [-1, flat_size])
            h_flat = tf.keras.layers.Flatten()(h_pool2)
            print(self.latent_dim,h_flat,h_pool2)
            z_e = tf.keras.layers.Dense(self.latent_dim)(h_flat)
        return tf.keras.models.Model(inputs=[h_0], outputs=[z_e], name='encoder')

    def get_z_e(self):
        return self.encoder_(self.inputs)

    #@lazy_scope
    #def z_e_old(self):
    #    """Aggregates the encodings of the respective previous time steps."""
    #    z_e_old = tf.concat([self.z_e[0:1], self.z_e[:-1]], axis=0)
    #    return z_e_old

    #@lazy_scope
    def get_z_dist_flat(self):
        """Computes the distances between the encodings and the embeddings."""
        z_dist = tf.math.squared_difference(tf.expand_dims(tf.expand_dims(self.z_e, 1), 1), tf.expand_dims(self.embeddings, 0))
        z_dist_red = tf.reduce_sum(input_tensor=z_dist, axis=-1)
        z_dist_flat = tf.reshape(z_dist_red, [self.batch_size, -1])
        return z_dist_flat

    #@lazy_scope
    def get_k(self):
        """Picks the index of the closest embedding for every encoding."""
        k = tf.argmin(input=self.z_dist_flat, axis=-1)
        tf.compat.v1.summary.histogram("clusters", k)
        return k

    #@lazy_scope
    def get_z_q(self):
        """Aggregates the respective closest embedding for every encoding."""
        k_1 = self.k // self.som_dim[1]
        k_2 = self.k % self.som_dim[1]
        k_stacked = tf.stack([k_1, k_2], axis=1)
        z_q = tf.gather_nd(self.embeddings, k_stacked)
        return z_q

    #@lazy_scope
    def get_z_q_neighbors(self):
        """Aggregates the respective neighbors in the SOM for every embedding in z_q."""
        k_1 = self.k // self.som_dim[1]
        k_2 = self.k % self.som_dim[1]
        k_stacked = tf.stack([k_1, k_2], axis=1)

        k1_not_top = tf.less(k_1, tf.constant(self.som_dim[0]-1, dtype=tf.int64))
        k1_not_bottom = tf.greater(k_1, tf.constant(0, dtype=tf.int64))
        k2_not_right = tf.less(k_2, tf.constant(self.som_dim[1]-1, dtype=tf.int64))
        k2_not_left = tf.greater(k_2, tf.constant(0, dtype=tf.int64))

        k1_up = tf.compat.v1.where(k1_not_top, tf.add(k_1, 1), k_1)
        k1_down = tf.compat.v1.where(k1_not_bottom, tf.subtract(k_1, 1), k_1)
        k2_right = tf.compat.v1.where(k2_not_right, tf.add(k_2, 1), k_2)
        k2_left = tf.compat.v1.where(k2_not_left, tf.subtract(k_2, 1), k_2)

        z_q_up = tf.compat.v1.where(k1_not_top, tf.gather_nd(self.embeddings, tf.stack([k1_up, k_2], axis=1)),
                          tf.zeros([self.batch_size, self.latent_dim]))
        z_q_down = tf.compat.v1.where(k1_not_bottom, tf.gather_nd(self.embeddings, tf.stack([k1_down, k_2], axis=1)),
                          tf.zeros([self.batch_size, self.latent_dim]))
        z_q_right = tf.compat.v1.where(k2_not_right, tf.gather_nd(self.embeddings, tf.stack([k_1, k2_right], axis=1)),
                          tf.zeros([self.batch_size, self.latent_dim]))
        z_q_left = tf.compat.v1.where(k2_not_left, tf.gather_nd(self.embeddings, tf.stack([k_1, k2_left], axis=1)),
                          tf.zeros([self.batch_size, self.latent_dim]))

        z_q_neighbors = tf.stack([self.z_q, z_q_up, z_q_down, z_q_right, z_q_left], axis=1)
        return z_q_neighbors

    def get_decoder(self):
        """Reconstructs the input from the latent space"""
        if not self.mnist:
            #with tf.compat.v1.variable_scope("decoder", reuse=tf.compat.v1.AUTO_REUSE):
            h_0 = tf.keras.layers.Input(shape=self.latent_dim, name='latent_code')
            h_3 = tf.keras.layers.Dense(128, activation="relu")(h_0)
            h_4 = tf.keras.layers.Dense(256, activation="relu")(h_3)
            x_hat = tf.keras.layers.Dense(self.input_channels, activation="sigmoid")(h_4)
        else:
            #with tf.compat.v1.variable_scope("decoder", reuse=tf.compat.v1.AUTO_REUSE):
            h_0 = tf.keras.layers.Input(shape=self.latent_dim, name='latent_code')
            flat_size = 7*7*256
            h_flat_dec = tf.keras.layers.Dense(flat_size)(h_0)
            h_reshaped = tf.reshape(h_flat_dec, [-1, 7, 7, 256])
            h_unpool1 = tf.keras.layers.UpSampling2D((2,2))(h_reshaped)
            h_deconv1 = tf.nn.relu(conv2d(h_unpool1, [4,4,256,256], "deconv1"))
            h_unpool2 = tf.keras.layers.UpSampling2D((2,2))(h_deconv1)
            h_deconv2 = tf.nn.sigmoid(conv2d(h_unpool2, [4,4,256,1], "deconv2"))
            x_hat = h_deconv2
        return tf.keras.models.Model(inputs=[h_0], outputs=[x_hat], name='encoder')

    def get_reconstruction_e(self):
        return self.decoder_(self.z_e)
    
    def get_reconstruction_q(self):
        return self.decoder_(self.z_q)

    #@lazy_scope
    def loss_reconstruction(self):
        """Computes the combined reconstruction loss for both reconstructions."""
        print(self.inputs.shape,self.reconstruction_q.shape)
        loss_rec_mse_zq = loss_mse(self.inputs, self.reconstruction_q)
        loss_mse_zq = tf.math.reduce_sum(loss_rec_mse_zq,axis=[1,2])
        loss_mse_zq = tf.math.reduce_mean(loss_mse_zq,axis=[0])
        print('test',loss_mse_zq)
        loss_rec_mse_ze = loss_mse(self.inputs, self.reconstruction_e)
        loss_rec_mse = loss_rec_mse_zq + loss_rec_mse_ze
        #tf.compat.v1.summary.scalar("loss_reconstruction", loss_rec_mse)
        print('Reconst',loss_rec_mse)
        return loss_rec_mse

    #@lazy_scope
    def loss_commit(self):
        """Computes the commitment loss."""
        loss_commit = tf.reduce_mean(input_tensor=tf.math.squared_difference(self.z_e, self.z_q))
        #tf.compat.v1.summary.scalar("loss_commit", loss_commit)
        print("Commit",loss_commit)
        return loss_commit


    #@lazy_scope
    def loss_som(self):
        """Computes the SOM loss."""
        loss_som = tf.reduce_mean(input_tensor=tf.math.squared_difference(tf.expand_dims(tf.stop_gradient(self.z_e), axis=1), self.z_q_neighbors))
        #tf.compat.v1.summary.scalar("loss_som", loss_som)
        print("som",loss_som)
        return loss_som


    #@lazy_scope
    def loss_probabilities(self):
        """Computes the negative log likelihood loss for the transition probabilities."""
        k_1 = self.k // self.som_dim[1]
        k_2 = self.k % self.som_dim[1]
        k_1_old = tf.concat([k_1[0:1], k_1[:-1]], axis=0)
        k_2_old = tf.concat([k_2[0:1], k_2[:-1]], axis=0)
        k_stacked = tf.stack([k_1_old, k_2_old, k_1, k_2], axis=1)
        transitions_all = tf.gather_nd(self.transition_probabilities, k_stacked)
        loss_probabilities = -self.gamma * tf.reduce_mean(input_tensor=tf.math.log(transitions_all))
        print("proba",loss_probabilities)
        return loss_probabilities


    #@lazy_scope
    def loss_z_prob(self):
        """Computes the smoothness loss for the transitions given their probabilities."""
        k_1 = self.k // self.som_dim[1]
        k_2 = self.k % self.som_dim[1]
        k_1_old = tf.concat([k_1[0:1], k_1[:-1]], axis=0)
        k_2_old = tf.concat([k_2[0:1], k_2[:-1]], axis=0)
        k_stacked_old = tf.stack([k_1_old, k_2_old], axis=1)
        out_probabilities_old = tf.gather_nd(self.transition_probabilities, k_stacked_old)
        out_probabilities_flat = tf.reshape(out_probabilities_old, [self.batch_size, -1])
        weighted_z_dist_prob = tf.multiply(self.z_dist_flat, out_probabilities_flat)
        loss_z_prob = tf.reduce_mean(input_tensor=weighted_z_dist_prob)
        print("z prob",loss_z_prob)
        return loss_z_prob


    #@lazy_scope
    def loss(self):
        """Aggregates the loss terms into the total loss.""" 
        loss = (self.loss_reconstruction() + self.alpha*self.loss_commit() + self.beta*self.loss_som()
                + self.gamma*self.loss_probabilities() + self.tau*self.loss_z_prob())
        return loss


    @lazy_scope
    def optimize(self):
        """Optimizes the model's loss using Adam with exponential learning rate decay."""
        lr_decay = tf.compat.v1.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_factor, staircase=True)
        optimizer = tf.compat.v1.train.AdamOptimizer(lr_decay)
        train_step = optimizer.minimize(self.loss, global_step=self.global_step)
        train_step_prob = optimizer.minimize(self.loss_probabilities, global_step=self.global_step)
        #return train_step, train_step_prob

        #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay)
        #train_step = optimizer.minimize(self.loss)
        #train_step_prob = optimizer.minimize(self.loss_probabilities)

        return train_step, train_step_prob

    #@lazy_scope
    def call(self,inputs):

        self.inputs=inputs
        self.batch=self.get_batch_size()
        self.z_e = self.get_z_e()
        print('z_e',self.z_e.shape)
        self.z_dist_flat = self.get_z_dist_flat()
        print('z_dis_flat',self.z_dist_flat.shape)
        self.k = self.get_k()
        print('k',self.k.shape)
        self.z_q = self.get_z_q()
        print('z_q',self.z_q.shape)
        self.z_q_neighbors = self.get_z_q_neighbors()
        print('z_q_neig',self.z_q_neighbors.shape)
        self.reconstruction_e = self.get_reconstruction_e()
        print('z_e_rec',self.reconstruction_e.shape)
        self.reconstruction_q = self.get_reconstruction_q()
        print('z_q_rec',self.reconstruction_q.shape)

        
    

