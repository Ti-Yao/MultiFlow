# Utilities
import os
import csv
import argparse
from time import time

# Keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Reshape, UpSampling2D, Conv2DTranspose, GlobalAveragePooling1D, Softmax
from tensorflow.keras.losses import kullback_leibler_divergence
import tensorflow.keras.backend as K

# scikit-learn
from sklearn.cluster import AgglomerativeClustering, KMeans
from keras.callbacks import ModelCheckpoint

# DTC components
from TSClusteringLayer import TSClusteringLayer
from TAE import *
import numpy as np

class DTC:
    """
    Deep Temporal Clustering (DTC) model

    # Arguments
        n_clusters: number of clusters
        input_dim: input dimensionality
        timesteps: length of input sequences (can be None for variable length)
        n_filters: number of filters in convolutional layer
        kernel_size: size of kernel in convolutional layer
        strides: strides in convolutional layer
        pool_size: pooling size in max pooling layer, must divide the time series length
        n_units: numbers of units in the two BiLSTM layers
        alpha: coefficient in Student's kernel
        dist_metric: distance metric between latent sequences
        cluster_init: cluster initialization method

    """

    def __init__(self, n_clusters, input_dim, timesteps, model_name,vessel_pair, tae_version,
                 n_filters=50, kernel_size=10, strides=1, pool_size=8, n_units=[50, 1], 
                 alpha=1.0, dist_metric='eucl', cluster_init='kmeans', heatmap=False):
        assert(timesteps % pool_size == 0)
        self.n_clusters = n_clusters
        self.model_name = model_name
        self.vessel_pair = vessel_pair
        self.input_dim = input_dim
        self.tae_version = tae_version
        self.timesteps = timesteps
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.n_units = n_units
        self.latent_shape = (self.timesteps // self.pool_size, self.n_units[1])
        self.alpha = alpha
        self.dist_metric = dist_metric
        self.cluster_init = cluster_init
        self.heatmap = heatmap
        self.pretrained = False
        self.model = self.autoencoder = self.encoder = self.decoder = None
        if self.heatmap:
            self.heatmap_model = None
            self.heatmap_loss_weight = None
            self.initial_heatmap_loss_weight = None
            self.final_heatmap_loss_weight = None
            self.finetune_heatmap_at_epoch = None

    def initialize(self):
        """
        Create DTC model
        """
        # Create AE models
        if self.tae_version == 1:
            tae = temporal_autoencoder
        if self.tae_version == 2:
            tae = temporal_autoencoder_v2

        self.autoencoder, self.encoder, self.decoder = tae(input_dim=self.input_dim,
                                                                            timesteps=self.timesteps,
                                                                            n_filters=self.n_filters,
                                                                            kernel_size=self.kernel_size,
                                                                            strides=self.strides,
                                                                            pool_size=self.pool_size,
                                                                            n_units=self.n_units)
        clustering_layer = TSClusteringLayer(self.n_clusters,
                                             alpha=self.alpha,
                                             dist_metric=self.dist_metric,
                                             name='TSClustering')(self.encoder.output)

        self.model = Model(inputs=self.autoencoder.input,
                               outputs=[self.autoencoder.output, clustering_layer])

    @property
    def cluster_centers_(self):
        """
        Returns cluster centers
        """
        return self.model.get_layer(name='TSClustering').get_weights()[0]

    @staticmethod
    def weighted_kld(loss_weight):
        """
        Custom KL-divergence loss with a variable weight parameter
        """
        def loss(y_true, y_pred):
            return loss_weight * kullback_leibler_divergence(y_true, y_pred)
        return loss

    def on_epoch_end(self, epoch):
        """
        Update heatmap loss weight on epoch end
        """
        if epoch > self.finetune_heatmap_at_epoch:
            K.set_value(self.heatmap_loss_weight, self.final_heatmap_loss_weight)

    def compile(self, gamma, optimizer, initial_heatmap_loss_weight=None, final_heatmap_loss_weight=None):
        """
        Compile DTC model

        # Arguments
            gamma: coefficient of TS clustering loss
            optimizer: optimization algorithm
            initial_heatmap_loss_weight (optional): initial weight of heatmap loss vs clustering loss
            final_heatmap_loss_weight (optional): final weight of heatmap loss vs clustering loss (heatmap finetuning)
        """
        if self.heatmap:
            self.initial_heatmap_loss_weight = initial_heatmap_loss_weight
            self.final_heatmap_loss_weight = final_heatmap_loss_weight
            self.heatmap_loss_weight = K.variable(self.initial_heatmap_loss_weight)
            self.model.compile(loss=['mse', DTC.weighted_kld(1.0 - self.heatmap_loss_weight), DTC.weighted_kld(self.heatmap_loss_weight)],
                               loss_weights=[1.0, gamma, gamma],
                               optimizer=optimizer)
        else:
            self.model.compile(loss=['mse', 'kld'],
                               loss_weights=[gamma, (1-gamma)],
                               optimizer=optimizer)

    def load_weights(self, weights_path):
        """
        Load pre-trained weights of DTC model

        # Arguments
            weight_path: path to weights file (.h5)
        """
        self.model.load_weights(weights_path)
        self.pretrained = True

    def load_ae_weights(self, ae_weights_path):
        """
        Load pre-trained weights of AE

        # Arguments
            ae_weight_path: path to weights file (.h5)
        """
        self.autoencoder.load_weights(ae_weights_path)
        self.pretrained = True

    def dist(self, x1, x2):
        """
        Compute distance between two multivariate time series using chosen distance metric

        # Arguments
            x1: first input (np array)
            x2: second input (np array)
        # Return
            distance
        """
        if self.dist_metric == 'eucl':
            return tsdistances.eucl(x1, x2)
        elif self.dist_metric == 'cid':
            return tsdistances.cid(x1, x2)
        elif self.dist_metric == 'cor':
            return tsdistances.cor(x1, x2)
        elif self.dist_metric == 'acf':
            return tsdistances.acf(x1, x2)
        else:
            raise ValueError('Available distances are eucl, cid, cor and acf!')

    def init_cluster_weights(self, X):
        """
        Initialize with complete-linkage hierarchical clustering or k-means.

        # Arguments
            X: numpy array containing training set or batch
        """
        assert(self.cluster_init in ['hierarchical', 'kmeans'])
        print('Initializing cluster...')

        features = self.encode(X)

        if self.cluster_init == 'hierarchical':
            if self.dist_metric == 'eucl':  # use AgglomerativeClustering off-the-shelf
                hc = AgglomerativeClustering(n_clusters=self.n_clusters,
                                             affinity='euclidean',
                                             linkage='complete').fit(features.reshape(features.shape[0], -1))
            else:  # compute distance matrix using dist
                d = np.zeros((features.shape[0], features.shape[0]))
                for i in range(features.shape[0]):
                    for j in range(i):
                        d[i, j] = d[j, i] = self.dist(features[i], features[j])
                hc = AgglomerativeClustering(n_clusters=self.n_clusters,
                                             affinity='precomputed',
                                             linkage='complete').fit(d)
            # compute centroid
            cluster_centers = np.array([features[hc.labels_ == c].mean(axis=0) for c in range(self.n_clusters)])
        elif self.cluster_init == 'kmeans':
            # fit k-means on flattened features
            km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state = 42).fit(features.reshape(features.shape[0], -1))
            cluster_centers = km.cluster_centers_.reshape(self.n_clusters, features.shape[1], features.shape[2])

        self.model.get_layer(name='TSClustering').set_weights([cluster_centers])
        print('Done!')

    def encode(self, x):
        """
        Encoding function. Extract latent features from hidden layer

        # Arguments
            x: data point
        # Return
            encoded (latent) data point
        """
        return self.encoder.predict(x)

    def decode(self, x):
        """
        Decoding function. Decodes encoded sequence from latent space.

        # Arguments
            x: encoded (latent) data point
        # Return
            decoded data point
        """
        return self.decoder.predict(x)

    def predict(self, x):
        """
        Predict cluster assignment.

        """
        q = self.model.predict(x, verbose=0)[1]
        return q.argmax(axis=1)

    @staticmethod
    def target_distribution(q):  # target distribution p which enhances the discrimination of soft label q
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def predict_heatmap(self, x):
        """
        Produces TS clustering heatmap from input sequence.

        # Arguments
            x: data point
        # Return
            heatmap
        """
        return self.heatmap_model.predict(x, verbose=0)

    def pretrain(self, X, 
                 optimizer='adam',
                 epochs=10,
                 batch_size=64,
                 learning_rate = 1e-3,
                 save_dir='models',
                 model_name = '',
                 verbose=1):
        """
        Pre-train the autoencoder using only MSE reconstruction loss
        Saves weights in h5 format.

        # Arguments
            X: training set
            optimizer: optimization algorithm
            epochs: number of pre-training epochs
            batch_size: training batch size
            save_dir: path to existing directory where weights will be saved
        """
        print('Pretraining...')
        
        mc = ModelCheckpoint(f'models/{model_name}_ae.h5',
                        save_best_only= True,
                            monitor='loss',
                            mode='min')

        self.autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate), loss='mse')

        # Begin pretraining
        t0 = time()
        self.autoencoder.fit(X, X, 
                             batch_size=batch_size, 
                             epochs=epochs, 
                             verbose=verbose, 
                             callbacks=[mc])

        print('Pretraining time: ', time() - t0)
        self.pretrained = True

    def fit(self, X_train, y_train=None,
            X_val=None, y_val=None,
            epochs=100,
            eval_epochs=10,
            save_epochs=100,
            batch_size=64,
            tol=0.001,
            patience=10,
            finetune_heatmap_at_epoch=8,
            save_dir='models'):
        """
        Training procedure

        # Arguments
           X_train: training set
           y_train: (optional) training labels
           X_val: (optional) validation set
           y_val: (optional) validation labels
           epochs: number of training epochs
           eval_epochs: evaluate metrics on train/val set every eval_epochs epochs
           save_epochs: save model weights every save_epochs epochs
           batch_size: training batch size
           tol: tolerance for stopping criterion
           patience: patience for stopping criterion
           finetune_heatmap_at_epoch: epoch number where heatmap finetuning will start. Heatmap loss weight will
                                      switch from `self.initial_heatmap_loss_weight` to `self.final_heatmap_loss_weight`
           save_dir: path to existing directory where weights and logs are saved
        """
        if not self.pretrained:
            print('Autoencoder was not pre-trained!')


        y_pred_last = None
        patience_cnt = 0
        loss_min = 100
        loss_recon_min = 100


        print('Training for {} epochs.\nEvaluating every {} and saving model every {} epochs.'.format(epochs, eval_epochs, save_epochs))

        for epoch in range(epochs):

            # Compute cluster assignments for training set
            q = self.model.predict(X_train)[1]
            p = DTC.target_distribution(q)

            # Evaluate losses and metrics on training set
            if epoch % eval_epochs == 0:

                y_pred = q.argmax(axis=1)
                print('epoch {}'.format(epoch))
                loss = self.model.evaluate(X_train, [X_train, p], batch_size=batch_size, verbose=False)
                L, Lr, Lc = loss

                if L<loss_min:
                    loss_min = L
                    self.model.save_weights(f'{save_dir}/{self.model_name}_{self.vessel_pair}.h5')
                    print('Saved model to:',f'{save_dir}/{self.model_name}_{self.vessel_pair}.h5')
               
                print(f'[Train] - Lr={Lr:f}, Lc={Lc:f} - total loss={L:f}')

                if y_pred_last is not None:
                    assignment_changes = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred

                if epoch > 50 and assignment_changes < tol:
                    patience_cnt += 1
                    print(f'Assignment changes {assignment_changes} < {tol} tolerance threshold. Patience: {patience_cnt}/{patience}')
                 
                    # if Lr == loss_recon_min:
                    #     patience_cnt = 0

                    if patience_cnt >= patience:
                        print('Reached max patience. Stopping training.')
                        self.model.save_weights(f'{save_dir}/{self.model_name}_{self.vessel_pair}.h5')
                        print('Saved model to:',f'{save_dir}/{self.model_name}_{self.vessel_pair}.h5')
                        break
                    
                else:
                    patience_cnt = 0

                
            # Train for one epoch
            self.model.fit(X_train, [X_train, p], epochs=1, batch_size=batch_size, verbose=False)
