#use python 3.5
#make sure tensorflow is running in the background

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, _),(x_test, _) = mnist.load_data()

x_train = np.reshape(x_train, newshape=(*x_train.shape, 1))
x_train = tf.image.resize_images(images=x_train, size=(14,14))
x = tf.Session().run(x_train)
x_train = np.asarray(x, dtype=np.uint8).reshape(x_train.shape[0], 196) / 255.0

x_test = np.reshape(x_test, newshape=(*x_test.shape, 1))
x_test = tf.image.resize_images(images=x_test, size=(14,14))
x = tf.Session().run(x_test)
x_test = np.asarray(x, dtype=np.uint8).reshape(x_test.shape[0], 196) / 255.0


def show_digit(x):
    x = np.reshape(x, (14,14))
    fig = plt.figure(figsize=(2, 2))
    fig.add_subplot(1,1,1)
    plt.imshow(x, cmap='gray')   
    plt.show()

class VAE():
    def __init__(self, input_dim, latent_dim, learn_rate):
        self.input_dim = input_dim
        self.learn_rate = learn_rate
        self.latent_dim = latent_dim
        self.output_dim = input_dim
        self.U = np.random.normal(0, 1, (self.latent_dim, self.input_dim))
        self.u0 = np.random.normal(0, 1, self.latent_dim)
        self.V = np.random.normal(0, 1, (self.latent_dim, self.input_dim))
        self.v0 = np.random.normal(0, 1, self.latent_dim)
        self.W = np.random.normal(0, 1, (self.output_dim, self.latent_dim))
        self.w0 = np.random.normal(0, 1, self.output_dim)
        

    def sigmoid(self, t):
        return 1/(1  + np.exp(-t))

    def dsigmoid(self, t):
        sigt = self.sigmoid(t)
        return sigt*(1-sigt)

    def tanh(self, t):
        return (2*self.sigmoid(2*t) - 1)
    
    def dtanh(self, t):
        return (1 - self.tanh(x)**2)

    def encoder(self, x):
        mean = self.tanh(np.dot(x, self.U.T) + self.u0)
        log_covar = self.tanh(np.dot(x, self.V.T) + self.v0)
        return [mean, log_covar] 

    def decoder(self, z):
        return self.sigmoid(np.dot(z, self.W.T) + self.w0)

    def reduce_loss(self, X):
        self.N = len(X)
       
        dLoss_U, dLoss_u0 = np.zeros_like(self.U), np.zeros_like(self.u0)
        dLoss_V, dLoss_v0 = np.zeros_like(self.V), np.zeros_like(self.v0)
        dLoss_W, dLoss_w0 = np.zeros_like(self.W), np.zeros_like(self.w0)
      
        [mean, log_covar] = self.encoder(X) 
        covar = np.exp(log_covar)
        e = np.random.normal(0, 1, mean.shape)
        Z = (covar* e) + mean
        Y = self.decoder(Z)
        dY = Y * (1-Y)
        dmean = 1 - mean**2
        dlog_covar = 1 - log_covar**2
        self.loss += np.linalg.norm(Y-X)**2 + 0.5*np.sum((covar - mean**2 -1 - log_covar))
        y_delta = 2* (Y-X) * dY
        mean_delta = np.multiply(np.matmul(y_delta, self.W), dmean) 
        dKL_U = np.sum(mean*dmean, axis=1)
        covar_delta = np.matmul(y_delta, self.W) * e * covar * dlog_covar
        dKL_V = 0.5 * np.sum((covar-1)*dlog_covar, axis=1)

        dLoss_W = np.matmul(y_delta.T, Z)
        dLoss_w0 = np.sum(y_delta, axis=0)
        dLoss_U = np.matmul((mean_delta.T + dKL_U) , X)
        dLoss_u0 = np.sum(mean_delta.T + dKL_U, axis=1)
        dLoss_V = np.matmul((covar_delta.T + dKL_V) , X)
        dLoss_v0 = np.sum((covar_delta.T + dKL_V), axis=1)
        self.U  -= self.learn_rate * dLoss_U
        self.u0 -= self.learn_rate * dLoss_u0
        self.V  -= self.learn_rate * dLoss_V
        self.v0 -= self.learn_rate * dLoss_v0
        self.W  -= self.learn_rate * dLoss_W
        self.w0 -= self.learn_rate * dLoss_w0
    
    
    def train(self, x_train, epochs, batch_size, shuffle=True): 
        N = len(x_train)
        epoch = 1
        while(epoch <= epochs):
            if shuffle:
                indices = np.arange(len(x_train))
                np.random.shuffle(indices)
                x_train = x_train[indices]
            self.loss = 0
            for batch in np.arange(0, N, batch_size):
                X = x_train[batch:batch+batch_size]
                self.reduce_loss(X=X)
                
            print('Epoch: ', epoch, ' Loss: ', self.loss)
           
            if epoch%5==0: 
                z = np.random.normal(0,1, self.latent_dim)
            epoch += 1
        print('Done Training')

vae = VAE(input_dim=196, latent_dim=3, learn_rate=1e-4)
vae.train(x_train=x_train, epochs=300, batch_size=600, shuffle=True)


fig=plt.figure(figsize=(8, 8))

for i in range(1, 21):
    z = np.random.normal(0, 1, vae.latent_dim)
    img = vae.decoder(z).reshape((14,14))
    fig.add_subplot(4, 5, i)
    plt.imshow(img, cmap='gray')

plt.show()

