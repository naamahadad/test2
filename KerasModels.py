
#from keras.models import Sequential, Graph
#from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adadelta
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.objectives import categorical_crossentropy
from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model
from keras import backend as K
from keras import objectives
import numpy as np
#from theano.compile.nanguardmode import NanGuardMode

def weighted_loss(base_loss,l):
    def loss_function(y_true, y_pred):
        return l*base_loss(y_true,y_pred)
    return loss_function
    
def add_dense_layer(model,dim,params,input_shape=None,act=LeakyReLU,alpha=0.1,dropout=True,batch_norm=True):
    if input_shape is not None:    
        model.add(Dense(dim,input_shape=input_shape,init='glorot_uniform'))
    else:
        model.add(Dense(dim,init='glorot_uniform'))
    model.add(act(alpha))
    #activation="relu"
    #model.add(act)
    if 'dropout' in params and params['dropout'] and dropout:
        model.add(Dropout(params['dropout_p']))
    if 'batch_norm' in params and params['batch_norm'] and batch_norm:
        model.add(BatchNormalization(dim))
        
def VAEenc(params,batch_size,original_dim,intermediate_dim,latent_dim):
    x = Input(batch_shape=(batch_size, original_dim))
    h = Dense(intermediate_dim, activation='relu',init='glorot_uniform')(x)
    #add_dense_layer(h,intermediate_dim,params)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    
    return x,z_mean,z_log_var
    
def VAEdec(params,in_z,x,intermediate_dim,original_dim):
    # we instantiate these layers separately so as to reuse them later
    x_decoded_mean = Dense(intermediate_dim, activation='relu',init='glorot_uniform')(in_z)
    x_decoded_log_std = Dense(intermediate_dim, activation='relu',init='glorot_uniform')(in_z)
    #decoder_mean = Dense(original_dim, activation='sigmoid',init='glorot_uniform')
    #h_decoded = decoder_h(in_z)
    #x_decoded_mean = decoder_mean(h_decoded)
    logpxz = 0.5* K.sum(x_decoded_log_std +np.square((x-x_decoded_mean))/np.exp(x_decoded_log_std),axis=-1)

    return x_decoded_mean,logpxz


def VAE(params,batch_size,original_dim,intermediate_dim,latent_dim):
    epsilon_std = 1.0
    
    print 'building enc...'
    x,z_mean,z_log_var = VAEenc(params,batch_size,original_dim,intermediate_dim,latent_dim)
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  std=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    
    print 'building dec...'
    x_decoded_mean,logpxz = VAEdec(params,z,x,intermediate_dim,original_dim)
    
    def vae_loss(x, x_decoded_mean,logpxz):
        #xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return logpxz + kl_loss
        
    print 'compile...'
    vae = Model(x, x_decoded_mean,logpxz)
    vae.compile(optimizer='rmsprop', loss=vae_loss)    
    
    return vae
    