#!/usr/bin/env python
# coding: utf-8

# # **Conditional Variational autoencoder (VAE) - Toy datasets**

# # **Utility functions**

# In[1]:

import wandb
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
print(tf. __version__)


# In[2]:


import datetime, os


# In[3]:


K.clear_session()


# In[4]:


def find_indices(lst, condition):
    return np.array([i for i, elem in enumerate(lst) if condition(elem)])
    
def plot_2d_data_categorical(data_2d, y, titles=None, figsize = (7, 7), category_count=10):
  fig, axs = plt.subplots(category_count, len(data_2d), figsize = figsize)
  colors = np.array(['#7FFFD4', '#458B74', '#0000CD', '#EE3B3B', '#7AC5CD', '#66CD00',
         '#EE7621', '#3D59AB', '#CD950C', '#483D8B'])
  for i in range(len(data_2d)):
      for k in range(category_count):

        index = find_indices(y[i], lambda e: e == k)

        data_2d_k = data_2d[i][index, ]
        y_k = y[i][index]

        if (titles != None):
          axs[k,i].set_title("{} - Class: {}".format(titles[i], k))

        scatter = axs[k, i].scatter(data_2d_k[:, 0], data_2d_k[:, 1],
                                s=1, c=colors[k], cmap=plt.cm.Paired)
        axs[k, i].legend(*scatter.legend_elements())
        axs[k, i].set_xlim([-3, 3])
        axs[k, i].set_ylim([-3, 3])
        wandb.log({"Embdedding_classes": wandb.Image(plt)})
        
        
def plot_2d_data(data_2d, y, titles=None, figsize = (7, 7)):
  _, axs = plt.subplots(1, len(data_2d), figsize = figsize)

  for i in range(len(data_2d)):
    
    if (titles != None):
      axs[i].set_title(titles[i])
    scatter=axs[i].scatter(data_2d[i][:, 0], data_2d[i][:, 1],
                            s=1, c=y[i], cmap=plt.cm.Paired)
    axs[i].legend(*scatter.legend_elements())
    wandb.log({"Embdedding": wandb.Image(plt)})

def plot_history(history,metric=None):
  fig, ax1 = plt.subplots(figsize=(10, 8))

  epoch_count=len(history.history['loss'])

  line1,=ax1.plot(range(1,epoch_count+1),history.history['loss'],
                  label='train_loss',color='orange')
  ax1.plot(range(1,epoch_count+1),history.history['val_loss'],
                  label='val_loss',color = line1.get_color(), linestyle = '--')
  ax1.set_xlim([1,epoch_count])
  ax1.set_ylim([0, max(max(history.history['loss']),
              max(history.history['val_loss']))])
  ax1.set_ylabel('loss',color = line1.get_color())
  ax1.tick_params(axis='y', labelcolor=line1.get_color())
  ax1.set_xlabel('Epochs')
  _=ax1.legend(loc='lower left')

  if (metric!=None):
    ax2 = ax1.twinx()
    line2,=ax2.plot(range(1,epoch_count+1),history.history[metric],
                    label='train_'+metric)
    ax2.plot(range(1,epoch_count+1),history.history['val_'+metric],
                    label='val_'+metric,color = line2.get_color(),
                    linestyle = '--')
    ax2.set_ylim([0, max(max(history.history[metric]),
                max(history.history['val_'+metric]))])
    ax2.set_ylabel(metric,color=line2.get_color())
    ax2.tick_params(axis='y', labelcolor=line2.get_color())
    _=ax2.legend(loc='upper right')
  

def plot_generated_images(generated_images, nrows, ncols,
                          no_space_between_plots=False, figsize=(10, 10)):
  _, axs = plt.subplots(nrows, ncols,figsize=figsize,squeeze=False)

  for i in range(nrows):
    for j in range(ncols):
      axs[i,j].axis('off')
      axs[i,j].imshow(generated_images[i][j], cmap='gray')

  if no_space_between_plots:
    plt.subplots_adjust(wspace=0,hspace=0)
  
  plt.show()


# In[5]:


def conditional_input(self, inputs, label_size=10):
    image_size = [self.shape[0], self.shape[1], self.shape[2]]
    input_img = layers.InputLayer(input_shape=image_size,
                                dtype ='float32')(inputs[0])
    input_label = layers.InputLayer(input_shape=(label_size, ),
                                    dtype ='float32')(inputs[1])
    labels = tf.reshape(inputs[1], [-1, 1, 1, label_size])
    labels = tf.cast(labels, dtype='float32')
    ones = tf.ones([inputs[0].shape[0]] + image_size[0:-1] + [label_size])
    labels = ones * labels
    conditional_input = layers.Concatenate(axis=3)([input_img, labels]) 
    return  input_img, input_label, conditional_input


# In[6]:


def sampling(z_mean, z_log_var, input_label):

    eps = tf.random.normal(tf.shape(z_log_var), dtype=tf.float32,
                            mean=0., stddev=1.0, name='epsilon')
    z = z_mean + tf.exp(z_log_var / 2) * eps
    z_cond = tf.concat([z, input_label], axis=1) 
    return z_cond


# In[7]:


def Train_Val_Plot(loss, val_loss, reconstruction_loss, val_reconstruction_loss, kl_loss, val_kl_loss):

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize= (16,4))
    fig.suptitle(" MODEL'S METRICS VISUALIZATION ")

    ax1.plot(range(1, len(loss) + 1), loss)
    ax1.plot(range(1, len(val_loss) + 1), val_loss)
    ax1.set_title('History of Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(['training', 'validation'])

    ax2.plot(range(1, len(reconstruction_loss) + 1), reconstruction_loss)
    ax2.plot(range(1, len(val_reconstruction_loss) + 1), val_reconstruction_loss)
    ax2.set_title('History of reconstruction_loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('reconstruction_loss')
    ax2.legend(['training', 'validation'])
    
    ax3.plot(range(1, len(kl_loss) + 1), kl_loss)
    ax3.plot(range(1, len(val_kl_loss) + 1), val_kl_loss)
    ax3.set_title(' History of kl_loss')
    ax3.set_xlabel(' Epochs ')
    ax3.set_ylabel('kl_loss')
    ax3.legend(['training', 'validation'])
    wandb.log({"Training": wandb.Image(plt)})
    plt.show()


    


# # **Data import and manipulation**

# In[8]:


dataset_name = "cifar10"
category_count=10 

(train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()

print('Train data flatten shape: ',train_x.shape)
print('Train label shape: ',train_y.shape)
print('Test data flatten shape: ',test_x.shape)
print('Test label shape: ',test_y.shape)


# In[9]:


labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
         "horse", "ship", "truck"]


# In[10]:


val_size=10000

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y,
                            test_size = val_size,random_state = 1,shuffle=True)

print('Train data flatten shape: ',train_x.shape)
print('Train label shape: ',train_y.shape)
print('Validation data flatten shape: ',val_x.shape)
print('Validation label shape: ',val_y.shape)


# In[11]:


if len(train_x.shape) == 3:
    train_x=np.expand_dims(train_x,axis=3)
    val_x=np.expand_dims(val_x,axis=3)
    test_x=np.expand_dims(test_x,axis=3)
    print('Train shape: ',train_x.shape)
    print('Validation shape: ',val_x.shape)
    print('Test shape: ',test_x.shape)


# In[12]:


input_shape = train_x.shape[1:]
input_shape


# In[13]:


train_x = train_x/255.0
val_x = val_x/255.0
test_x = test_x/255.0

print('Min value: ',train_x.min())
print('Max value: ',train_x.max())


# In[14]:


original_image_shape=(train_x.shape[1], train_x.shape[2])

train_x_flatten=np.reshape(train_x,(train_x.shape[0],-1))
val_x_flatten=np.reshape(val_x,(val_x.shape[0],-1))
test_x_flatten=np.reshape(test_x,(test_x.shape[0],-1))

print('Train data flatten shape: ',train_x_flatten.shape)
print('Validation data flatten shape: ',val_x_flatten.shape)
print('Test data flatten shape: ',test_x_flatten.shape)


# In[15]:


train_y_one_hot = to_categorical(train_y,category_count)
val_y_one_hot=to_categorical(val_y,category_count)
test_y_one_hot=to_categorical(test_y,category_count)

print('Train label one hot encoding shape: ',train_y_one_hot.shape)
print('Validation label one hot encoding shape: ',val_y_one_hot.shape)
print('Test label one hot encoding shape: ',test_y_one_hot.shape)


# # **CVAE model**
# Creating a CVAE class and plugging encoder and decoder

# In[16]:


wandb.init(project="HistoDL", entity="nrderus",
  config = {
  "dataset": "cifar10",
  "model": "CVAE",
  "encoded_dim": 128,
  "kl_coefficient": 0.05,
  "learning_rate": 0.0001,
  "epochs": 100,
  "batch_size": 100,
  "patience": 10,
  
})

encoded_dim = wandb.config.encoded_dim


# In[17]:


#relu brings a lot of activation values = 0, leaky seems better
# https://towardsdatascience.com/the-dying-relu-problem-clearly-explained-42d0c54e0d24

def bn_relu(inputs):
    bn = layers.BatchNormalization()(inputs)
    relu = layers.LeakyReLU(0.2)(bn)
    return(relu)


# In[18]:


def residual_block(x, filters: int, kernel_size: int = 3):
    y = layers.Conv2D(kernel_size=kernel_size,
               strides= 1,
               filters=filters,
               padding="same")(x)
    y = bn_relu(y)
    y = layers.Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    out = layers.Concatenate()([x, y])
    out = bn_relu(out)
    return out


# In[19]:


def encoder3( input_shape = (28, 28, 1),  label_size=10, encoded_dim = 2): 

    inputs = layers.Input(shape=(input_shape[0],
                input_shape[1], input_shape[2] + label_size), dtype='float32',
                name='Input')
    #inputs = layers.Input(shape = input_shape)
    #labels_inputs = layers.Input(shape = (50, 50, 2))
    #encoder_inputs = layers.Concatenate()([inputs, labels_inputs])
    #block 1
    x = layers.Conv2D(16, (3, 3),
                      padding='same',
                      name='block1_conv1')(inputs)
    x = layers.Conv2D(16, (3, 3),
                      padding='same',
                      name='block1_conv2')(x)
    x = bn_relu(x)
    # block 2
    x = layers.Conv2D(32, (3, 3),
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(32, (3, 3),
                      padding='same',
                      name='block2_conv2')(x)

    x = bn_relu(x)
    x = layers.MaxPool2D(pool_size=2, strides=2,name='S4')(x)

    # block 3
    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(64, (3, 3),
                padding='same',
                name='block3_conv2')(x)    
    x = bn_relu(x)            
    x = layers.Flatten()(x)
    y = layers.Dense(encoded_dim * 2 )(x)
    mu = layers.Dense(encoded_dim, name='mu')(y)
    log_var = layers.Dense(encoded_dim, name='log_var')(y)
    

    model = keras.Model(inputs, [mu, log_var], name='encoder')
    
    return model


# In[20]:


def encoder5( input_shape = (28, 28, 1),  label_size=10, encoded_dim = 2): 

    inputs = layers.Input(shape=(input_shape[0],
            input_shape[1], input_shape[2] + label_size), dtype='float32',name='Input')
    #inputs = layers.Input(shape = input_shape)
    #labels_inputs = layers.Input(shape = (50, 50, 2))
    #encoder_inputs = layers.Concatenate()([inputs, labels_inputs])
    #block 1
    x = layers.Conv2D(16, (3, 3),
                      padding='same',
                      name='block1_conv1')(inputs)
    x = layers.Conv2D(16, (3, 3),
                      padding='same',
                      name='block1_conv2')(x)
    x = bn_relu(x)
    # block 2
    x = residual_block(x, 32)


    x = layers.MaxPool2D(pool_size=2, strides=2,name='S4')(x)

    # block 3
    x = residual_block(x, 64)         
    x = layers.Flatten()(x)
    y = layers.Dense(encoded_dim * 2 )(x)
    mu = layers.Dense(encoded_dim, name='mu')(y)
    log_var = layers.Dense(encoded_dim, name='log_var')(y)
    

    model = keras.Model(inputs, [mu, log_var], name='encoder')
    
    return model


# In[21]:


def decoder5(input_shape, encoded_dim = 2,label_size=10): 

    decoder_inputs = layers.Input(shape=(encoded_dim + label_size,) , name='decoder_input')
    x = layers.Dense(encoded_dim)
    x = layers.Dense(encoded_dim * 2 )
    x = layers.Dense(input_shape[0]/2 * input_shape[1]/2 *64)(decoder_inputs)
    x = layers.Reshape(target_shape=(int(input_shape[0]/2),
                     int(input_shape[1]/2), 64))(x)
    x = residual_block(x, 64) 
    # block 2
    x = residual_block(x, 32) 

    x = layers.UpSampling2D()(x)
    
    # block 3
    x = residual_block(x, 16) 
                        
    outputs = layers.Conv2DTranspose(filters=input_shape[-1], kernel_size=2,
                             strides=1, activation='sigmoid',padding='same')(x)

    model = keras.Model(decoder_inputs, outputs, name='decoder')
    return model


# In[22]:


def encoder4( input_shape = (28, 28, 1),  label_size=10, encoded_dim = 2): 

    inputs = layers.Input(shape=(input_shape[0],
            input_shape[1], input_shape[2] + label_size), dtype='float32',name='Input')
    #inputs = layers.Input(shape = input_shape)
    #labels_inputs = layers.Input(shape = (50, 50, 2))
    #encoder_inputs = layers.Concatenate()([inputs, labels_inputs])
    #block 1
    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      name='block1_conv1')(inputs)
    x = layers.Conv2D(64, (3, 3),
                        padding='same',
                        name='block1_conv2')(x)

    x = bn_relu(x)
    # block 2
    x = layers.Conv2D(128, (3, 3),
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      padding='same',
                      name='block2_conv2')(x)
    x = bn_relu(x)   
    x = layers.MaxPool2D(pool_size=2, strides=2,name='S4')(x)
    
    
    # block 3
    x = layers.Conv2D(256, (3, 3),
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
        padding='same',
        name='block3_conv2')(x)

    x = bn_relu(x)   

    x = layers.Conv2D(512, (3, 3),
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      padding='same',
                      name='block4_conv2')(x)
    x = bn_relu(x)          
    #x = layers.Conv2D(filters=5, kernel_size=5,strides=1,padding='same')(x)
    x = layers.Flatten()(x)
    
    y = layers.Dense(encoded_dim * 2 )(x)
    mu = layers.Dense(encoded_dim, name='mu')(y)
    log_var = layers.Dense(encoded_dim, name='log_var')(y)
    

    model = keras.Model(inputs, [mu, log_var], name='encoder')
    
    return model


# In[23]:


def decoder4(input_shape, encoded_dim = 2,label_size=10): 

    decoder_inputs = layers.Input(shape=(encoded_dim + label_size,) , name='decoder_input')
    x = layers.Dense(encoded_dim)
    x = layers.Dense(encoded_dim * 2 )
    x = layers.Dense(input_shape[0]/2 * input_shape[1]/2 * 512)(decoder_inputs)
    x = layers.Reshape(target_shape=(int(input_shape[0]/2),
                     int(input_shape[1]/2), 512))(x)
    x = bn_relu(x) 
    x = layers.Conv2DTranspose(512, (3, 3),
                padding='same',
                name='up_block0_conv1')(x)
    x = bn_relu(x) 

    x = layers.Conv2DTranspose(256, (3, 3),
                      padding='same',
                      name='up_block1_conv1')(x)

    x = bn_relu(x) 
    # block 2
    x = layers.Conv2DTranspose(128, (3, 3),
                      padding='same',
                      name='up_block5_conv1')(x)

    x = bn_relu(x) 
    x = layers.UpSampling2D()(x)
    
    # block 3
    x = layers.Conv2DTranspose(64, (3, 3),
                      padding='same',
                      name='up_block6_conv1')(x)

    x = bn_relu(x)                                
    outputs = layers.Conv2DTranspose(filters=input_shape[-1], kernel_size=2,
                             strides=1, activation='sigmoid',padding='same')(x)

    model = keras.Model(decoder_inputs, outputs, name='decoder')
    return model


# In[24]:


def decoder3(input_shape, encoded_dim = 2,label_size=10): 

    decoder_inputs = layers.Input(shape=(encoded_dim + label_size,),
                                 name='decoder_input')
    x = layers.Dense(encoded_dim)

    x = layers.Dense(encoded_dim * 2 )
 
    x = layers.Dense(input_shape[0]/2 * input_shape[1]/2 *64)(decoder_inputs)
   
    x = layers.Reshape(target_shape=(int(input_shape[0]/2),
                     int(input_shape[1]/2), 64))(x)
    x = bn_relu(x) 
    x = layers.Conv2DTranspose(64, (3, 3),
                      padding='same',
                      name='up_block4_conv1')(x)
    x = layers.Conv2DTranspose(64, (3, 3),
                    padding='same',
                    name='up_block4_conv2')(x)  
    x = bn_relu(x) 
    # block 2
    x = layers.Conv2DTranspose(32, (3, 3),
                      padding='same',
                      name='up_block5_conv1')(x)
    x = layers.Conv2DTranspose(32, (3, 3),
                      padding='same',
                      name='up_block5_conv2')(x)
    x = bn_relu(x) 
    x = layers.UpSampling2D()(x)
    
    # block 3
    x = layers.Conv2DTranspose(16, (3, 3),
                      padding='same',
                      name='up_block6_conv1')(x)

    x = layers.Conv2DTranspose(16, (3, 3),
                    padding='same',
                    name='up_block6_conv2')(x)
    x = bn_relu(x)                                
    outputs = layers.Conv2DTranspose(filters=input_shape[-1], kernel_size=2,
                             strides=1, activation='sigmoid',padding='same')(x)

    model = keras.Model(decoder_inputs, outputs, name='decoder')
    return model


# In[25]:


def encoder2( input_shape = (28, 28, 1),  label_size=10, encoded_dim = 2): 

    inputs = layers.Input(shape=(input_shape[0],
            input_shape[1], input_shape[2] + label_size), dtype='float32',name='Input')
    #inputs = layers.Input(shape = input_shape)
    #labels_inputs = layers.Input(shape = (50, 50, 2))
    #encoder_inputs = layers.Concatenate()([inputs, labels_inputs])
   #block 0
    x = layers.Conv2D(16, (3, 3),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer='HeNormal',
                      name='block0')(inputs)
    x = layers.BatchNormalization()(x)                  
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(16, (3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='HeNormal',
                      name='block0_1')(x)
    x = layers.BatchNormalization()(x)                  
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(16, (3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='HeNormal',
                      name='block0_2')(x)
    x = layers.BatchNormalization()(x)                  
    x = layers.LeakyReLU(0.2)(x)


    #block 1
    x = layers.Conv2D(32, (3, 3),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer='HeNormal',
                      name='block1')(x)
    x = layers.BatchNormalization()(x)                  
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(32, (3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='HeNormal',
                      name='block1_1')(x)
    x = layers.BatchNormalization()(x)                  
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(32, (3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='HeNormal',
                      name='block1_2')(x)
    x = layers.BatchNormalization()(x)                  
    x = layers.LeakyReLU(0.2)(x)

    # block 2
    x = layers.Conv2D(64, (3, 3),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer='HeNormal',
                      name='block2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(64, (3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='HeNormal',
                      name='block2_1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(64, (3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='HeNormal',
                      name='block2_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # block 3
    x = layers.Conv2D(128, (3, 3),
                      strides=(2, 2),                      
                      padding='same',
                      kernel_initializer='HeNormal',
                      name='block3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, (3, 3),
                        strides=(1, 1),                      
                        padding='same',
                        kernel_initializer='HeNormal',
                        name='block3_1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, (3, 3),
                        strides=(1, 1),                      
                        padding='same',
                        kernel_initializer='HeNormal',
                        name='block3_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
     # block 4
    x = layers.Conv2D(256, (3, 3),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer='HeNormal',
                      name='block4')(x)
    x = layers.Conv2D(256, (3, 3),
                strides=(1, 1),
                padding='same',
                kernel_initializer='HeNormal',
                name='block4_1')(x)

    x = layers.Conv2D(256, (3, 3),
                strides=(1, 1),
                padding='same',
                kernel_initializer='HeNormal',
                name='block4_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
               
   # x = layers.Conv2D(filters=5, kernel_size=5, strides=1,padding='same')(x)
    x = layers.Flatten()(x)
    y = layers.Dense(encoded_dim * 2)(x)
    mu = layers.Dense(encoded_dim, name='mu')(y)
    log_var = layers.Dense(encoded_dim, name='log_var')(y)
    

    model = keras.Model(inputs, [mu, log_var], name='encoder')
    
    return model


# In[26]:


def decoder2(input_shape, encoded_dim = 2,label_size=10): 

    decoder_inputs = layers.Input(shape=(encoded_dim + label_size,) , name='decoder_input')
    
    x = layers.Dense(encoded_dim * 2)(decoder_inputs)

    x = layers.Dense(int(input_shape[0]/16 * input_shape[1]/16 * 256))(x)

    x = layers.Reshape(target_shape=(int(input_shape[0]/16),
                     int(input_shape[1]/16), 256))(x)
                     
        #block 1 up
    x = layers.Conv2DTranspose(256, (3, 3),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer='HeNormal',
                      name='block1_up')(x)
    x = layers.BatchNormalization()(x)                  
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(256, (3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='HeNormal',
                      name='block1_1_up')(x)
    x = layers.BatchNormalization()(x)                  
    x = layers.LeakyReLU(0.2)(x)
    

        #block 2 up
    x = layers.Conv2DTranspose(128, (3, 3),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer='HeNormal',
                      name='block2_up')(x)
    x = layers.BatchNormalization()(x)                  
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(128, (3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='HeNormal',
                    name='block2_1_up')(x)
    x = layers.BatchNormalization()(x)                  
    x = layers.LeakyReLU(0.2)(x)

            #block 3 up
    x = layers.Conv2DTranspose(64, (3, 3),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer='HeNormal',
                      name='block3_up')(x)
    x = layers.BatchNormalization()(x)                  
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(64, (3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='HeNormal',
                    name='block3_1_up')(x)
    x = layers.BatchNormalization()(x)                  
    x = layers.LeakyReLU(0.2)(x)

             #block 4 up
    x = layers.Conv2DTranspose(32, (3, 3),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer='HeNormal',
                      name='block4_up')(x)
    x = layers.BatchNormalization()(x)                  
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(32, (3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='HeNormal',
                    name='block4_1_up')(x)
    x = layers.BatchNormalization()(x)                  
    x = layers.LeakyReLU(0.2)(x)

                 #block 5 up
    x = layers.Conv2DTranspose(16, (3, 3),
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='HeNormal',
                      name='block5_up')(x)
    x = layers.BatchNormalization()(x)                  
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2DTranspose(16, (3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='HeNormal',
                    name='block5_1_up')(x)
    x = layers.BatchNormalization()(x)                  
    x = layers.LeakyReLU(0.2)(x)
                                   
    outputs = layers.Conv2DTranspose(filters=input_shape[-1], kernel_size=1,
                             strides=1, activation='sigmoid',padding='same')(x)

    model = keras.Model(decoder_inputs, outputs, name='decoder')
    return model


# In[27]:


def encoder( input_shape = (28, 28, 1),  label_size=10, encoded_dim = 2): 

    inputs = layers.Input(shape=(input_shape[0],
            input_shape[1], input_shape[2] + label_size), dtype='float32',name='Input')
    #inputs = layers.Input(shape = input_shape)
    #labels_inputs = layers.Input(shape = (50, 50, 2))
    #encoder_inputs = layers.Concatenate()([inputs, labels_inputs])


    #block 1
    x = layers.Conv2D(16, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(inputs)

    x = layers.Conv2D(16, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.BatchNormalization()(x)
    # block 2
    x = layers.Conv2D(32, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(32, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPool2D(pool_size=2, strides=2,name='S4')(x)
    
    x = layers.BatchNormalization()(x)
    
    # block 3
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(64, (3, 3),
                activation='relu',
                padding='same',
                name='block3_conv2')(x)    
                    
    x = layers.Conv2D(filters=5, kernel_size=5,strides=1,padding='same')(x)
    x = layers.Flatten()(x)
    y = layers.Dense(encoded_dim * 2)(x)
    mu = layers.Dense(encoded_dim, name='mu')(y)
    log_var = layers.Dense(encoded_dim, name='log_var')(y)
    

    model = keras.Model(inputs, [mu, log_var], name='encoder')
    
    return model


# In[28]:


def decoder(input_shape, encoded_dim = 2,label_size=10): 
    #add normalization layers???
    decoder_inputs = layers.Input(shape=(encoded_dim + label_size,) , name='decoder_input')
    x = layers.Dense(encoded_dim)
    x = layers.Dense(encoded_dim * 2)
    x = layers.Dense(input_shape[0]/2 * input_shape[1]/2 *64)(decoder_inputs)
    x = layers.Reshape(target_shape=(int(input_shape[0]/2),
                     int(input_shape[1]/2), 64))(x)
    x = layers.Conv2DTranspose(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='up_block4_conv1')(x)
    x = layers.Conv2DTranspose(64, (3, 3),
                    activation='relu',
                    padding='same',
                    name='up_block4_conv2')(x)  
    
    # block 2
    x = layers.Conv2DTranspose(32, (3, 3),
                      activation='relu',
                      padding='same',
                      name='up_block5_conv1')(x)
    x = layers.Conv2DTranspose(32, (3, 3),
                      activation='relu',
                      padding='same',
                      name='up_block5_conv2')(x)
    x = layers.UpSampling2D()(x)
    
    # block 3
    x = layers.Conv2DTranspose(16, (3, 3),
                      activation='relu',
                      padding='same',
                      name='up_block6_conv1')(x)

    x = layers.Conv2DTranspose(16, (3, 3),
                    activation='relu',
                    padding='same',
                    name='up_block6_conv2')(x)
                                   
    outputs = layers.Conv2DTranspose(filters=input_shape[-1], kernel_size=2,
                             strides=1, activation='sigmoid',padding='same')(x)

    model = keras.Model(decoder_inputs, outputs, name='decoder')
    return model


# In[29]:


cvae_encoder = encoder3(encoded_dim = encoded_dim, input_shape = input_shape)
cvae_decoder = decoder3(encoded_dim = encoded_dim, input_shape = input_shape)


# In[30]:


cvae_encoder.summary()


# In[31]:


cvae_decoder.summary()


# In[32]:


class CVAE(keras.Model):
    def __init__(self, encoder, decoder, beta, shape, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.shape = shape
        self.latent_var = []
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        #
        self.val_total_loss_tracker = keras.metrics.Mean(name="val_loss")
        self.val_reconstruction_loss_tracker = keras.metrics.Mean(
            name="val_reconstruction_loss")
        self.val_kl_loss_tracker = keras.metrics.Mean(name="val_kl_loss")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_kl_loss_tracker,

        ]
       
    def call(self, inputs):
        _, input_label, conditional_input = self.conditional_input(inputs)
        z_mean, z_log_var = self.encoder(conditional_input)
        z_cond = self.sampling(z_mean, z_log_var, input_label)
        return self.decoder(z_cond)
    
    def conditional_input(self, inputs, label_size=10): 
  
        image_size = [self.shape[0], self.shape[1], self.shape[2]]
    
        input_img = layers.InputLayer(input_shape=image_size,
                                      dtype ='float32')(inputs[0])
        input_label = layers.InputLayer(input_shape=(label_size, ),
                                        dtype ='float32')(inputs[1])

        labels = tf.reshape(inputs[1], [-1, 1, 1, label_size])
        labels = tf.cast(labels, dtype='float32')
        ones = tf.ones([inputs[0].shape[0]] + image_size[0:-1] + [label_size]) 
        labels = ones * labels
        conditional_input = layers.Concatenate(axis=3)([input_img, labels]) 
        return  input_img, input_label, conditional_input

    def sampling(self, z_mean, z_log_var, input_label):
        if len(input_label.shape) == 1:
            input_label = np.expand_dims(input_label, axis=0)

        eps = tf.random.normal(tf.shape(z_log_var), dtype=tf.float32,
                               mean=0., stddev=1.0, name='epsilon')
        z = z_mean + tf.exp(z_log_var / 2) * eps
        z_cond = tf.concat([z, input_label], axis=1)
        return z_cond

    def train_step(self, data):

        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
        
            input_img, input_label, conditional_input = self.conditional_input(data)
            z_mean, z_log_var = self.encoder(conditional_input)
            self.latent_var.append(z_log_var)
            z_cond = self.sampling(z_mean, z_log_var, input_label)
            reconstruction = self.decoder(z_cond)
            #reconstruction_loss = np.prod(self.shape) * tf.keras.losses.MSE(tf.keras.backend.flatten(input_img),
            #                        tf.keras.backend.flatten(reconstruction))
            reconstruction_loss = tf.reduce_sum(
                 keras.losses.MSE(input_img, # removed np.prod(self.shape) *
                                    reconstruction), axis=(1, 2))            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean)
                      - tf.exp(z_log_var))
            #kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) # was just reduce_sum
            kl_loss = tf.reduce_sum(kl_loss, axis=1) #sum over encoded dimensiosn, average over batch
            kl_loss = self.beta * kl_loss
            total_loss = reconstruction_loss + kl_loss
            total_loss = tf.reduce_mean(total_loss) #not necessary since I added red mean in kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        #wandb.log({"loss": total_loss, "reconstructon_loss": reconstruction_loss, "kl_loss": kl_loss,})
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        input_img, input_label, conditional_input = self.conditional_input(data)
        z_mean, z_log_var = self.encoder(conditional_input)
        z_cond = self.sampling(z_mean, z_log_var, input_label)
        reconstruction = self.decoder(z_cond)
        reconstruction_loss = tf.reduce_sum(
                 keras.losses.MSE(input_img,
                                    reconstruction), axis=(1, 2))   
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean)
                  - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        kl_loss = self.beta * kl_loss
        total_loss =  reconstruction_loss + kl_loss
        total_loss = tf.reduce_mean(total_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return{
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }


# In[33]:


#kl_coefficient = encoded_dim / (input_shape[0] * input_shape[1] * input_shape[2])
kl_coefficient = wandb.config.kl_coefficient
print('kl coefficient: {:.3f}'.format(kl_coefficient))
# from b vae paper, use beta = encoded_dimension / pixel_dimension i.e. -> 0.068
cvae = CVAE(cvae_encoder, cvae_decoder, kl_coefficient, input_shape)
cvae.built = True
cvae.summary()


# In[34]:


cvae_input = cvae.encoder.input[0]
cvae_output = cvae.decoder.output
mu = cvae.encoder.get_layer('mu').output
log_var = cvae.encoder.get_layer('log_var').output


learning_rate = 0.0001


opt = keras.optimizers.Adam(learning_rate = learning_rate)
cvae.compile(optimizer = opt)
#cvae.compile(optimizer='adam')


# In[36]:


np.prod(input_shape)


# In[37]:


# dim = batch_size
#kl_loss = tf.reduce_sum(kl_loss, axis=1) #sum over encoded dimensiosn, average over batch
#kl_loss.shape







# In[42]:


epoch_count = 100



# In[43]:





from wandb.keras import WandbCallback
#wandb.init(project="my-test-project", entity="nrderus")


# In[45]:


patience = 10





# In[46]:

batch_size = 100


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
             patience=patience, restore_best_weights=False)

history = cvae.fit([train_x,train_y_one_hot],
                   validation_data=([val_x,val_y_one_hot],None),
                   epochs=epoch_count,
                   batch_size=batch_size,
                   callbacks=[early_stop, WandbCallback(save_weights_only=True)  ])


# In[47]:


plot_history(history)


# In[48]:


Train_Val_Plot(history.history['loss'][1:],
               history.history['val_loss'][1:],
               history.history['reconstruction_loss'][1:],
               history.history['val_reconstruction_loss'][1:],
               history.history['kl_loss'][1:],
               history.history['val_kl_loss'][1:]
               )


# In[49]:


#%tensorboard --logdir logs


# In[50]:




# In[51]:


#forward the port 6006 on server on 12006 on  my machine
# ssh -N -L 16006:127.0.0.1:6006 nicolas.derus2@137.204.48.211
# access with http://127.0.0.1:16006


# In[52]:


test_loss = cvae.evaluate([test_x, test_y_one_hot],None,
            batch_size=batch_size,verbose=0)
print('Test loss: {:.3f}'.format(test_loss[2]))


# ## **Embdedding**

# In[53]:



label_size = 10
_, input_label_train, train_input = cvae.conditional_input([train_x, train_y_one_hot])
_, input_label_test, test_input = cvae.conditional_input([test_x, test_y_one_hot])
_, input_label_val, val_input = cvae.conditional_input([val_x, val_y_one_hot])


print(input_label_train.shape)
print(train_input.shape)


# In[54]:


train_x_mean, train_log_var = cvae.encoder.predict(train_input)
test_x_mean, test_log_var = cvae.encoder.predict(test_input)
val_x_mean, val_log_var = cvae.encoder.predict(val_input)

print(train_x_mean.shape)
print(train_log_var.shape)


# In[55]:


if encoded_dim > 2:
    from sklearn import manifold
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    train_x_tsne = tsne.fit_transform(train_x_mean[:2000])
    test_x_tsne = tsne.fit_transform(test_x_mean[:2000])
    val_x_tsne = tsne.fit_transform(val_x_mean[:2000])
    plot_2d_data( [train_x_tsne, test_x_tsne, val_x_tsne],
            [train_y[:2000], test_y[:2000] ,val_y[:2000]],
            ['Train','Test', 'Validation'],(18,6))
    plot_2d_data_categorical( [train_x_mean, test_x_mean, val_x_mean],
            [train_y, test_y ,val_y],
            ['Train','Test', 'Validation'],(12,36))


# In[56]:


if encoded_dim == 2:
    plot_2d_data( [train_x_mean, test_x_mean, val_x_mean],
                [train_y, test_y ,val_y],
                ['Train','Test', 'Validation'],(18,6))
    plot_2d_data_categorical( [train_x_mean, test_x_mean, val_x_mean],
                [train_y, test_y ,val_y],
                ['Train','Test', 'Validation'],(12,36))


# ## **Reconstruction**

# In[57]:


#reconstructions...
z_cond_train = sampling(train_x_mean, train_log_var, input_label_train)
z_cond_test = sampling(test_x_mean, test_log_var, input_label_test)
z_cond_val = sampling(val_x_mean, val_log_var, input_label_val)


# In[58]:


reconstruction_train = cvae.decoder(z_cond_train)
reconstruction_test = cvae.decoder(z_cond_test)
reconstruction_val = cvae.decoder(z_cond_val)


# In[59]:


random_idx = random.randint(0, reconstruction_train.shape[0])
random_idx


# In[60]:


image_count = 10

fig, axs = plt.subplots(2, image_count, figsize=(12, 3))
for i in range(image_count):
  random_idx = random.randint(0, reconstruction_train.shape[0])
  axs[0, i].imshow(train_x[random_idx])
  axs[0, i].axis('off')
  #axs[0, i].set_title(train_y[random_idx])
  axs[0, i].set_title( labels[int(train_y[random_idx])]  )
  axs[1, i].imshow(reconstruction_train[random_idx])
  axs[1, i].axis('off')
  wandb.log({"Reconstructions": wandb.Image(plt)})


# In[61]:





# ## **Generation**

# In[62]:


def reparametrization(z_mean, z_log_var, input_label):
    """ Performs the riparametrization trick"""

    eps = tf.random.normal(shape = (input_label.shape[0], encoded_dim),
                             mean = 0.0, stddev = 1.0)       
    z = z_mean + tf.math.exp(z_log_var * .5) * eps
    z_cond = tf.concat([z, input_label], axis=1) # (batch_size, label_dim + latent_dim)

    return z_cond


# In[63]:


digit_label = 5
digit_label_one_hot = to_categorical(digit_label, category_count).reshape(1,-1)
a = tf.convert_to_tensor(digit_label_one_hot)
b = tf.concat([a, a], axis=0) # with 1 dimension, it fails...
b.shape


# In[64]:


z_cond = reparametrization(z_mean=0, z_log_var=0.2, input_label = b)
decoded_x = cvae_decoder.predict(z_cond)
digit = decoded_x[0].reshape(input_shape) 
plt.axis('off')
plt.imshow(digit)


# In[65]:


digit_label = 5
_, axs = plt.subplots(2, image_count, figsize=(12, 3))
for i in range(image_count):
    digit_label_one_hot = to_categorical(digit_label, category_count).reshape(1,-1)
    a = tf.convert_to_tensor(digit_label_one_hot)
    b = tf.concat([a, a], axis=0) # with 1 dimension, it fails...
    z_cond = reparametrization(z_mean=0, z_log_var=0.2, input_label = b)
    decoded_x = cvae_decoder.predict(z_cond)
    digit_0 = decoded_x[0].reshape(input_shape) 
    digit_1 = decoded_x[1].reshape(input_shape) 
    axs[0, i].imshow(digit_0)
    axs[0, i].axis('off')
    #axs[0, i].set_title(digit_label)
    axs[0, i].set_title( labels[digit_label]  )
    axs[1, i].imshow(digit_1)
    axs[1, i].axis('off')
    axs[1, i].set_title( labels[digit_label]  )
    wandb.log({"Generations": wandb.Image(plt)})


# In[66]:


cvae.save_weights('weights/cvae_toy.h5')
cvae_encoder.save('models/cvae_encoder_toy.h5')
cvae_decoder.save('models/cvae_decoder_toy.h5')


# In[67]:


if encoded_dim == 2:
  n = 10  # number of images per row and column
  limit=3 # random values are sampled from the range [-limit,+limit]
  first_dim_const= 0  # constant value of the second latent dimension

  grid_y = np.linspace(-limit,limit, n) 

  generated_images=[]
  for digit_label in range(category_count):
    digit_label_one_hot=to_categorical(digit_label, category_count).reshape(1,-1)
    
    single_row_generated_images=[]
    for i, yi in enumerate(grid_y):
      random_sample = np.array([[first_dim_const, yi]])
      z_cond = sampling(z_mean=random_sample, z_log_var=0.3,
                      input_label = digit_label_one_hot )
      decoded_x = cvae_decoder.predict(z_cond)
      single_row_generated_images.append(decoded_x[0].reshape(input_shape))
    generated_images.append(single_row_generated_images)      

  plot_generated_images(generated_images,n,n)


# In[68]:


#cvae.built = True
#cvae.load_weights('weights/vae_toy.h5')


# # **Visualize activation functions**

# In[69]:


def visualize_activations(model):
    test = test_x[1]
    plt.imshow(test)
    #test = image.img_to_array(test)
    test = np.expand_dims(test, axis=0)
    test.shape
    test_label = test_y_one_hot[0]
    img_tensor = [test, test_label]
    from keras import models

    # Extracts the outputs of the top 8 layers:
    import tensorflow as tf

    layer_outputs = []
    layer_names = []
    for layer in model.layers[1:]:
        
        try: 
            layer_outputs.append(layer.get_output_at(1))
            layer_names.append(layer.name)
        
        except:
            layer_outputs.append(layer.output)
            layer_names.append(layer.name)

    # Creates a model that will return these outputs, given the model input:
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    
    # This will return a list of 5 Numpy arrays:
    # one array per layer activation
    if 'encoder' in model.name:
        input_img, input_label, conditional_input = cvae.conditional_input(img_tensor)
        activations = activation_model.predict(conditional_input) #for encoder

    if 'decoder' in model.name:
        input_img, input_label, conditional_input = cvae.conditional_input(img_tensor)
        input_label = np.expand_dims(input_label, axis=0)
        z_mean, z_log_var = cvae.encoder(conditional_input)
        z_cond = cvae.sampling(z_mean, z_log_var, input_label)
        
        activations = activation_model.predict(z_cond) #for decoder
    
    for activation, name in zip(activations[0:], layer_names[0:]):
        print(name)
        print(activation.shape)
    
    for counter, (activation, name) in enumerate(zip(activations[0:], layer_names[0:])):
        print(name)
        plot_filters(activation, name, counter, model_name=model.name)
            


# In[70]:


import math
def plot_filters(activation_layer, layer_name, counter, model_name):
    if len(activation_layer.shape) == 2: # if flat layer
        print('flat')
        return None
        if activation_layer.shape[1] == 1875:
            activation_layer = activation_layer.reshape(1, 25, 25, 3)
        if activation_layer.shape[1] == 1024:
           activation_layer = activation_layer.reshape(1, 16, 16, 4)
        if activation_layer.shape[1] == 512:
           activation_layer = activation_layer.reshape(1, 8, 8, 8)

    n = math.floor(np.sqrt(activation_layer.shape[3]))

    if int(n + 0.5) ** 2 == activation_layer.shape[3]:

        m = n
    else:
        m = math.floor(activation_layer.shape[3] / n)

    if activation_layer.shape[3] == 1:
        fig, ax = plt.subplots(1, 1, sharex='col', sharey='row',
                                figsize=(15, 15))
        fig.suptitle(layer_name)

        ax.imshow(activation_layer[0,:, :, 0], cmap='viridis')
        wandb.log({"Activations": wandb.Image(plt, caption="{}_{}".format(model_name, layer_name)) })
      
        return None   

            
    if n == 1:

        fig, ax = plt.subplots(1, 3, sharex='col', sharey='row',figsize=(15, 15))
        fig.suptitle(layer_name)
        for i in range(3):
            ax[i].imshow(activation_layer[0,:, :, i], cmap='viridis')
        wandb.log({"Activations": wandb.Image(plt, caption="{}_{}".format(model_name, layer_name)) })

        return None   

    fig, ax = plt.subplots(n, m, sharex='col', sharey='row',figsize=(15, 15))
    fig.suptitle(layer_name)
    
 

    filter_counter = 0
    for i in range(n):
        for j in range(m):
            ax[i, j].imshow(activation_layer[0,:, :, filter_counter], cmap='viridis')
            filter_counter += 1
            if filter_counter == (activation_layer.shape[3] ):
                break

    wandb.log({"Activations": wandb.Image(plt, caption="{}_{}".format(model_name, layer_name)) })
    
    return None


# In[71]:


visualize_activations( cvae.encoder)
visualize_activations(cvae.decoder)


# # **Visualize filters**

# In[72]:


model = cvae.encoder


# In[73]:


# util function to convert a tensor into a valid image
def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    #img = img[ 25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img


# In[74]:


def filter_conditional_input( inputs, label_size=10): 
  
        image_size = [input_shape[0], input_shape[1], input_shape[2]]

        input_img = layers.InputLayer(input_shape=image_size,
                                        dtype ='float32')(inputs[0])
        input_label = layers.InputLayer(input_shape=(label_size, ),
                                        dtype ='float32')(inputs[1])

        labels = tf.reshape(inputs[1], [-1, 1, 1, label_size])
        labels = tf.cast(labels, dtype='float32')
        ones = tf.ones([inputs[0].shape[0]] + image_size[0:-1] + [label_size]) 
        labels = ones * labels
        conditional_input = layers.Concatenate(axis=3)([input_img, labels]) 
        return  input_img, input_label, conditional_input


# In[75]:


def build_nth_filter_loss(filter_index, layer_name):
    """
    We build a loss function that maximizes the activation
    of the nth filter of the layer considered
    """
    # Create a connection between the input and the target layer
    
    submodel = tf.keras.models.Model([model.inputs[0]],
                                     [model.get_layer(layer_name).output])

# Initiate random noise

    input_img_data = np.random.random((1, input_shape[0], input_shape[1],
                                     input_shape[2]))
    input_img_data =(input_img_data - 0.5) * 0.25
    # Cast random noise from np.float64 to tf.float32 Variable
    input_img_data = tf.Variable(tf.cast(input_img_data, tf.float32))

    data = [input_img_data, train_y_one_hot[0]]
    _, _, conditional_input_img = filter_conditional_input(data) 
    conditional_input_img= tf.Variable(tf.cast(conditional_input_img,
                                         tf.float32))

    for _ in range(epochs):
        with tf.GradientTape() as tape:
            outputs = submodel(conditional_input_img)
            loss_value = tf.reduce_mean(outputs[:, 2:-2, 2:-2, filter_index]) #removed borders in loss
        grads = tape.gradient(loss_value, conditional_input_img)
        normalized_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads)))
                                   + 1e-5)
        #normalized_grads = tf.math.l2_normalize(grads)
        conditional_input_img.assign_add(normalized_grads * step_size)

    # this function returns the loss and grads given the input picture
    #iterate = K.function([input_img], [loss_value, grads])

    if loss_value > 0:
        
        #img = conditional_input_img.numpy().astype(np.float64)
        #img = img.squeeze()
        #img = deprocess_image(img) / 255.
        img = conditional_input_img.numpy().astype(np.float64)
        img = img.squeeze()
        img = deprocess_image(img)
        kept_filters.append((img, loss_value))


# In[76]:


#dimensions of the generated pictures for each filter.
img_width = input_shape[0]
img_height = input_shape[1]

# this is the placeholder for the input images
input_img = model.input
print(input_img.shape)

# get the symbolic outputs of each "key" layer (we gave them unique names).
#layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
layer_dict


# In[77]:


layers_filters = [layer.name for layer in model.layers]


# In[78]:


epochs = 30
step_size = 10.
kept_filters = []
filters_dict = dict()
for layer_name in layers_filters:
    if 'conv' in layer_name:
        layer = model.get_layer(layer_name)
        print('Processing filter for layer:', layer_name)
        for filter_index in range(min(layer.output.shape[-1], 100)):
            # print('Processing filter %d' % filter_index)
            build_nth_filter_loss(filter_index, layer_name)
        filters_dict[layer.name] = kept_filters
        kept_filters = []


# In[79]:


from keras.preprocessing.image import save_img

def stich_filters(kept_filters, layer_name):
    # By default, we will stich the best 64 (n*n) filters on a 8 x 8 grid.
    n = int(np.sqrt(len(kept_filters)))
    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin

    stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, _ = kept_filters[i * n + j]
            width_margin = (img_width + margin) * i
            height_margin = (img_height + margin) * j
            stitched_filters[
                width_margin: width_margin + img_width,
                height_margin: height_margin + img_height, :] = img[:, :, :3] 

    wandb.log({"Filters": wandb.Image(stitched_filters, caption="{}_{}".format(model.name, layer_name)) })
    # save the result to disk
   
    


# In[80]:


for layer_name, kept_filters in filters_dict.items():
    print('Stiching filters for {}'.format(layer_name))
    stich_filters(kept_filters, layer_name)
    print('number of filters kept:', len(kept_filters))
    print('Completed.')


# In[87]:


wandb.finish(exit_code=0, quiet = True)


# In[81]:


#from src.cvae import CVAE
#CVAE(cvae_encoder, cvae_decoder, kl_coefficient, input_shape)


# # **Report activations**




# In[ ]:






