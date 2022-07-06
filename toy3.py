#!/usr/bin/env python
# coding: utf-8

# # **Conditional Variational autoencoder (VAE) - Toy datasets**

# # **Utility functions**

# In[1]:


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
import wandb

# In[2]:


import datetime, os


# In[3]:


K.clear_session()


wandb.init(project="HistoDL", entity="nrderus",
  config = {
  "dataset": "CelebA",
  "model": "CVAE",
  "encoded_dim": 2,
  "kl_coefficient": 1,
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 100,
  "patience": 5,
  
})


# 

# 

# 

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
        
def plot_2d_data(data_2d, titles=None, figsize = (7, 7)):
  _, axs = plt.subplots(1, len(data_2d), figsize = figsize)

  for i in range(len(data_2d)):
    
    if (titles != None):
      axs[i].set_title(titles[i])
    scatter=axs[i].scatter(data_2d[i][:, 0], data_2d[i][:, 1], s=1,
                             cmap=plt.cm.Paired)
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


import os 
import numpy as np 
from imageio import imread, imsave
from PIL import Image


def load_dataset(name, root_folder):
    data_folder = os.path.join(root_folder, name)
    if name.lower() == 'celeba':
        x = np.load(os.path.join(data_folder, 'train.npy'))
        side_length = 64
        channels = 3
    else:
        raise Exception('No such dataset called {}.'.format(name))
    return x, side_length, channels


def load_test_dataset(name, root_folder):
    data_folder = os.path.join(root_folder, name)
   
    if name.lower() == 'celeba':
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 64
        channels = 3
    else:
        raise Exception('No such dataset called {}.'.format(name))
    return x, side_length, channels


def load_val_dataset(name, root_folder):
    data_folder = os.path.join(root_folder, name)
   
    if name.lower() == 'celeba':
        x = np.load(os.path.join(data_folder, 'val.npy'))
        side_length = 64
        channels = 3
    else:
        raise Exception('No such dataset called {}.'.format(name))
    return x, side_length, channels


# In[9]:


train_x, _, _ = load_dataset("celeba", "datasets")
test_x, _, _ = load_test_dataset("celeba", "datasets")
val_x, _, _ = load_val_dataset("celeba", "datasets")

train_x = train_x/255.0
test_x = test_x/255.0
val_x = val_x/255.0


print('Min value: ',train_x.min())

print('Max value: ',train_x.max())


# In[10]:


import pandas as pd
file_path = "datasets/list_attr_celeba.csv"
df = pd.read_csv(file_path, header = 0, index_col = 0).replace(-1,0)

df.shape


# In[11]:


train_y = np.array(df.iloc[ 0: 162770], dtype = "float32")
print(train_y.shape)
print(train_x.shape)


# In[12]:


test_y = np.array(df.iloc[182637: 202599], dtype = "float32")
print(test_y.shape)
print(test_x.shape)


# In[13]:


val_y = np.array(df.iloc[162770: 182637], dtype = "float32")
print(val_y.shape)
print(val_x.shape)


# In[14]:


train_x = np.array(train_x, dtype = "float")
test_x = np.array(test_x, dtype = "float")
val_x = np.array(val_x, dtype = "float")


# In[15]:


input_shape = (64, 64, 3)


# # **CVAE model**
# Creating a CVAE class and plugging encoder and decoder

# In[16]:


encoded_dim = wandb.config.encoded_dim


# In[17]:


#relu brings a lot of activation values = 0, leaky seems better
# https://towardsdatascience.com/the-dying-relu-problem-clearly-explained-42d0c54e0d24

def bn_relu(inputs):
    bn = layers.BatchNormalization()(inputs)
    relu = layers.LeakyReLU(0.2)(bn)
    return(relu)


# In[18]:


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


# In[19]:


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


# In[20]:


cvae_encoder = encoder3(encoded_dim = encoded_dim, input_shape = input_shape, label_size=40)
cvae_decoder = decoder3(encoded_dim = encoded_dim, input_shape = input_shape, label_size=40)


# In[21]:


cvae_encoder.summary()


# In[22]:


cvae_decoder.summary()


# In[38]:


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
        self.total_loss_no_weights_tracker = keras.metrics.Mean(name="loss_no_weights")
        #
        self.val_total_loss_tracker = keras.metrics.Mean(name="val_loss")
        self.val_reconstruction_loss_tracker = keras.metrics.Mean(
            name="val_reconstruction_loss")
        self.val_kl_loss_tracker = keras.metrics.Mean(name="val_kl_loss")
        self.val_total_loss_no_weights_tracker = keras.metrics.Mean(name="val_loss_no_weights")
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_kl_loss_tracker,
            self.total_loss_no_weights_tracker,
            self.val_total_loss_no_weights_tracker,

        ]
       
    def call(self, inputs):
        _, input_label, conditional_input = self.conditional_input(inputs)
        z_mean, z_log_var = self.encoder(conditional_input)
        z_cond = self.sampling(z_mean, z_log_var, input_label)
        return self.decoder(z_cond)
    
    def conditional_input(self, inputs, label_size=40): 
  
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

            reconstruction_loss = tf.reduce_sum(
                 keras.losses.MSE(input_img, # removed np.prod(self.shape) *
                                    reconstruction), axis=(1, 2))            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean)
                      - tf.exp(z_log_var))

            kl_loss = tf.reduce_sum(kl_loss, axis=1) #sum over encoded dimensiosn, average over batch
            total_loss_no_weights = reconstruction_loss + kl_loss
            total_loss_no_weights = tf.reduce_mean(total_loss_no_weights)
            kl_loss = self.beta * kl_loss
            total_loss = reconstruction_loss + kl_loss
            total_loss = tf.reduce_mean(total_loss) 
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_no_weights_tracker.update_state(total_loss_no_weights)
        #wandb.log({"loss": total_loss, "reconstructon_loss": reconstruction_loss, "kl_loss": kl_loss,})
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "loss_no_weights": self.total_loss_no_weights_tracker.result()
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
        total_loss_no_weights = reconstruction_loss + kl_loss
        total_loss_no_weights = tf.reduce_mean(total_loss_no_weights)
        kl_loss = self.beta * kl_loss
        total_loss =  reconstruction_loss + kl_loss
        total_loss = tf.reduce_mean(total_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_no_weights_tracker.update_state(total_loss_no_weights)
        return{
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
            "loss_no_weights": self.total_loss_no_weights_tracker.result()
        }


# In[39]:


#kl_coefficient = encoded_dim / (input_shape[0] * input_shape[1] * input_shape[2])
kl_coefficient = 1
print('kl coefficient: {:.3f}'.format(kl_coefficient))
# from b vae paper, use beta = encoded_dimension / pixel_dimension i.e. -> 0.068
cvae = CVAE(cvae_encoder, cvae_decoder, kl_coefficient, input_shape)
cvae.built = True
cvae.summary()


# In[40]:


cvae_input = cvae.encoder.input[0]
cvae_output = cvae.decoder.output
mu = cvae.encoder.get_layer('mu').output
log_var = cvae.encoder.get_layer('log_var').output

#learning_rate = 0.0001
learning_rate = 0.001
opt = keras.optimizers.Adam(learning_rate = learning_rate)
cvae.compile(optimizer = opt)
#cvae.compile(optimizer='adam')


# In[41]:


z_mean = np.random.normal(size=(100, 512))
z_log_var =  np.random.normal(size=(100, 512))

kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean)
            - tf.exp(z_log_var))

kl_loss.shape


# In[42]:


np.prod(input_shape)


# In[ ]:





# In[43]:


# dim = batch_size
#kl_loss = tf.reduce_sum(kl_loss, axis=1) #sum over encoded dimensiosn, average over batch
#kl_loss.shape


# In[44]:


kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) #sum over encoded dimensiosn, average over batch
kl_loss.shape


# In[45]:


input_img = np.random.normal(size=(100, 32, 32, 3))
reconstruction = np.random.normal(size=(100, 32, 32, 3))

reconstruction_loss = np.prod(input_img.shape) * tf.keras.losses.MSE(tf.keras.backend.flatten(input_img),
                                    tf.keras.backend.flatten(reconstruction)) 

reconstruction_loss


# In[46]:


#dim = batch_size
reconstruction_loss = tf.reduce_sum(

                keras.losses.MSE(input_img,
                                    reconstruction), axis=(1, 2))

reconstruction_loss.shape


# ## **Training**

# In[47]:


epoch_count = 100
batch_size = wandb.config.batch_size

# In[48]:


label_size = 40


# In[49]:



from wandb.keras import WandbCallback
#wandb.init(project="my-test-project", entity="nrderus")


# In[50]:


train_x = train_x[:100000,:, :, :]
train_y = train_y[:100000, :]

test_x = test_x[:19000,:, :, :]
test_y = test_y[:19000, :]

val_x = val_x[:19000, :, :, :]
val_y = val_y[:19000, :]


# In[51]:


patience = 5




# In[52]:


patience = 5
batch_size = 100
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
             patience=patience, restore_best_weights=True)

history = cvae.fit([train_x, train_y],
                   validation_data=([val_x, val_y],None),
                   epochs=epoch_count,
                   batch_size=batch_size,
                   callbacks=[early_stop, WandbCallback(save_weights_only=True) ])


# In[ ]:


plot_history(history)


# In[ ]:


Train_Val_Plot(history.history['loss'][1:],
               history.history['val_loss'][1:],
               history.history['reconstruction_loss'][1:],
               history.history['val_reconstruction_loss'][1:],
               history.history['kl_loss'][1:],
               history.history['val_kl_loss'][1:]
               )


# In[ ]:


#%tensorboard --logdir logs


# In[ ]:


#forward the port 6006 on server on 12006 on  my machine
# ssh -N -L 16006:127.0.0.1:6006 nicolas.derus2@137.204.48.211
# access with http://127.0.0.1:16006


# In[ ]:


train_y[:100].shape


# ## **Embdedding**

# In[ ]:



label_size = 40
_, input_label_train, train_input = cvae.conditional_input([train_x[:5000], train_y[:5000]])
_, input_label_test, test_input = cvae.conditional_input([test_x[:5000], test_y[:5000]])
_, input_label_val, val_input = cvae.conditional_input([val_x[:5000], val_y[:5000]])


print(input_label_train.shape)
print(train_input.shape)


# In[ ]:


train_x_mean, train_log_var = cvae.encoder.predict(train_input)
test_x_mean, test_log_var = cvae.encoder.predict(test_input)
val_x_mean, val_log_var = cvae.encoder.predict(val_input)

print(train_x_mean.shape)
print(train_log_var.shape)


# In[ ]:


if encoded_dim > 2:
    from sklearn import manifold
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    train_x_tsne = tsne.fit_transform(train_x_mean[:2000])
    test_x_tsne = tsne.fit_transform(test_x_mean[:2000])
    val_x_tsne = tsne.fit_transform(val_x_mean[:2000])
    plot_2d_data( [train_x_tsne, test_x_tsne, val_x_tsne],
            ['Train','Test', 'Validation'],(18,6))
   


# In[ ]:


if encoded_dim == 2:
    plot_2d_data( [train_x_mean, test_x_mean, val_x_mean],
                ['Train','Test', 'Validation'],(18,6))


# ## **Reconstruction**

# In[ ]:


#reconstructions...
z_cond_train = sampling(train_x_mean, train_log_var, input_label_train)
z_cond_test = sampling(test_x_mean, test_log_var, input_label_test)
z_cond_val = sampling(val_x_mean, val_log_var, input_label_val)


# In[ ]:


reconstruction_train = cvae.decoder(z_cond_train)
reconstruction_test = cvae.decoder(z_cond_test)
reconstruction_val = cvae.decoder(z_cond_val)


# In[ ]:


random_idx = random.randint(0, reconstruction_train.shape[0])
random_idx


# In[ ]:


image_count = 10

fig, axs = plt.subplots(2, image_count, figsize=(12, 3))
for i in range(image_count):
  random_idx = random.randint(0, reconstruction_train.shape[0])
  axs[0, i].imshow(train_x[random_idx])
  axs[0, i].axis('off')
  #axs[0, i].set_title(train_y[random_idx])
  #axs[0, i].set_title( labels[int(train_y[random_idx])]  )
  axs[1, i].imshow(reconstruction_train[random_idx])
  axs[1, i].axis('off')
  wandb.log({"Reconstructions": wandb.Image(plt)})


# ## **Generation**

# In[ ]:


def reparametrization(z_mean, z_log_var, input_label):
    """ Performs the riparametrization trick"""

    eps = tf.random.normal(shape = (input_label.shape[0], encoded_dim),
                             mean = 0.0, stddev = 1.0)       
    z = z_mean + tf.math.exp(z_log_var * .5) * eps
    z_cond = tf.concat([z, input_label], axis=1) # (batch_size, label_dim + latent_dim)

    return z_cond


# In[ ]:


#target_attr = [2,9,15,21,24,31,33,39]
target_attr = [2, 9, 12, 21, 26, 27, 31, 39]

_, axs = plt.subplots(2, image_count, figsize=(12, 3))

for i in range(image_count):
    attr_vect = np.zeros(40)
    for attr in target_attr:
        attr_vect[attr] = 1
    labels = np.tile(attr_vect, reps = [batch_size, 1])
    print("Generation ofimages with attributes: ", target_attr )
    a = tf.convert_to_tensor(labels,dtype="float")
    b = tf.concat([a, a], axis=0) # with 1 dimension, it fails...
    z_cond = reparametrization(z_mean=0, z_log_var=0.3, input_label = b)
    decoded_x = cvae_decoder.predict(z_cond)
    digit_0 = decoded_x[0].reshape(input_shape) 
    digit_1 = decoded_x[1].reshape(input_shape) 
    axs[0, i].imshow(digit_0)
    axs[0, i].axis('off')
    axs[1, i].imshow(digit_1)
    axs[1, i].axis('off')


attributes = str(df.columns[target_attr].tolist())
wandb.log(   {"Generations:_{}".format(attributes): wandb.Image(plt)}     )


# In[ ]:


target_attr = [0, 8, 15, 20]


_, axs = plt.subplots(2, image_count, figsize=(12, 3))

for i in range(image_count):
    attr_vect = np.zeros(40)
    for attr in target_attr:
        attr_vect[attr] = 1
    labels = np.tile(attr_vect, reps = [batch_size, 1])
    print("Generation ofimages with attributes: ", target_attr )
    a = tf.convert_to_tensor(labels,dtype="float")
    b = tf.concat([a, a], axis=0) # with 1 dimension, it fails...
    z_cond = reparametrization(z_mean=0, z_log_var=0.3, input_label = b)
    decoded_x = cvae_decoder.predict(z_cond)
    digit_0 = decoded_x[0].reshape(input_shape) 
    digit_1 = decoded_x[1].reshape(input_shape) 
    axs[0, i].imshow(digit_0)
    axs[0, i].axis('off')
    axs[1, i].imshow(digit_1)
    axs[1, i].axis('off')


attributes = str(df.columns[target_attr].tolist())
wandb.log(   {"Generations:_{}".format(attributes): wandb.Image(plt)}     )


# In[ ]:


attr_vect = np.zeros(40)
for attr in target_attr:
    attr_vect[attr] = 1
labels = np.tile(attr_vect, reps = [2, 1])
labels.shape


# In[ ]:





# In[ ]:


df.columns.get_loc("Wearing_Hat")


# In[ ]:


a = tf.convert_to_tensor(labels, dtype="float")
b = tf.concat([a, a], axis=0) # with 1 dimension, it fails...
z_cond = reparametrization(z_mean=0, z_log_var=0.3, input_label = b)


# # **Visualize activation functions**

# In[ ]:


def visualize_activations(model):
    test = test_x[1]
    plt.imshow(test)
    #test = image.img_to_array(test)
    test = np.expand_dims(test, axis=0)
    test.shape
    test_label = test_y[0]
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
            


# In[ ]:


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


# In[ ]:


visualize_activations( cvae.encoder)
visualize_activations(cvae.decoder)


# # **Visualize filters**

# In[ ]:


model = cvae.encoder


# In[ ]:


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


# In[ ]:


def filter_conditional_input( inputs, label_size=40): 
  
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


# In[ ]:


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

    data = [input_img_data, train_y[0]]
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


# In[ ]:


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


# In[ ]:


layers_filters = [layer.name for layer in model.layers]


# In[ ]:


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


# In[ ]:


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

    


# In[ ]:


for layer_name, kept_filters in filters_dict.items():
    print('Stiching filters for {}'.format(layer_name))
    stich_filters(kept_filters, layer_name)
    print('number of filters kept:', len(kept_filters))
    print('Completed.')


# In[ ]:


wandb.finish(exit_code=0, quiet = True)


# In[ ]:


#from src.cvae import CVAE
#CVAE(cvae_encoder, cvae_decoder, kl_coefficient, input_shape)


# # **Report activations**

# # **Report filters**

# # **Report**

# In[ ]:




