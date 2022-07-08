#!/usr/bin/env python
# coding: utf-8

# # **Conditional Variational autoencoder (VAE) - Toy datasets**

# # **Utility functions**

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from glob import glob
print(tf. __version__)


# In[ ]:


import datetime, os


# In[ ]:


K.clear_session()


# In[ ]:


def find_indices(lst, condition):
    return np.array([i for i, elem in enumerate(lst) if condition(elem)])
    
def plot_2d_data_categorical(data_2d, y, titles=None, figsize = (7, 7), category_count=2):
  fig, axs = plt.subplots(category_count, len(data_2d), figsize = figsize)
  colors = np.array(['#7FFFD4', '#458B74'])
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
        axs[k, i].set_xlim([-60, 60])
        axs[k, i].set_ylim([-60, 60])
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

def plot_generated_images(generated_images, nrows, ncols, digit_label,
                          no_space_between_plots=False, figsize=(10, 10)):
  _, axs = plt.subplots(nrows, ncols,figsize=figsize,squeeze=False)

  for i in range(nrows):
    for j in range(ncols):
      axs[i,j].axis('off')
      axs[i,j].imshow(generated_images[i][j], cmap='gray')

  if no_space_between_plots:
    plt.subplots_adjust(wspace=0,hspace=0)
  
  wandb.log({"Latent_interpolation_class: {}".format(digit_label): wandb.Image(plt)})

  plt.show()


# In[ ]:


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


# In[ ]:


def sampling(z_mean, z_log_var, input_label):

    eps = tf.random.normal(tf.shape(z_log_var), dtype=tf.float32,
                            mean=0., stddev=1.0, name='epsilon')
    z = z_mean + tf.exp(z_log_var / 2) * eps
    z_cond = tf.concat([z, input_label], axis=1) 
    return z_cond


# In[ ]:


def Train_Val_Plot(loss, val_loss, reconstruction_loss, val_reconstruction_loss, kl_loss, val_kl_loss):

    fig, (ax1, ax2, ax3) = plt.subplots(1,4, figsize= (16,4))
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

# In[ ]:


dataset_name = "Histo"
# import data
imagePatches = glob('datasets/breast-histopathology/IDC_regular_ps50_idx5/**/*.png', recursive=True)
for filename in imagePatches[0:10]:
    print(filename)


# In[ ]:


class0 = [] # 0 = no cancer
class1 = [] # 1 = cancer

for filename in imagePatches:
    if filename.endswith("class0.png"):
         class0.append(filename)
    else:
        class1.append(filename)


# In[ ]:


sampled_class0 = random.sample(class0, 50000)
sampled_class1 = random.sample(class1, 50000)
len(sampled_class0)


# In[ ]:


from matplotlib.image import imread
import cv2

def get_image_arrays(data, label):
    img_arrays = []
    for i in data:
        if i.endswith('.png'):
            img = cv2.imread(i ,cv2.IMREAD_COLOR)
            img_sized = cv2.resize(img, (50, 50), #was (70,70)
                        interpolation=cv2.INTER_LINEAR)
            img_arrays.append([img_sized, label]) 
    return img_arrays


# In[ ]:


class0_array = get_image_arrays(sampled_class0, 0)
class1_array = get_image_arrays(sampled_class1, 1)


# In[ ]:


combined_data = np.concatenate((class0_array, class1_array))
random.seed(41)
random.shuffle(combined_data)


# In[ ]:


X = []
y = []

for features, label in combined_data:
    X.append(features)
    y.append(label)


# In[ ]:


X = np.array(X).reshape(-1, 50, 50, 3)
y = np.array(y)

print(X.shape)
print(y.shape)


# In[ ]:


train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2,
                                    random_state = 11)

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, 
                            test_size = 0.25, random_state = 11) 
                            # 0.25 x 0.8 = 0.2
train_y_one_hot = to_categorical(train_y)
test_y_one_hot = to_categorical(test_y)
val_y_one_hot = to_categorical(val_y)
train_y = np.argmax(train_y_one_hot, axis=1) # from one-hot encoding to integer
test_y = np.argmax(test_y_one_hot, axis=1)
val_y = np.argmax(val_y_one_hot, axis=1)
labels = ['non-cancer','cancer']
category_count = 2
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)


# In[ ]:


print('Min value: ', train_x.min())
print('Max value: ', train_x.max())


# In[ ]:


image_count = 10

_, axs = plt.subplots(1, image_count, figsize=(20, 20))
for i in range(image_count):
  random_idx=random.randint(0, train_x.shape[0])
  axs[i].imshow(train_x[random_idx], cmap='gray')
  axs[i].axis('off')
  axs[i].set_title(labels[train_y[random_idx]])


# In[ ]:


if len(train_x.shape) == 3:
    train_x=np.expand_dims(train_x,axis=3)
    val_x=np.expand_dims(val_x,axis=3)
    test_x=np.expand_dims(test_x,axis=3)
    print('Train shape: ',train_x.shape)
    print('Validation shape: ',val_x.shape)
    print('Test shape: ',test_x.shape)


# In[ ]:


input_shape = train_x.shape[1:]
input_shape


# In[ ]:


train_x = train_x/255.0
val_x = val_x/255.0
test_x = test_x/255.0

print('Min value: ',train_x.min())
print('Max value: ', train_x.max())


# In[ ]:


original_image_shape=(train_x.shape[1], train_x.shape[2])

train_x_flatten=np.reshape(train_x,(train_x.shape[0],-1))
val_x_flatten=np.reshape(val_x,(val_x.shape[0],-1))
test_x_flatten=np.reshape(test_x,(test_x.shape[0],-1))

print('Train data flatten shape: ',train_x_flatten.shape)
print('Validation data flatten shape: ',val_x_flatten.shape)
print('Test data flatten shape: ',test_x_flatten.shape)


# # **CVAE model**
# Creating a CVAE class and plugging encoder and decoder

# In[ ]:


def bn_relu(inputs):
    bn = layers.BatchNormalization()(inputs)
    relu = layers.ReLU()(bn)
    return(relu)


# In[ ]:


encoded_dim = 256


# In[ ]:


class EncoderResBlock(keras.Model):
    def __init__(self, filters, downsample):
        super().__init__()
        if downsample:
            self.conv1 = layers.Conv2D(filters, 3, 2, padding='same')
            self.shortcut = keras.Sequential([
                layers.Conv2D(filters, 1, 2),
                layers.BatchNormalization()
            ])
        else:
            self.conv1 = layers.Conv2D(filters, 3, 1, padding='same')
            self.shortcut = keras.Sequential()
 
        self.conv2 = layers.Conv2D(filters, 3, 1, padding='same')
        
    def __call__(self, input):
        
        shortcut = self.shortcut(input)

        input = self.conv1(input)
        input = layers.BatchNormalization()(input)
        input = layers.ReLU()(input)
        input = layers.BatchNormalization()(input)
        input = layers.ReLU()(input)

        input= input + shortcut
        return layers.ReLU()(input)


# In[ ]:


class EncoderResNet(keras.Model):
    def __init__(self, resblock, repeat, encoded_dim):
   
        super().__init__()
        
        self.layer0 = keras.Sequential([
            layers.Conv2D(64, 7, 2, padding='same'),
            layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name='layer0')

        self.layer1 = keras.Sequential([
            resblock(64, downsample=False) for _ in range(repeat[0])
        ], name='layer1')

        self.layer2 = keras.Sequential([
            resblock(128, downsample=True)
        ] + [
            resblock(128, downsample=False) for _ in range(1, repeat[1])
        ], name='layer2')

        self.layer3 = keras.Sequential([
            resblock(256, downsample=True)
        ] + [
            resblock(256, downsample=False) for _ in range(1, repeat[2])
        ], name='layer3')

        self.layer4 = keras.Sequential([
            resblock(512, downsample=True)
        ] + [
            resblock(512, downsample=False) for _ in range(1, repeat[3])
        ], name='layer4')

        self.flat = layers.Flatten(name = 'flatten')
        self.bottleneck = layers.Dense(encoded_dim * 2, name='encoder_bottleneck')
        self.mu = layers.Dense(encoded_dim, name='mu')
        self.log_var = layers.Dense(encoded_dim, name='log_var')
 

    def call(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.flat(input)
        input = self.bottleneck(input)
        mu = self.mu(input)
        log_var = self.log_var(input)

        return [mu, log_var]

    def get_config(self):
        return super().get_config()


# In[ ]:


class EncoderResNet18(EncoderResNet):
    def __init__(self, encoded_dim):
        super().__init__(EncoderResBlock, [2, 2, 2, 2], encoded_dim)

    def call(self, input):
        return super().call(input)

    def model(self, input_shape):
        x = layers.Input(input_shape, name='input', dtype='float32')
        return keras.models.Model(x, self.call(x))


# In[ ]:


#encoder, ok.
resnet18 = EncoderResNet18(encoded_dim = encoded_dim)
resnet18 = resnet18.model(input_shape=(50, 50, 5))
resnet18.summary()


# In[ ]:


class DecoderResBlock(keras.Model):
    def __init__(self, filters, upsample):
        super().__init__()
        if upsample:
            self.conv1 = layers.Conv2DTranspose(filters, 3, 2, padding='same')
            self.shortcut = keras.Sequential([
                layers.Conv2DTranspose(filters, 1, 2),
                layers.BatchNormalization()
            ])
        else:
            self.conv1 = layers.Conv2DTranspose(filters, 3, 1, padding='same')
            self.shortcut = keras.Sequential()
 
        self.conv2 = layers.Conv2DTranspose(filters, 3, 1, padding='same')

    def __call__(self, input):
        shortcut = self.shortcut(input)

        input = self.conv1(input)
        input = layers.BatchNormalization()(input)
        input = layers.ReLU()(input)

        input = self.conv2(input)
        input = layers.BatchNormalization()(input)
        input = layers.ReLU()(input)

        input = input + shortcut
        return layers.ReLU()(input)


# In[ ]:


class DecoderResNet(keras.Model):
    def __init__(self, resblock, repeat, encoded_dim):
        super().__init__()
        
        self.layer5 = keras.Sequential([
            resblock(512, upsample=False)
        ] + [
            resblock(512, upsample=False)  for _ in range(1, repeat[0])
        ], name='layer5')


        self.layer6 = keras.Sequential([
            resblock(256, upsample=True)
        ] + [
            resblock(256, upsample=False) for _ in range(1, repeat[1])
        ], name='layer6')


        self.layer7 = keras.Sequential([
            resblock(128, upsample=True)
        ] + [
            resblock(128, upsample=False) for _ in range(1, repeat[2])
        ], name='layer7')
        
        # self.layer7 = keras.Sequential([ # TO DO: change back this into resblock, heigth/depth issue
        #     layers.Conv2DTranspose(128, 4, 1, padding='valid'),
        #     #layers.BatchNormalization(),
        #     layers.ReLU()
        # ], name='layer7')

        self.layer8 =  keras.Sequential([ 
            resblock(64, upsample=True)
        ] + [
            resblock(64, upsample=False) for _ in range(repeat[3])
        ], name='layer8')

        self.layer9 = keras.Sequential([
                layers.Conv2DTranspose(64, 7, 1, padding='same'),
                layers.UpSampling2D(3),
                #layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU()
            ], name='layer9')
          
        self.bottleneck = layers.Dense(encoded_dim * 2, name='bottleneck')
        self.pre_reshape = layers.Dense(2*2*512, name='pre_reshape')
        self.reshape = layers.Reshape(target_shape=(2, 2, 512), name = 'reshape')
        self.output_layer = layers.Conv2DTranspose(filters = 3, kernel_size=3, strides=1, activation='sigmoid' ,padding='valid', name='outputs')
    
    def call(self, input):
        input = self.bottleneck(input)
        input = self.pre_reshape(input)
        input = self.reshape(input)
        input = self.layer5(input)
        input = self.layer6(input)
        input = self.layer7(input)
        input = self.layer8(input)
        input = self.layer9(input)
        out = self.output_layer(input)
        return out

    def get_config(self):
        return super().get_config()


# In[ ]:


class DecoderResNet18(DecoderResNet):
    def __init__(self, encoded_dim):
        super().__init__(DecoderResBlock, [2, 2, 2, 1], encoded_dim)

    def call(self, input):
        return super().call(input)

    def model(self, input_shape):
        x = keras.Input(input_shape, name='input')
        return keras.models.Model(x, self.call(x))


# In[ ]:


#decoder, ok?
decoder = DecoderResNet18( encoded_dim = encoded_dim)
decoder = decoder.model(input_shape=(encoded_dim + category_count,))
decoder.summary()


# In[ ]:


cvae_encoder = resnet18
cvae_decoder = decoder


# In[ ]:


a = tf.random.normal((10, 50, 50, 5))


# In[ ]:


cvae_encoder.summary()


# In[ ]:


cvae_decoder.summary()


# In[ ]:


import visualkeras
visualkeras.layered_view(cvae_encoder, legend=True)


# In[ ]:


visualkeras.layered_view(cvae_decoder, legend=True)


# In[ ]:


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
        self.latent_var_tracker = keras.metrics.Mean(name="latent_var") #added
    
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
            self.latent_var_tracker,

        ]
       
    def call(self, inputs):
        _, input_label, conditional_input = self.conditional_input(inputs)
        z_mean, z_log_var = self.encoder(conditional_input)
        z_cond = self.sampling(z_mean, z_log_var, input_label)
        return self.decoder(z_cond)

  
    def conditional_input(self, inputs, label_size=2):
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
            z_cond = self.sampling(z_mean, z_log_var, input_label)
            reconstruction = self.decoder(z_cond)

            #reconstruction_loss = np.prod(self.shape) * tf.keras.losses.MSE(tf.keras.backend.flatten(input_img),
            #                        tf.keras.backend.flatten(reconstruction))
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
        self.latent_var_tracker.update_state(z_log_var)
        #wandb.log({"loss": total_loss, "reconstructon_loss": reconstruction_loss, "kl_loss": kl_loss,})
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "loss_no_weights": self.total_loss_no_weights_tracker.result(),
            "latent_var": self.latent_var_tracker.result()
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


# In[ ]:


#kl_coefficient = encoded_dim / (input_shape[0] * input_shape[1] * input_shape[2])
kl_coefficient = .65
print('kl coefficient: {:.3f}'.format(kl_coefficient))
# from b vae paper, use beta = encoded_dimension / pixel_dimension i.e. -> 0.068
cvae = CVAE(cvae_encoder, cvae_decoder, kl_coefficient, input_shape)
cvae.built = True
cvae.summary()


# In[ ]:


def conditional_input(self, inputs, label_size=10): 
    print('ones_dim', inputs[0].shape[0])
    image_size = [self.shape[0], self.shape[1], self.shape[2]]

    input_img = layers.InputLayer(input_shape=image_size,
                                    dtype ='float32')(inputs[0])
    input_label = layers.InputLayer(input_shape=(label_size, ),
                                    dtype ='float32')(inputs[1])
    print('image_size:', image_size)

    labels = tf.reshape(inputs[1], [-1, 1, 1, label_size])
    labels = tf.cast(labels, dtype='float32')
    print('labels dim:', labels.shape)

    ones = tf.ones([batch_size] + image_size[0:-1] + [label_size]) 
    labels = ones * labels
    print('labels dim:', labels.shape)
    conditional_input = layers.Concatenate(axis=3)([input_img, labels]) 
    return  input_img, input_label, conditional_input


# In[ ]:


cvae_input = cvae.encoder.input[0]
cvae_output = cvae.decoder.output
mu = cvae.encoder.get_layer('mu').output
log_var = cvae.encoder.get_layer('log_var').output

learning_rate = 0.0001 

opt = keras.optimizers.Adam(learning_rate = learning_rate)
cvae.compile(optimizer = opt, run_eagerly=False)
#cvae.compile(optimizer='adam')


# ## **Training**

# In[ ]:


epoch_count = 100
batch_size = 100


# In[ ]:


train_x.shape


# In[ ]:


import wandb
from wandb.keras import WandbCallback
#wandb.init(project="my-test-project", entity="nrderus")


# In[ ]:


patience = 5


wandb.init(project="HistoDL", entity="nrderus",
  config = {
  "dataset": dataset_name,
  "model": "CVAE_resnet",
  "encoded_dim": encoded_dim,
  "kl_coefficient": kl_coefficient,
  "learning_rate": learning_rate,
  "epochs": epoch_count,
  "batch_size": batch_size,
  "patience": patience,
  
})


# In[ ]:





# In[ ]:



early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
             patience=patience, restore_best_weights=False)

history = cvae.fit([train_x,train_y_one_hot],
                   validation_data=([val_x,val_y_one_hot],None),
                   epochs=epoch_count,
                   batch_size=100,
                   callbacks=[early_stop, WandbCallback(save_weights_only=True) ])


# In[ ]:


#K.get_value(cvae.latent_var[0])


# In[ ]:


#tf.numpy_function(cvae.latent_var[0], tf.float32)


# In[ ]:


plot_history(history)


# In[ ]:


# Train_Val_Plot(history.history['loss'][1:],
#                history.history['val_loss'][1:],
#                history.history['reconstruction_loss'][1:],
               
#                history.history['val_reconstruction_loss'][1:],
#                history.history['kl_loss'][1:],
#                history.history['val_kl_loss'][1:]
#                )


# In[ ]:


#%tensorboard --logdir logs


# In[ ]:


cvae.save_weights('weights/cvae_toy.h5')
cvae_encoder.save('models/cvae_encoder_toy.h5')
cvae_decoder.save('models/cvae_decoder_toy.h5')


# In[ ]:


#forward the port 6006 on server on 12006 on  my machine
# ssh -N -L 16006:127.0.0.1:6006 nicolas.derus2@137.204.48.211
# access with http://127.0.0.1:16006


# ## **Embdedding**

# In[ ]:



label_size = 10
_, input_label_train, train_input = cvae.conditional_input([train_x, train_y_one_hot])
_, input_label_test, test_input = cvae.conditional_input([test_x, test_y_one_hot])
_, input_label_val, val_input = cvae.conditional_input([val_x, val_y_one_hot])


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
            [train_y[:2000], test_y[:2000] ,val_y[:2000]],
            ['Train','Test', 'Validation'],(18,6))
    plot_2d_data_categorical( [train_x_tsne, test_x_tsne, val_x_tsne],
            [train_y[:2000], test_y[:2000] ,val_y[:2000]],
            ['Train','Test', 'Validation'],(18,12))


# In[ ]:


if encoded_dim == 2:
    plot_2d_data( [train_x_mean, test_x_mean, val_x_mean],
                [train_y, test_y ,val_y],
                ['Train','Test', 'Validation'],(18,6))
    plot_2d_data_categorical( [train_x_mean, test_x_mean, val_x_mean],
                [train_y, test_y ,val_y],
                ['Train','Test', 'Validation'],(12,36))


# ## **Reconstruction**

# In[ ]:


#reconstructions...
z_cond_train = sampling(train_x_mean, train_log_var, input_label_train)
z_cond_test = sampling(test_x_mean, test_log_var, input_label_test)
z_cond_val = sampling(val_x_mean, val_log_var, input_label_val)


# In[ ]:


reconstruction_train = cvae.decoder(z_cond_train[:100])
reconstruction_test = cvae.decoder(z_cond_test[:100])
reconstruction_val = cvae.decoder(z_cond_val[:100])


# In[ ]:


random_idx = random.randint(0, reconstruction_train.shape[0])
random_idx


# In[ ]:


reconstruction_train.shape[0]


# In[ ]:


image_count = 10

fig, axs = plt.subplots(2, image_count, figsize=(12, 3))
for i in range(image_count):
  random_idx = random.randint(0, reconstruction_train.shape[0]-1)
  axs[0, i].imshow(train_x[random_idx])
  axs[0, i].axis('off')
  #axs[0, i].set_title(train_y[random_idx])
  axs[0, i].set_title( labels[int(train_y[random_idx])]  )
  axs[1, i].imshow(reconstruction_train[random_idx])
  axs[1, i].axis('off')
wandb.log({"Reconstructions": wandb.Image(plt)})


# In[ ]:


(cvae.latent_var)


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


digit_label = 0
digit_label_one_hot = to_categorical(digit_label, category_count).reshape(1,-1)
a = tf.convert_to_tensor(digit_label_one_hot)
b = tf.concat([a, a], axis=0) # with 1 dimension, it fails...
b.shape


# In[ ]:


z_cond = reparametrization(z_mean=0, z_log_var=0.2, input_label = b)
decoded_x = cvae_decoder.predict(z_cond)
digit = decoded_x[0].reshape(input_shape) 
plt.axis('off')
plt.imshow(digit)


# In[ ]:


digit_label = 0
_, axs = plt.subplots(2, image_count, figsize=(12, 3))
for i in range(image_count):
    digit_label_one_hot = to_categorical(digit_label, category_count).reshape(1,-1)
    a = tf.convert_to_tensor(digit_label_one_hot)
    b = tf.concat([a, a], axis=0) # with 1 dimension, it fails...
    z_cond = reparametrization(z_mean=0, z_log_var=0.3, input_label = b)
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


# In[ ]:


def generations_class(digit_label=1):
    _, axs = plt.subplots(2, image_count, figsize=(12, 3))
    for i in range(image_count):
        digit_label_one_hot = to_categorical(digit_label, category_count).reshape(1,-1)
        a = tf.convert_to_tensor(digit_label_one_hot)
        b = tf.concat([a, a], axis=0) # with 1 dimension, it fails...
        z_cond = reparametrization(z_mean=0, z_log_var=0.3, input_label = b)
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
        wandb.log({"Generations: {}".format(digit_label): wandb.Image(plt)})

    


# In[ ]:


label_size = category_count
if (label_size <= 10):
    for i in range(label_size):
        generations_class(i)


# In[ ]:





# In[ ]:


def latent_space_interpolation(digit_label=1):
  n = 10 # number of images per row and column
  limit=3 # random values are sampled from the range [-limit,+limit]
  digit_label_one_hot = to_categorical(digit_label, category_count).reshape(1,-1)
  a = tf.convert_to_tensor(digit_label_one_hot)
  grid_x = np.linspace(-limit,limit, n) 
  grid_y = np.linspace(limit,-limit, n)

  generated_images=[]
  for i, yi in enumerate(grid_y):
    single_row_generated_images=[]
    for j, xi in enumerate(grid_x):
      random_sample = np.array([[xi, yi]])
      digit_label_one_hot = to_categorical(digit_label, category_count).reshape(1,-1)
      a = tf.convert_to_tensor(digit_label_one_hot)
      b = tf.concat([a, a], axis=0) # with 1 dimension, it fails...
      z_cond = reparametrization(z_mean=random_sample, z_log_var=0.0, input_label = b)
      decoded_x = cvae.decoder.predict(z_cond)
      single_row_generated_images.append(decoded_x[0].reshape(original_image_shape))
    generated_images.append(single_row_generated_images)      
  plot_generated_images(generated_images,n,n,True)


# In[ ]:


label_size = category_count
if (encoded_dim == 2 & label_size <= 10):
    for i in range(label_size):
        latent_space_interpolation(i)


# In[ ]:


#cvae.built = True
#cvae.load_weights('weights/vae_toy.h5')


# In[ ]:


wandb.finish(exit_code=0, quiet = True) # TEMPORARY


# # **Visualize activation functions**

# In[ ]:


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
            


# In[ ]:


test = test_x[1]
plt.imshow(test)
#test = image.img_to_array(test)
test = np.expand_dims(test, axis=0)
test.shape
test_label = test_y_one_hot[0]
img_tensor = [test, test_label]
input_img, input_label, conditional_input = cvae.conditional_input(img_tensor)


# In[ ]:


print(input_img.shape)
print(input_label.shape)
print(conditional_input.shape)


# In[ ]:


inputs = [input_img, input_label]
image_size = [28, 28, 1]
input_img = layers.InputLayer(input_shape=image_size,
                            dtype ='float32')(inputs[0])
input_label = layers.InputLayer(input_shape=(label_size, ),
                                dtype ='float32')(inputs[1])


# In[ ]:


labels = tf.reshape(inputs[1], [-1, 1, 1, label_size])
print(labels)
labels = tf.cast(labels, dtype='float32')
print(labels)
ones = tf.ones([inputs[0].shape[0]] + image_size[0:-1] + [label_size])
print(ones.shape)
labels = ones * labels 
print(labels)
conditional_input = layers.Concatenate(axis=3)([input_img, labels]) 


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




