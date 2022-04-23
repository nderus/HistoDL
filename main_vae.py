#!/usr/bin/env python
# coding: utf-8

# In[1]:


# TO DO:
# add batch_normalization layer

# try with flattened inputs ?


# In[2]:


# import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.np_utils import to_categorical
import keras.backend as K
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from sklearn.model_selection import train_test_split
from glob import glob
import math


# In[3]:


import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K


# In[4]:


# import data
imagePatches = glob('datasets/breast-histopathology/IDC_regular_ps50_idx5/**/*.png', recursive=True)
for filename in imagePatches[0:10]:
    print(filename)


# In[5]:


class0 = [] # 0 = no cancer
class1 = [] # 1 = cancer

for filename in imagePatches:
    if filename.endswith("class0.png"):
         class0.append(filename)
    else:
        class1.append(filename)


# In[6]:


sampled_class0 = random.sample(class0, 78786)
sampled_class1 = random.sample(class1, 78786)
len(sampled_class0)


# In[7]:


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


# In[8]:


class0_array = get_image_arrays(sampled_class0, 0)
class1_array = get_image_arrays(sampled_class1, 1)


# In[9]:


combined_data = np.concatenate((class0_array, class1_array))
#random.seed(41)
#random.shuffle(combined_data)


# In[10]:


X = []
y = []

for features, label in combined_data:
    X.append(features)
    y.append(label)


# In[11]:


X = np.array(X).reshape(-1, 50, 50, 3)
y = np.array(y)

print(X.shape)
print(y.shape)


# In[12]:


train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2,
                                    random_state = 11)

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, 
                            test_size = 0.25, random_state = 11) 
                            # 0.25 x 0.8 = 0.2
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)
val_y = to_categorical(val_y)
train_y_label = np.argmax(train_y, axis=1) # from one-hot encoding to integer
test_y_label = np.argmax(test_y, axis=1)
val_y_label = np.argmax(val_y, axis=1)
class_names = ('non-cancer','cancer')
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)


# In[13]:


print('Min value: ', train_x.min())
print('Max value: ', train_x.max())


# In[14]:


train_x = train_x / 255
test_x = test_x / 255
print('Min value: ', train_x.min())
print('Max value: ', train_x.max())


# In[15]:


# visualize random images from data
image_count = 10

_, axs = plt.subplots(1, image_count, figsize=(20, 20))
for i in range(image_count):
  random_idx=random.randint(0, train_x.shape[0])
  axs[i].imshow(train_x[random_idx], cmap='gray')
  axs[i].axis('off')
  axs[i].set_title(class_names[train_y_label[random_idx]])


# In[16]:


batch_size = 250
input_shape = (50, 50, 3)

num_features = 7500#50*50*3
latent_dim = 512


# In[17]:


#vae = keras.models.load_model('models/vae.h5')
#encoder = keras.models.load_model('models/encoder.h5')
#decoder = keras.models.load_model('models/decoder.h5')


# In[18]:


# load encoder and decoder models
vae_encoder = keras.models.load_model('models/vae_encoder.h5')
vae_decoder = keras.models.load_model('models/vae_decoder.h5')


# In[19]:


class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        #self.beta_coefficient = beta_coefficient
    
    def call(self, inputs):
        x = self.encoder(inputs)[2]
        return self.decoder(x)
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_sum(
                    keras.losses.MSE(data, reconstruction), axis=(1, 2) # mod
                )
           # )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + (self.beta * kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# In[20]:


beta_coeff = 1
vae = VAE(encoder=vae_encoder, decoder=vae_decoder, beta = beta_coeff)
vae.compile(optimizer='Adam')


# 

# In[21]:


model = vae
epochs = 50
#model.compile( optimizer='adam')
tf.config.run_functions_eagerly(True)
early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
history = model.fit(train_x, epochs=epochs, batch_size=250, callbacks=early_stop)


# In[37]:


def Train_Val_Plot(loss, reconstruction_loss, kl_loss):
    
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize= (16,4))
    fig.suptitle(" MODEL'S METRICS VISUALIZATION ")

    ax1.plot(range(1, len(loss) + 1), loss)
    ax1.set_title('History of Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')

    ax2.plot(range(1, len(reconstruction_loss) + 1), reconstruction_loss)
    ax2.set_title('History of reconstruction_loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('reconstruction_loss')
    #ax1.legend(['training', 'validation'])

    #ax2.legend(['training', 'validation'])
    
    ax3.plot(range(1, len(kl_loss) + 1), kl_loss)

    ax3.set_title(' History of kl_loss')
    ax3.set_xlabel(' Epochs ')
    ax3.set_ylabel('kl_loss')
    #ax3.legend(['training', 'validation'])
     

    plt.show()
    

Train_Val_Plot(history.history['loss'],
               history.history['reconstruction_loss'],
               history.history['kl_loss']
               )


# 

# In[38]:


model.save_weights('weights/vae.h5')


# In[24]:


import pickle

with open('trainHistoryDict.txt', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


# In[25]:


import pandas as pd

object = pd.read_pickle(r'trainHistoryDict.txt')


# In[26]:


plt.imshow(train_x[0])
plt.show()

train_x[0].shape

print(train_y[0])
print(train_y_label[0])


# In[27]:


p = vae.predict(train_x[:1000])


# In[28]:


p[0].shape


# In[ ]:





# In[29]:


vae(np.zeros((1,50,50,3)))
vae.load_weights('weights/vae.h5')


# In[ ]:





# In[39]:


plt.imshow(p[0])
plt.show()


# In[40]:


plt.imshow(train_x[15])
plt.show()


# In[41]:


def plot_predictions(y_true, y_pred):    
    f, ax = plt.subplots(2, 10, figsize=(15, 4))
    for i in range(10):
        ax[0][i].imshow(np.reshape(y_true[i], (50, 50, 3)), aspect='auto')
        ax[1][i].imshow(np.reshape(y_pred[i], (50, 50, 3)), aspect='auto')
    plt.tight_layout()


# In[42]:


plot_predictions(train_x[:100], p)


# In[43]:


# Scatter with images instead of points
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
img_size = 50
def imscatter(x, y, ax, imageData, zoom):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        img = imageData[i]*255.
        img = img.astype(np.uint8).reshape([img_size,img_size,3])
        #img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))
    
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()


# In[44]:


#https://github.com/despoisj/LatentSpaceVisualization/blob/master/visuals.py
from sklearn import manifold

def computeTSNEProjectionOfLatentSpace(X, X_encoded, display=True, save=True):
    # Compute latent space representation
    print("Computing latent space projection...")
    #X_encoded = encoder.predict(X)

    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X_encoded)

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots(figsize=(15, 15))
        ax = fig.add_subplot(111, facecolor='black')
        imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X, ax=ax, zoom=0.5)
        if save:
            fig.savefig('img/t-SNE-embedding_vae_epochs:{}_beta:{}.png'.format(epochs, beta_coeff))
        plt.show()
    else:
        return X_tsne


# In[48]:


X_encoded = vae_encoder.predict(train_x[:1000])[2]
X_encoded.shape
#need to reshape for TSNE
#X_encoded_flatten = X_encoded.reshape(-1,25*25*3)
#X_encoded_flatten.shape
X_encoded_flatten = X_encoded
X_encoded_flatten.shape


# In[ ]:





# In[49]:


tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X_encoded_flatten)


# In[50]:


computeTSNEProjectionOfLatentSpace(train_x[:1000,], X_encoded_flatten, display=True, save=True)


# In[51]:


import pandas as pd
df = pd.DataFrame()
df['y'] = train_y_label[:1000]
df['comp-1'] = X_tsne[:,0]
df['comp-2'] = X_tsne[:,1]


# In[52]:


fig, ax = plt.subplots(figsize=(15, 15))
colors = {0:'blue', 1:'red'}

ax.scatter(df["comp-1"], df["comp-2"], c=df['y'].map(colors), label=colors) 
ax.legend()
plt.show()


# In[53]:


def computeTSNEProjectionOfPixelSpace(X, display=True):
    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X.reshape([-1, 50* 50* 3]))

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots(figsize=(15, 15))
        ax = fig.add_subplot(111, facecolor='black')
        imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X, ax=ax, zoom=0.5)
        fig.savefig('img/t-SNE_original_space.png')
        plt.show()
    else:
        return X_tsne


# In[55]:


computeTSNEProjectionOfPixelSpace(train_x[:1000], display=True)


# In[56]:


def getReconstructedImages(X, encoder, decoder):
    img_size = 50
    nbSamples = X.shape[0]
    nbSquares = int(math.sqrt(nbSamples))
    nbSquaresHeight = 2*nbSquares
    nbSquaresWidth = nbSquaresHeight
    resultImage = np.zeros((nbSquaresHeight*img_size,int(nbSquaresWidth*img_size/2),X.shape[-1]))

    reconstructedX = decoder.predict(encoder.predict(X)[2])

    for i in range(nbSamples) :     # 
        original = X[i]
        reconstruction = reconstructedX[i]
        rowIndex = i%nbSquaresWidth
        columnIndex = int((i-rowIndex)/nbSquaresHeight)
        resultImage[rowIndex*img_size:(rowIndex+1)*img_size,columnIndex*2*img_size:(columnIndex+1)*2*img_size,:] = np.hstack([original,reconstruction])

    return resultImage


# In[57]:


# Reconstructions for samples in dataset
def visualizeReconstructedImages(X_train, X_test, encoder, decoder, save=False):
    trainReconstruction = getReconstructedImages(X_train, encoder, decoder)
    testReconstruction = getReconstructedImages(X_test, encoder, decoder)

    if not save:
        print("Generating 10 image reconstructions...")

    result = np.hstack([trainReconstruction,
            np.zeros([trainReconstruction.shape[0],5,
            trainReconstruction.shape[-1]]),
            testReconstruction])
    result = (result*255.).astype(np.uint8)

    if save:
        fig, _ = plt.subplots(figsize=(15, 15))
        plt.imshow(result)
        fig.savefig('img/vae_reconstructions_epochs:{}_beta:{}.png'.format(epochs, beta_coeff))
    else:
        plt.show()


# In[58]:


visualizeReconstructedImages(train_x[:100], test_x[:100],vae_encoder, vae_decoder, save = True)


# In[59]:


noise = np.random.normal(size=(1, 25, 25, 3))
#noise = noise.reshape((1,25,25,3))
decoded = vae_decoder.predict(noise)
plt.imshow((decoded[0]*255.).astype(np.uint8))


# In[ ]:


noise = []
for i in range(0,1875):
    noise.append( random.randint(-4, 4) )
noise = np.array(noise)
noise = noise.reshape(1, 25, 25, 3)
decoded = vae_decoder.predict(noise)
plt.imshow((decoded[0]))


# In[ ]:


def generate_images(decoder):    
    _, ax = plt.subplots(2, 10, figsize=(15, 4))
    for i in range(2):
        for j in range(10):
            noise = []
            for k in range(0,1875):
                noise.append( random.randint(-1.5, 1.5) )
                
            noise = np.array(noise)
            noise = noise.reshape(1, 25, 25, 3)

            decoded = vae_decoder.predict(noise)
            ax[i][j].imshow(decoded[0], aspect='auto')
       
    plt.tight_layout()


# In[ ]:


def generate_images(decoder):    
    _, ax = plt.subplots(2, 10, figsize=(15, 4))
    for i in range(2):
        for j in range(10):
            noise = np.random.normal(loc=0, scale = 1, size=1875)
                
            #noise = np.array(noise)
            noise = noise.reshape(1, 25, 25, 3)

            decoded = vae_decoder.predict(noise).squeeze()
            ax[i][j].imshow( (decoded*255.).astype(np.uint8) )
       
    plt.tight_layout()


# In[ ]:


generate_images(vae_decoder)


# In[ ]:


#Shows linear inteprolation in image space vs latent space
def visualizeInterpolation(start, end, encoder, decoder, save=False, nbSteps=5):
    print("Generating interpolations...")

    # Create micro batch
    X = np.array([start, end])

    # Compute latent space projection
    latentX = encoder.predict(X)[2]
    latentStart, latentEnd = latentX

    # Get original image for comparison
    startImage, endImage = X

    vectors = []
    normalImages = []
    #Linear interpolation
    alphaValues = np.linspace(0, 1, nbSteps)
    for alpha in alphaValues:
        # Latent space interpolation
        vector = latentStart*(1-alpha) + latentEnd*alpha
        vectors.append(vector)
        # Image space interpolation
        blendImage = cv2.addWeighted(startImage, 1-alpha, endImage, alpha, 0)
        normalImages.append(blendImage)

    # Decode latent space vectors
    vectors = np.array(vectors)
    reconstructions = decoder.predict(vectors)

    # Put final image together
    resultLatent = None
    resultImage = None

    for i in range(len(reconstructions)):
        interpolatedImage = normalImages[i]*255
        interpolatedImage = cv2.resize(interpolatedImage,(50,50))
        interpolatedImage = interpolatedImage.astype(np.uint8)
        resultImage = interpolatedImage if resultImage is None else np.hstack([resultImage,interpolatedImage])

        reconstructedImage = reconstructions[i]*255.
        reconstructedImage = reconstructedImage.reshape(50,50,3)
        #reconstructedImage = cv2.resize(reconstructedImage,(50,50))
        reconstructedImage = reconstructedImage.astype(np.uint8)
        resultLatent = reconstructedImage if resultLatent is None else np.hstack([resultLatent,reconstructedImage])

    result = np.vstack([resultImage,resultLatent])
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.imshow(result)
    plt.tight_layout()
       #    plt.imshow(result)
    if save:
        
        fig.savefig('img/vector_interpolation_epochs:{}_beta:{}.png'.format(epochs, beta_coeff))
    


# In[ ]:


visualizeInterpolation(train_x[random.randint(0,train_x.shape[0])], train_x[random.randint(0, train_x.shape[0])],
                     vae_encoder, vae_decoder, save=True, nbSteps=10)


# In[ ]:


# Computes A, B, C, A+B, A+B-C in latent space
def visualizeArithmetics(a, b, c, encoder, decoder):
    print("Computing arithmetics...")
    # Create micro batch
    X = np.array([a, b, c])

    # Compute latent space projection
    latentA, latentB, latentC = encoder.predict(X)[2]

    add = latentA+latentB
    addSub = latentA+latentB-latentC

    # Create micro batch
    X = np.array([latentA, latentB, latentC, add,addSub])

    # Compute reconstruction
    reconstructedA, reconstructedB, reconstructedC, reconstructedAdd, reconstructedAddSub = decoder.predict(X)
    _, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(np.hstack([reconstructedA, reconstructedB, reconstructedC, reconstructedAdd, reconstructedAddSub]))
    plt.tight_layout()
    #plt.imshow(np.hstack([reconstructedA, reconstructedB, reconstructedC, reconstructedAdd, reconstructedAddSub]))
    #cv2.waitKey()


# In[ ]:


#visualizeArithmetics(test_x[random.randint(0,test_x.shape[0])],test_x[random.randint(0,test_x.shape[0])], test_x[random.randint(0, test_x.shape[0])], vae_encoder, vae_decoder)
visualizeArithmetics(train_x[random.randint(0,train_x.shape[0])],train_x[random.randint(0,train_x.shape[0])], train_x[random.randint(0, train_x.shape[0])], vae_encoder, vae_decoder)


# In[ ]:


def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)


# In[ ]:


#https://www.researchgate.net/publication/324057819_Comparing_Generative_Adversarial_Network_Techniques_for_Image_Creation_and_Modification/link/5abcd63caca27222c7543718/download
#bilinear interpolation
#1. sample 4 random real images (corners)
#2. encode them 
#2b t-sne?
#3. use bilinear interpolation between these images
#4. decode the interpolations
input1 = train_x[random.randint(0,train_x.shape[0])]
input2 = train_x[random.randint(0,train_x.shape[0])]
input3 = train_x[random.randint(0,train_x.shape[0])]
input4 = train_x[random.randint(0,train_x.shape[0])]

X = np.array([input1, input2, input3, input4])

# Compute latent space projection
latentX = vae_encoder.predict(X)[2]
latent1, latent2, latent3, latent4 = latentX


# In[ ]:


xx, yy = np.meshgrid(latent1, latent2)


# In[ ]:


import scipy
z = (xx+yy) / 2
f = scipy.interpolate.interp2d(latent1, latent2, z, kind='linear')


# In[ ]:




