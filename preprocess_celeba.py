#from https://github.com/daib13/TwoStageVAE/blob/871862382746d619e7dc7527f3d02f924e1b0af5/preprocess.py
import numpy as np 
import pickle
#from mnist import MNIST
import os
#from PIL import imread, imresize, imsave
from imageio import imread, imsave
from PIL import Image
import pickle
import pandas as pd
ROOT_FOLDER = 'datasets/'


def load_celeba_data(flag='training', side_length=None, num=None):
    dir_path = os.path.join(ROOT_FOLDER, 'img_align_celeba/img_align_celeba')
    filelist = [filename for filename in os.listdir(dir_path) if filename.endswith('jpg')]
    assert len(filelist) == 202599
    if flag == 'training':
        start_idx, end_idx = 0, 162770
    elif flag == 'val':
        start_idx, end_idx = 162770, 182637
    else:
        start_idx, end_idx = 182637, 202599
    
    
    imgs = []
    for i in range(start_idx, end_idx):
        img = np.array(imread(dir_path + os.sep + filelist[i]))
        img = img[45:173,25:153]
        if side_length is not None:
            img = np.array(Image.fromarray(img).resize(( side_length, side_length )))
        new_side_length = np.shape(img)[1]
        img = np.reshape(img, [1, new_side_length, new_side_length, 3])
        imgs.append(img) #added
        if num is not None and len(imgs) >= num:
            break
        if len(imgs) % 5000 == 0:
            print('Processing {} images...'.format(len(imgs)))
    imgs = np.concatenate(imgs, 0)

    return imgs.astype(np.uint8)


# Center crop 140x140 and resize to 64x64
# Consistent with the preporcess in WAE [1] paper
# [1] Ilya Tolstikhin, Olivier Bousquet, Sylvain Gelly, and Bernhard Schoelkopf. Wasserstein auto-encoders. International Conference on Learning Representations, 2018.


# Center crop 128x128 and resize to 64x64
def preprocess_celeba():
    x_val = load_celeba_data('val', 64)
    np.save(os.path.join('datasets', 'celeba', 'val.npy'), x_val)
    x_test = load_celeba_data('test', 64)
    np.save(os.path.join('datasets', 'celeba', 'test.npy'), x_test)
    x_train = load_celeba_data('training', 64)
    np.save(os.path.join('datasets', 'celeba', 'train.npy'), x_train)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic

if __name__ == '__main__':
    preprocess_celeba()
    #preprocess_celeba140()
    #preprocess_mnist()
    #preprocess_fashion()
    #preporcess_cifar10()