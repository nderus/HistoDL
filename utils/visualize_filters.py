import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from keras.preprocessing.image import save_img
from tensorflow import keras

class VisualizeFilters(keras.Model):
    def __init__(self, model, **kwargs):
        self.model = model
        self.epochs = 100
        self.step_size = 1.

    # util function to convert a tensor into a valid image
    def deprocess_image(x):
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + K.epsilon())
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

    def filter_conditional_input(inputs, label_size=10): 
  
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

    
    def build_nth_filter_loss(filter_index, layer_name):
        """
        We build a loss function that maximizes the activation
        of the nth filter of the layer considered
        """
        layer_output = layer_dict[layer_name].output
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # Initiate random noise
        # Create a connection between the input and the target layer
        
        submodel = tf.keras.models.Model([model.inputs[0]],
                                     [model.get_layer(layer_name).output])

    # Initiate random noise

        input_img_data = np.random.random((1, input_shape[0], input_shape[1],
                                        input_shape[2]))
        input_img_data = (input_img_data - 0.5) * 20 + 128.

        # Cast random noise from np.float64 to tf.float32 Variable
        input_img_data = tf.Variable(tf.cast(input_img_data, tf.float32))
        data = [input_img_data, train_y_one_hot[0]]
        _, _, conditional_input_img = self.filter_conditional_input(data)
        conditional_input_img= tf.Variable(tf.cast(conditional_input_img,
                                             tf.float32))

        for _ in range(epochs):
            with tf.GradientTape() as tape:
                outputs = submodel(conditional_input_img)
                loss_value = tf.reduce_mean(outputs[:, :, :, filter_index])
            grads = tape.gradient(loss_value, conditional_input_img)
            normalized_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
            conditional_input_img.assign_add(normalized_grads * step_size)

        # this function returns the loss and grads given the input picture
        #iterate = K.function([input_img], [loss_value, grads])

        if loss_value > 0:
            img = conditional_input_img.numpy().astype(np.float64)
            img = img.squeeze()
            img = self.deprocess_image(img) / 255.
            self.kept_filters.append((img, loss_value))
    
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
                img, loss = kept_filters[i * n + j]
                width_margin = (img_width + margin) * i
                height_margin = (img_height + margin) * j
                stitched_filters[
                    width_margin: width_margin + img_width,
                    height_margin: height_margin + img_height, :] = img[:, :, :3] 

        # save the result to disk
        save_img('img/filters/vae/{}_toy_stitched_filters_{}.png'.format(model.name, layer_name), stitched_filters)
    

    def __call__(self):
        #dimensions of the generated pictures for each filter.
        img_width = input_shape[0]
        img_height = input_shape[1]

        # this is the placeholder for the input images
        input_img = self.model.input
        layers_filters = [layer.name for layer in self.model.layers]
   
        
        kept_filters = []
        filters_dict = dict()
        for layer_name in layers_filters:
            if 'conv' in layer_name:
                layer = self.model.get_layer(layer_name)
                print('Processing filter for layer:', layer_name)
                for filter_index in range(min(layer.output.shape[-1], 100)):
                    # print('Processing filter %d' % filter_index)
                    self.build_nth_filter_loss(filter_index, layer_name)
                filters_dict[layer.name] = kept_filters
                kept_filters = []

        layer_dict = dict([(layer.name, layer) for layer in self.model.layers[1:]])
        
        for layer_name, kept_filters in filters_dict.items():
            print('Stiching filters for {}'.format(layer_name))
            self.stich_filters(kept_filters, layer_name)
            print('number of filters kept:', len(kept_filters))
            print('Completed.')