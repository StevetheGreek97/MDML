import cv2
import numpy as np
from tensorflow.keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt

class Cam():
    def __init__(self, model, classIdx): 
        
        self.model = model
        self.classIdx = classIdx
        self.layers = [layer.name for layer in reversed(model.layers) \
                       if len(layer.output_shape) == 4 and            \
                       (layer.__class__.__name__ == 'ReLU' or         \
                        isinstance(layer, tf.keras.layers.Conv2D))]

    def GradCam(self, img_array, layer_name = None, eps=1e-8):
        '''
        Creates a grad-cam heatmap given a model and a layer name contained with that model
        

        Args:
        img_array: (img_width x img_width) numpy array
        layer_name: str


        Returns 
        uint8 numpy array with shape (img_height, img_width)

        '''
        # if conv layer not provided use last layer
        if not layer_name:
          layer_name = self.layers[-1]


        gradModel = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(layer_name).output,
                    self.model.output])
        
        with tf.GradientTape() as tape:
                # cast the image tensor to a float-32 data type, pass the
                # image through the gradient model, and grab the loss
                    # associated with the specific class index
            inputs = tf.cast(img_array, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
                # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)
        
        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
            # the convolution and guided gradients have a batch dimension
            # (which we don't need) so let's grab the volume itself and
            # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        # compute the average of the gradient values, and using them
            # as weights, compute the ponderation of the filters with
            # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
    
        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (img_array.shape[2], img_array.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def superimpose(self, img_bgr, cam, thresh, emphasize=False):
      
      '''
      Superimposes a grad-cam heatmap onto an image for model interpretation and visualization.
      

      Args:
        image: (img_width x img_height x 3) numpy array
        grad-cam heatmap: (img_width x img_width) numpy array
        threshold: float
        emphasize: boolean

      Returns 
        uint8 numpy array with shape (img_height, img_width, 3)

      '''
      heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
      if emphasize:
          heatmap = sigmoid(heatmap, 50, thresh, 1)
      heatmap = np.uint8(255 * heatmap)
      heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
      
      hif = .8
      superimposed_img = heatmap * hif + img_bgr
      superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255  
      superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
      
      return superimposed_img_rgb
    
    def fuse_layers(self, img):
      '''
        Fuses grad-cam heatmaps from a list of model layers into a single heatmap
        and superimposes the heatmap onto an image.

        Args:
          layers: list of strings
          model: tf model
          img: (img_width x img_height x 3) numpy array


        Returns 
          uint8 numpy array with shape (img_height, img_width, 3)

        '''
      cams = []
      for layer in self.layers:

        cam = self.GradCam(np.expand_dims(img, axis=0), layer)
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        cams.append(cam)
    #     print(cams)

      fused = np.mean(cams, axis=0)

      return fused

    def Guided_GradCam(self, img):
        sal = SaliencyMap(self.model, img)
        sal_map = sal.guided_saliency_map()
        gradcam = self.fuse_layers(img) / 255

        GGC = np.multiply(sal_map, gradcam )
        return GGC

def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x-b)))

class SaliencyMap():
    def __init__(self, model, img):
        self.model = model
        self.cloned_model = tf.keras.models.clone_model(model)
        self.img = img
        self.classId = self.get_class_id()
        self.layers = self.get_layers()
        
    def get_layers(self):
        return [layer for layer in self.cloned_model.layers[1:] if hasattr(layer,'activation')]
    
    def change_activation(self):
        for layer in self.layers:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu
                

    def prep_img(self):
        
        input_img = self.img
        input_img = tf.cast(input_img, tf.float32)
        input_img = tf.keras.applications.densenet.preprocess_input(input_img)
        return input_img
    
    def get_class_id(self):
        return tf.argmax(self.cloned_model( self.prep_img() ), axis = 1)
    
    def guided_saliency_map(self):
        self.change_activation()
        max_idx = self.classId

        
        with tf.GradientTape() as tape:
            input_img = self.prep_img()
            cloned_model = self.cloned_model
            tape.watch(input_img)
            result = cloned_model(input_img)
            max_score = result[0,max_idx[0]]
        grads = tape.gradient(max_score, input_img)[0]
        
        grads_norm = grads[:,:,0] + grads[:,:,1] + grads[:,:,2]
        grads_norm = (grads_norm - tf.reduce_min(grads_norm))/ (tf.reduce_max(grads_norm)- tf.reduce_min(grads_norm))
        
        return grads_norm
    
    def gradient_saliency_map(self):
      """
      Saliency Map with Gradient Based Backpropogation. The map is extracted 
      using a single backpropagation pass through a Convolutional Neural Network
      """

      images = tf.Variable(self.img, dtype=float)
      # calculate the gradient with respect to the top class score 
      # to see which pixels in the image contribute the most.
      with tf.GradientTape() as tape:
          pred = self.model(images, training=False)
          class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
    #         print(class_idxs_sorted)
          loss = pred[0][class_idxs_sorted[0]]
    #         print(loss)

      grads = tape.gradient(loss, images)
      
      dgrad_abs = tf.math.abs(grads)
      # Find the max of the absolute values of the gradient along each RGB channel
      dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
      
      # Normalize the grad to between 0 and 1
      arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
      grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
      
      
      
      # fig, axes = plt.subplots(1,2,figsize=(14,5))
      # axes[0].imshow(img)
      # i = axes[1].imshow(grad_eval,cmap='viridis',alpha=0.8)
    #     fig.colorbar(i)
      return grad_eval

@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0,tf.float32)  * tf.cast(x>0,tf.float32) * dy
    return tf.nn.relu(x), grad
