import tensorflow as tf
import numpy as np
import cv2

class Saliencymap():
    """
    A class for generating saliency maps for image classification models.
    """

    def __init__(self, model, img):
        """
        Initialize an instance of the Saliencymap class.

        Parameters:
        - model: The image classification model.
        - img: The input image.
        """
        self.model = model
        # Add an extra dimension to the input image
        self.img = np.expand_dims(img, axis=0) 
    
    def _compute_grads(self):
        """
        Compute the gradients of the prediction with respect to the input image.
        """
        img = tf.Variable(self.img, dtype=float)

        with tf.GradientTape() as tape:
            # Get the predictions from the model
            preds = self.model(img)[0] # Nested list
            # Get the index of the class with the highest prediction
            classIdx = np.argsort(preds)[::-1]
            # Use the highest prediction as the loss
            loss = preds[classIdx[0]] 
        # Compute the gradients of the loss with respect to the input image
        grads = tape.gradient(loss, img)

        return grads

    def _normalize(self, arr):
        """
        Normalize an array between 0 and 1.

        Parameters:
        - arr: The input array.

        Returns:
        - The normalized array.
        """
        # Calculate the minimum and maximum values of the array
        min_values = np.min(arr)
        max_values = np.max(arr)
        # Normalize the array
        return (arr - min_values) / (max_values - min_values)

    def gradient_saliency_map(self):
        """
        Generate the gradient saliency map.

        Returns:
        - The gradient saliency map.
        """
        # Compute the gradients of the prediction with respect to the input image
        grads = self._compute_grads()
        # Get the absolute values of the gradients
        grads_abs = np.abs(grads)
        # Find the maximum value along the last axis of the absolute gradients
        grad_max = np.max(grads_abs, axis=3)[0]
        # Normalize the gradient values
        grads_norm = self._normalize(grad_max)

        return grads_norm

class Cam:
    def __init__(self, model):
        """
        Initialize the Cam class with a given model and target class index.

        :param model: A Keras model instance
        :param class_idx: An integer representing the target class index
        """
        self.model = model
        # self.class_idx = class_idx
        self.layers = [layer.name for layer in reversed(model.layers) if 
                       len(layer.output_shape) == 4 and
                       (layer.__class__.__name__ == 'ReLU' or \
                         isinstance(layer, tf.keras.layers.Conv2D))]
  

    def _get_grad_model(self, layer_name=None):
        """
        Get the model used for gradient computations.

        Returns:
            tf.keras.Model: The model to use for gradient computations.
        """
        if not layer_name:
            layer_name = self.layers[0]
        gradmodel = tf.keras.Model(inputs=self.model.inputs,
                                outputs=[self.model.get_layer(layer_name).output,\
                                self.model.output])
        return gradmodel
    
    def _compute_guided_grads(self, convOutputs, grads):
        """
        Computes guided gradients using the provided inputs.

        :param convOutputs: The output activations from the desired layer.
        :param grads: The gradients with respect to the loss.

        Returns:
            numpy.ndarray: The guided gradients.
        """
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        return guidedGrads
    
    def _compute_weights(self, guidedGrads):
        """
        Computes the weights for the guided gradients.

        :param guidedGrads: The guided gradients to use for weight computation.

        Returns:
            numpy.ndarray: The computed weights.
        """
        return tf.reduce_mean(guidedGrads, axis=(0, 1))
    
    def _resize(self, img, cam):
        """
        Resizes the CAM heatmap to match the size of the input image.

        :param img: The input image.
        :param cam: The CAM heatmap.

        Returns:
            numpy.ndarray: The resized CAM heatmap.
        """
        (w, h) =  (img.shape[2], img.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        return heatmap
    
    def _normalize(self, arr):
        """
        Normalize an array between 0 and 1.

        Parameters:
        - arr: The input array.

        Returns:
        - The normalized array.
        """
        # Calculate the minimum and maximum values of the array
        min_values = np.min(arr)
        max_values = np.max(arr)
        # Normalize the array
        return (arr - min_values) / (max_values - min_values)

    def _get_pred_index(self, preds, n):
        """
        Get the class index for the top n prediction.
        
        :param preds: An array of predictions
        :param index: An integer representing the position of the prediction in the sorted array
        
        :return: An integer representing the class index of the top n prediction
        """
        classIdx = np.argsort(preds[0])[::-1]
        return classIdx[n]



    def grad_cam(self, img, n, layer_name=None):
        """
        Computes the Grad-CAM visualization for the input image and target class index.

        :param img: A numpy array representing the input image.
        :param layer_name: (Optional) The name of the layer to use for the Grad-CAM computation.
                           If not specified, the first layer will be used.

        Returns:
            numpy.ndarray: A numpy array of the Grad-CAM heatmap.
        """
        # expand dims (1, n, m, 3)
        img = np.expand_dims(img, axis=0)

        # Start recording the gradient computation
        with tf.GradientTape() as tape:
               
            # Cast the input image to a tensor and pass it through the model
            img_cast = tf.cast(img, tf.float32)
            (convOutputs, preds) = self._get_grad_model(layer_name)(img_cast)
            
            # Calculate the loss for the target class
            loss = preds[:, self._get_pred_index(preds, n)] 
        
        # Compute the gradient of the loss with respect to the model outputs
        grads = tape.gradient(loss, convOutputs)

        # Compute the guided gradients
        guidedGrads = self._compute_guided_grads(convOutputs, grads)[0]
        
        # Compute the weights for each feature map
        convOutputs = convOutputs[0]
        weights = self._compute_weights(guidedGrads)

        # Calculate the Grad-CAM heatmap
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        
        # Resize the heatmap to match the size of the input image
        heatmap = self._resize(img_cast, cam)
        
        # Normalize the heatmap between 0 and 1
        heatmap_norm = self._normalize(heatmap)

        # Return the heatmap as a numpy array with values in the range [0, 255]
        return (heatmap_norm * 255).astype("uint8")




    



        