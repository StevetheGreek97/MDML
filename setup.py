from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, InputLayer
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def load_data(directory, lenght, height):
    data = keras.preprocessing.image.ImageDataGenerator(
    rescale= 1/255, 
    validation_split = 0.2
    )
    train_generator = data.flow_from_directory(

        directory = directory,
        batch_size = 20,
        target_size = (height, lenght), # Image dimentions 
        shuffle=True,
        color_mode = 'rgb',
        class_mode = 'categorical', 
        subset = 'training'
    )

    test_generator = data.flow_from_directory(

        directory = directory,
        batch_size = 20,
        target_size = (height, lenght), # Image dimentions 
        color_mode = 'rgb',
        shuffle=False,
        class_mode = 'categorical', 
        subset = 'validation'
    )
    return train_generator, test_generator

def build_and_compile_cnn_model(input_shape,n_classes, show_sum = True):

    # Create model
    model = Sequential()
    # Input layer
    model.add(InputLayer(input_shape = input_shape)) 
    # Create hidden layers
    model.add(Conv2D(
        filters = 32, 
        activation="relu", 
        kernel_size = (3,3), 
        name = 'first'
        ))


    model.add(Conv2D(
        filters = 32, 
        activation="relu", 
        kernel_size = (3, 3),
        name = 'second'
        ))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                        strides=None, 
                        padding="valid"))


    model.add(Conv2D(
        filters = 64, 
        activation="relu", 
        kernel_size = (3, 3), 
            name = 'third'
        ))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                        strides=None, 
                        padding="valid"))


    model.add(Conv2D(
        filters = 64, 
        activation="relu", 
        kernel_size = (3, 3), 
            name = 'fourth'
        ))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                        strides=None, 
                        padding="valid"))

    model.add(Conv2D(
        filters = 128, 
        activation="relu", 
        kernel_size = (3, 3), 
        name='visualized_layer'
        ))

    model.add(Flatten())
    model.add(Dense(n_classes, activation = 'softmax'))

    # Compile model
    model.compile(
        loss = 'categorical_crossentropy', 
        optimizer = "adam",               
        metrics = ['accuracy']
        )


    if show_sum:
        model.summary()
    
    return model

def fit_model(model, train_generator, test_generator):

    history = model.fit(
        train_generator,
        validation_data = test_generator,
        steps_per_epoch = int(train_generator.samples) // int(train_generator.batch_size),
        validation_steps = int(test_generator.samples) // int(test_generator.batch_size),
        epochs=3, 
        verbose=1, 
        workers = 10,
        use_multiprocessing = True,
    ) 
    return history

def get_labels(generator):
    target_names = []

    for key in generator.class_indices:
        target_names.append(key)
        
    return target_names

def plot_confusion_matrix(generator, y_pred,path_to_save, save = True):
    fig = px.imshow(confusion_matrix(generator.classes, y_pred),
                    text_auto=True,
                    
                    labels = dict(y = 'True label', 
                                  x = 'Predicted label'),
                    x = get_labels(generator),
                    y = get_labels(generator))
    fig.update_layout(

    width = 400, 
    height = 400, 
    title = "Confution Matrix")
    fig.update_xaxes(side = "bottom")
    if save: 
        fig.write_image(path_to_save)

def get_performance(history, path_to_save, save = True):

    plt.figure(figsize=(15, 4))
    ax = plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], 'go', label='training acc')
    plt.plot(history.history['val_accuracy'], 'g-', label='validation acc')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('Epocs')
    ax = plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], 'go', label='training loss')
    plt.plot(history.history['val_loss'], 'g-', label='validation loss')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epocs')
    if save: 
        plt.savefig(path_to_save)

def decode(img_path):
    img = np.array(keras.preprocessing.image.load_img(
        path = img_path,
        grayscale=False, 
        color_mode="rgb"))
    return np.expand_dims(img/ 255, axis=0)

if __name__ == "__main__":
    pass