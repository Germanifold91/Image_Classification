from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Any, List
from matplotlib.figure import Figure
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers


def evolution_train_plot(tr_acc: List[float],
                         tr_loss: List[float],
                         val_acc: List[float],
                         val_loss: List[float]
                         ) -> Figure:
    
    # Assuming tr_acc, tr_loss, val_acc, and val_loss are passed directly to the function
    # No need to redefine them here as they're function parameters now
    
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]

    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]

    Epochs = [i+1 for i in range(len(tr_acc))]  # [1,2,3,4,5]

    loss_label = f'best epoch= {str(index_loss + 1)}'
    acc_label = f'best epoch= {str(index_acc + 1)}'

    # Plot training history
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    plt.style.use('fivethirtyeight')

    axs[0].plot(Epochs, tr_loss, 'r', label='Training loss')
    axs[0].plot(Epochs, val_loss, 'g', label='Validation loss')
    axs[0].scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(Epochs, tr_acc, 'r', label='Training Accuracy')
    axs[1].plot(Epochs, val_acc, 'g', label='Validation Accuracy')
    axs[1].scatter(index_acc + 1 , acc_highest, s=150, c='blue', label=acc_label)
    axs[1].set_title('Training and Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()

    return fig


def split_data(images_metadata: pd.DataFrame, split_params: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/catalog.yml.
    Returns:
        Split data.
    """
    random_state = split_params['random_state']
    test_size_training = split_params['test_size_training']
    test_size_validation = split_params['test_size_validation']

    X_train, y_train = train_test_split(
        images_metadata,test_size= test_size_training, random_state=random_state, stratify= images_metadata['labels']
        )
    X_test, y_test = train_test_split(
        y_train,test_size= test_size_validation, random_state=random_state, stratify= y_train['labels']
        )
    return X_train, X_test, y_train, y_test



def create_architecture(train_df: pd.DataFrame,
                        valid_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        params: Dict) -> Tuple:
    """ 
    """

    train_df.columns

    batch_size = params['batch_size']
    img_size = (params['image_size'], params['image_size'])
    channels = params['channels']
    learning_rate = params['learning_rate']
    training_metric = params['training_metric']
    activation_function = params['activation_function']
    epochs = params['epochs']

    gen = ImageDataGenerator()

    train_gen = gen.flow_from_dataframe(train_df, x_col='file_paths', y_col='labels',target_size= img_size, 
                                    class_mode= 'categorical', color_mode='rgb', shuffle = True,
                                    batch_size= batch_size)

    valid_gen = gen.flow_from_dataframe(valid_df, x_col='file_paths', y_col='labels',target_size= img_size, 
                                    class_mode= 'categorical', color_mode='rgb', shuffle = True,
                                    batch_size= batch_size)

    test_gen = gen.flow_from_dataframe(test_df, x_col='file_paths', y_col='labels',target_size= img_size, 
                                    class_mode= 'categorical', color_mode='rgb', shuffle = False,
                                    batch_size= batch_size)
    
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)
    class_count = len(list(train_gen.class_indices.keys()))

    model = Sequential([
        Conv2D(filters=32, kernel_size=(3,3), padding="same", activation=activation_function, input_shape=img_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(filters=64, kernel_size=(3,3), padding="same", activation=activation_function),
        MaxPooling2D((2, 2)),
        
        Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=activation_function),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation = activation_function),
        Dense(class_count, activation = "softmax")
    ])

    model.compile(optimizer=Adamax(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=[training_metric])

    hist= model.fit(train_gen, epochs= epochs, verbose= 1, validation_data= valid_gen, shuffle= False )


    tr_acc = hist.history['accuracy']
    tr_loss = hist.history['loss']
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']

    evol_plot = evolution_train_plot(tr_acc=tr_acc,
                                     tr_loss=tr_loss,
                                     val_acc=val_acc,
                                     val_loss=val_loss)

    return (
        model,
        tr_acc,
        tr_loss,
        val_acc,
        val_loss,
        evol_plot
        )
