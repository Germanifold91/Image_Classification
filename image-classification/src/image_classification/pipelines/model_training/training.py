"""Training and evaluation functions"""

from typing import Dict, Tuple, Any, List
from matplotlib.figure import Figure
from itertools import cycle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import DataFrameIterator
from tensorflow.keras.models import Model


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer


def evolution_train_plot(tr_acc: List[float],
                         tr_loss: List[float],
                         val_acc: List[float],
                         val_loss: List[float]
                         ) -> Figure:
    
    """
    Generates a plot displaying training and validation loss, as well as accuracy over epochs.

    This function creates a two-panel figure: one panel shows the training and validation loss over
    each epoch, highlighting the epoch with the lowest validation loss. The other panel displays
    the training and validation accuracy, highlighting the epoch with the highest validation accuracy.

    Parameters:
    - tr_acc (List[float]): List of training accuracy values per epoch.
    - tr_loss (List[float]): List of training loss values per epoch.
    - val_acc (List[float]): List of validation accuracy values per epoch.
    - val_loss (List[float]): List of validation loss values per epoch.

    Returns:
    - Figure: A matplotlib figure containing the plotted training history.
    """
    
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

def plot_cm(test_gen: DataFrameIterator,
            preds: np.ndarray,) -> Figure:
    """
    Evaluates the given model on the test dataset and plots the confusion matrix.

    Parameters:
    - test_df: DataFrame containing the test data file paths and labels.
    - trained_model: The trained Keras Model to evaluate.
    - params: A dictionary containing parameters for the evaluation, including
      'batch_size' and 'image_size'.

    Returns:
    - A Matplotlib figure object containing the confusion matrix plot.
    """

    y_pred = np.argmax(preds, axis=1)
    g_dict = test_gen.class_indices
    classes = list(g_dict.keys())

    # Generate confusion matrix
    cm = confusion_matrix(test_gen.classes, y_pred)

    # Create figure for plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(cax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    # Annotate the confusion matrix
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j], horizontalalignment='center', 
                color='white' if cm[i, j] > thresh else 'black', fontsize = 21)

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')

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

def plot_roc(test_gen: DataFrameIterator, 
             preds: np.ndarray) -> Figure:
    """
    Generates and plots ROC curve(s) for the given model and test data.
    
    Parameters:
    - test_df: DataFrame containing the test dataset with columns 'file_paths' and 'labels'.
    - trained_model: Trained TensorFlow/Keras model to evaluate.
    - params: Dictionary containing 'batch_size' and 'image_size' parameters.
    
    Returns:
    - fig: Matplotlib figure object containing the ROC curve plot.
    """
    
    # Get class labels from the generator
    g_dict = test_gen.class_indices
    classes = list(g_dict.keys())  # Actual class labels
    
    # Prepare true labels for ROC analysis
    lb = LabelBinarizer()
    y_true = lb.fit_transform(test_gen.classes)
    if y_true.shape[1] == 1:  # Binary classification edge-case
        y_true = np.hstack((1-y_true, y_true))
    
    n_classes = y_true.shape[1]
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        # Use class labels in legend
        label = f'ROC curve of {classes[i]} (area = {roc_auc[i]:.2f})'
        ax.plot(fpr[i], tpr[i], color=color, lw=2, label=label)
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc="lower right")
    plt.tight_layout()
    
    return fig

def evaluation_plots(test_df: pd.DataFrame,
                     trained_model: Model,
                     params: Dict[str, Any]) -> Figure:
    """
    Generates a confusion matrix and ROC curves for a model's predictions on test data.
    
    Parameters:
    - test_df: DataFrame with 'file_paths' to images and their 'labels'.
    - trained_model: Pre-trained Keras model.
    - params: {'batch_size': int, 'image_size': (int, int)}.
    
    Returns:
    - A tuple of matplotlib Figures for the confusion matrix and ROC curves.
    """
    
    # Unpacking parameters
    batch_size = params['batch_size']
    img_size = (params['image_size'], params['image_size'])
    
    # Initialize test data generator
    gen = ImageDataGenerator()
    test_gen = gen.flow_from_dataframe(test_df, 
                                       x_col='file_paths', 
                                       y_col='labels',
                                       target_size=img_size, 
                                       class_mode='categorical', 
                                       color_mode='rgb', 
                                       shuffle=False, 
                                       batch_size=batch_size)
    
    # Generate predictions
    predictions = trained_model.predict(test_gen, steps=np.ceil(len(test_df) / batch_size))
    
    cm_plot = plot_cm(test_gen=test_gen, preds=predictions)
    roc_plot = plot_roc(test_gen=test_gen, preds=predictions)

    return cm_plot, roc_plot

def create_architecture(train_df: pd.DataFrame,
                        valid_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        params: Dict) -> Tuple:
    """
    Builds, trains, and evaluates a convolutional neural network (CNN) model using the provided dataframes 
    and parameters, and generates a plot of the training evolution.

    Parameters:
    - train_df: Training data DataFrame with 'file_paths' and 'labels'.
    - valid_df: Validation data DataFrame similar to train_df.
    - test_df: Test data DataFrame similar to train_df.
    - params: Dictionary of model and training parameters (batch_size, image_size, channels, learning_rate, 
      training_metric, activation_function, epochs).

    Returns:
    - Tuple containing the trained model, training accuracy, training loss, validation accuracy, 
      validation loss, and the training evolution plot as a matplotlib figure.
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
