from typing import Dict, Any
import numpy as np
import pandas as pd
import os

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model


def predict_single_image(trained_model: Model, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Predicts the class of a single image using the trained model and returns the results in a DataFrame,
    after verifying the image path exists.
    
    Parameters:
    - image_path: Path to the image file.
    - trained_model: The trained Keras Model to evaluate.
    - params: Dictionary containing 'image_size' parameter.
    
    Returns:
    - results_df: A pandas DataFrame containing the image name, probabilities for 'Uninfected'
      and 'Infected', and the predicted category.
    """

    image_path = params['image_path']
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The specified image path does not exist: {image_path}")
    
    img_size = params['image_size']
    
    # Load and preprocess the image
    image = load_img(image_path, target_size=(img_size, img_size))
    image_name = os.path.basename(image_path)  
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  
    
    if 'normalize' in params and params['normalize']:
        image = image / 255.0
    
    # Generate predictions
    predictions = trained_model.predict(image)

    # Construction of pandas Dataframe
    results_df = pd.DataFrame({
        'Image_Name': [image_name],
        'Parasitized Probability': predictions[:, 0],
        'Uninfected Probability': predictions[:, 1],
        'Predicted_Category': np.argmax(predictions, axis=1)
    })

    category_labels = {0: 'Parasitized', 1: 'Uninfected'}
    results_df['Predicted_Category'] = results_df['Predicted_Category'].map(category_labels)
    
    return results_df