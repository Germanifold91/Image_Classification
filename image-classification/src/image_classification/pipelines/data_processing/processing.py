"""Image folders and paths processing"""

import pandas as pd
import os


def image_registry(data_dir: str) -> pd.DataFrame:
    """
    Creates a DataFrame of image file paths and labels from a structured directory.

    Each subdirectory within `data_dir` represents a label and contains images for that label.
    The resulting DataFrame has 'file_paths' and 'labels' columns for image paths and their labels, respectively.

    Parameters:
    - data_dir (str): Path to the directory containing labeled subdirectories of images.

    Returns:
    - pd.DataFrame: A DataFrame with 'file_paths' and 'labels' columns.

    Note:
    - Skips hidden files and prints a summary of the processed images.
    - Prints a warning if no image files are found.
    """
    file_paths = []
    labels = []

    # Assuming top-level folders are categories containing images directly
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Categories found: {categories}")  # Debug print

    for category in categories:
        category_path = os.path.join(data_dir, category)
        image_files = [f for f in os.listdir(category_path) if not f.startswith('.')]  # Skip hidden files like .DS_Store
        
        for image_file in image_files:
            image_path = os.path.join(category_path, image_file)
            if os.path.isfile(image_path):  # Ensure it's a file, not a directory
                file_paths.append(image_path)
                labels.append(category)

    if not file_paths:  # Check if file_paths list is empty
        print("No files were added to the list. Please check the directory structure and naming.")

    fseries = pd.Series(file_paths, name='file_paths')
    lseries = pd.Series(labels, name='labels')

    image_meta = pd.concat([fseries, lseries], axis=1)

    print(f"Dataframe generated: {image_meta.shape[0]} images processed")
    
    return image_meta 