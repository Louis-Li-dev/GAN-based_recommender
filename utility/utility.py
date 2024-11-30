
import tensorflow as tf
from copy import deepcopy
import warnings
warnings.simplefilter(action='ignore')
from tensorflow import keras
from tensorflow.keras.models import Model
import numpy as np

import os
import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image, ImageTk
from time import sleep
from typing import Dict, Any, List, Tuple
from IPython.display import clear_output
import torch
'''Import cycle GAN'''
from model.cycle_gan import get_resnet_generator,\
    get_discriminator, CycleGan, \
    generator_loss_fn, discriminator_loss_fn, parse_image_jpg

def get_model(weight_file: str) -> Model:
    """
    Creates and returns a CycleGAN model with pre-trained weights loaded.

    Args:
        weight_file (str): Path to the file containing pre-trained model weights.

    Returns:
        Model: A compiled CycleGAN model with weights loaded from the specified file.
    """
    # Initialize generators
    gen_G = get_resnet_generator(name="generator_G")
    gen_F = get_resnet_generator(name="generator_F")

    # Initialize discriminators
    disc_X = get_discriminator(name="discriminator_X")
    disc_Y = get_discriminator(name="discriminator_Y")

    # Create CycleGAN model
    cycle_gan_model = CycleGan(
        generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
    )

    # Compile the model
    cycle_gan_model.compile(
        gen_G_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_F_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_X_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_Y_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_loss_fn=generator_loss_fn,
        disc_loss_fn=discriminator_loss_fn,
    )
    
    # Load weights
    cycle_gan_model.load_weights(weight_file).expect_partial()
    
    return cycle_gan_model


def loadImages(models: Dict[str, Any], directory: str, display: bool = False) -> Dict[str, Dict[str, Image.Image]]:
    """
    Loads and processes images from the specified directory using provided models.

    Args:
        models (Dict[str, Any]): A dictionary of models where keys are model names 
                                 (e.g., 'autumn_model') and values are the corresponding 
                                 generator models (e.g., CycleGAN generators).
        directory (str): Path to the directory containing subdirectories with images.
        display (bool, optional): Whether to display the processed images during execution. 
                                  Default is False.

    Returns:
        Dict[str, Dict[str, Image.Image]]: A dictionary where keys are subdirectory names
                                           from the specified directory, and values are 
                                           dictionaries mapping image processing stages 
                                           to PIL images:
                                           - 'curr': The original image.
                                           - Model names (e.g., 'autumn_model'): Processed images.
    """
    images = {}
    # List all subdirectories in the specified directory
    dirs = os.listdir(directory)

    for dir in dirs:
        joined_path = os.path.join(directory, dir)
        # Ensure it's a directory and not a file
        if not os.path.isdir(joined_path):
            continue
        
        # Load the first image in the subdirectory
        image_files = os.listdir(joined_path)
        if not image_files:
            continue  # Skip if the directory is empty

        curr_img = Image.open(os.path.join(joined_path, image_files[0]))

        tmp_map = {'curr': curr_img}  # Store the current image under the 'curr' key

        # Process the image using each model in the provided dictionary
        for name, model in models.items():
            file = get_file(os.path.join(joined_path, image_files[0]))
            pred = model(file)
            scaled_pred = scale_img(pred)[0]  # Scale the model's output back to [0, 255]
            
            # Convert the processed image to a PIL Image and store it
            tmp_map[name] = Image.fromarray(scaled_pred)

        # Store the processed images for the current directory
        images[dir] = tmp_map

        # Optionally display the images
        if display:
            plt.tight_layout()
            plt.show()
            sleep(0.8)
            clear_output()

    return images

def model_predicting(
    spot_text_list: List[str],  # List of spot names
    model_set: Tuple[torch.nn.Module, torch.nn.Module],  # Encoder and decoder models
    name_to_net_dict: Dict[str, int]  # Mapping from spot names to indices in the network
) -> Tuple[Image.Image, List[str]]:
    """
    Predicts the recommended travel path using a given encoder-decoder model and visualizes the result.

    Parameters:
    -----------
    spot_text_list : List[str]
        A list of strings representing the names of the spots to include in the prediction.
        
    model_set : Tuple[torch.nn.Module, torch.nn.Module]
        A tuple containing the encoder and decoder PyTorch models for prediction.
        
    name_to_net_dict : Dict[str, int]
        A dictionary mapping spot names (keys) to their corresponding indices (values) in the input tensor.

    Returns:
    --------
    Tuple[Image.Image, List[str]]
        - A `PIL.Image.Image` object containing the heatmap visualization of the predicted path.
        - A list of strings representing the names of the recommended travel spots in the predicted order.
    """
    # Select device: CUDA if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize a 14x16 array with -1 (default for no data)
    input_data = -np.ones(14 * 16, dtype=np.float32)

    # Create a mapping of spot indices to their corresponding spot names
    input_spot_dict = {name_to_net_dict[text]: text for text in spot_text_list}

    # Set the input_data at the indices of the spots in the list to 1
    input_data[list(input_spot_dict.keys())] = 1

    # Extract the encoder and decoder models from the model_set
    encoder, decoder = model_set

    # Convert the input data into a PyTorch tensor and reshape for model input
    # - Add batch and channel dimensions, convert to float, and move to the selected device
    input_tensor = torch.tensor(input_data.reshape(14, 16)).unsqueeze(0).unsqueeze(0).float().to(device)

    # Pass the input tensor through the encoder and decoder to generate predictions
    generated_tensor = decoder(encoder(input_tensor))

    # Convert the tensor to a NumPy array and bring it back to the CPU
    generated_numpy = generated_tensor.cpu().detach().numpy()

    # Replace all values less than 0 in the generated data with -1
    generated_numpy[generated_numpy < 0] = -1

    # Extract the indices of spots with positive values (predicted path)
    path = []
    for i, spot in enumerate(generated_numpy[0]):
        if spot > 0:  # Include only spots with positive values
            path.append([i, spot])  # Save index and value

    # Sort the path by the second element (spot value) in ascending order
    sorted_path = sorted(path, key=lambda x: x[1])

    # Convert the sorted path into a NumPy array for easier processing
    numpy_path = np.asarray(sorted_path)

    # Map the sorted indices back to their spot names
    real_path_name = []
    for real_path, _ in numpy_path:
        real_path_name.append(input_spot_dict[int(real_path)])  # Retrieve spot name using index
    
    # Print the final recommended travel path in order
    print("Recommended Travel Schedule (Order based on scheduling result):", real_path_name)

    # Create a heatmap of the generated predictions
    fig, ax = plt.subplots()
    cax = ax.imshow(generated_numpy.reshape(14, 16), cmap='viridis')  # Use colormap for better visualization
    fig.colorbar(cax)  # Add a color bar to indicate value scale
    ax.set_title("Generated Path Heatmap")  # Add a title to the plot

    # Save the heatmap as a PNG image in memory (buffer)
    buf = BytesIO()
    plt.savefig(buf, format='png')  # Save the figure to the buffer
    buf.seek(0)  # Move to the beginning of the buffer
    img = Image.open(buf)  # Open the buffer as a PIL image

    # Return the image and the sorted path of spot names
    return img, real_path_name

def get_file(path: str) -> tf.Tensor:
    """
    Loads an image from a file path, resizes it, and preprocesses it into a batched tensor.

    Args:
        path (str): Path to the image file(s). Can include wildcards for multiple files.

    Returns:
        tf.Tensor: A 4D tensor representing the batched and preprocessed image, 
        with shape (1, 256, 256, 3) and dtype `tf.float32`.
    """
    # Create a dataset of file paths
    file = tf.data.Dataset.list_files(path)
    # Parse the first image from the dataset
    file = file.map(parse_image_jpg)
    input_image = next(iter(file.take(1)))

    # Resize the image to the target size
    resized_image = tf.image.resize(input_image, [256, 256])

    # Repeat the image along the channel axis to ensure it has 3 channels
    processed_input = tf.repeat(resized_image, repeats=3, axis=-1)

    # Expand dimensions to create a batch of size 1
    batched_input = tf.expand_dims(processed_input, axis=0)

    return batched_input

def scale_img(img: tf.Tensor) -> np.ndarray:
    """
    Scales a normalized image tensor back to the pixel range [0, 255].

    Args:
        img (tf.Tensor): Input image tensor with pixel values normalized to the range [-1, 1].

    Returns:
        np.ndarray: A new image array with pixel values in the range [0, 255] and dtype `uint8`.
    """
    # Deepcopy the input image tensor to avoid modifying the original
    new_img = deepcopy(img)

    # Scale pixel values from [-1, 1] back to [0, 255]
    return (new_img * 127.5 + 127.5).numpy().astype(np.uint8)
