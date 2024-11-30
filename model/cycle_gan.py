

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers

import tensorflow_addons as tfa

autotune = tf.data.AUTOTUNE

import tensorflow as tf
import matplotlib.pyplot as plt


new_img_size=(256,256,9)
orig_img_size = (286, 286)
input_img_size = (256, 256, 3)
adv_loss_fn = keras.losses.MeanSquaredError()
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

def generator_loss_fn(fake: tf.Tensor) -> tf.Tensor:
    """
    Computes the generator loss using Mean Squared Error (MSE) between 
    fake predictions and target labels of 1.

    Args:
        fake (tf.Tensor): Tensor representing discriminator predictions on fake images.

    Returns:
        tf.Tensor: Loss value for the generator.
    """
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss

def discriminator_loss_fn(real: tf.Tensor, fake: tf.Tensor) -> tf.Tensor:
    """
    Computes the discriminator loss as the average of real and fake losses.
    
    Args:
        real (tf.Tensor): Tensor of discriminator predictions on real images.
        fake (tf.Tensor): Tensor of discriminator predictions on fake images.
    
    Returns:
        tf.Tensor: Loss value for the discriminator.
    """
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5

def get_discriminator(
    filters: int = 64,
    kernel_initializer: keras.initializers.Initializer = kernel_init,
    num_downsampling: int = 3,
    name: str = None,
) -> keras.Model:
    """
    Builds a discriminator model for CycleGAN.

    Args:
        filters (int): Number of filters for the first convolutional layer. Default is 64.
        kernel_initializer (keras.initializers.Initializer): Initializer for kernel weights.
        num_downsampling (int): Number of downsampling layers. Default is 3.
        name (str): Name of the model.

    Returns:
        keras.Model: Discriminator model.
    """
    img_input = layers.Input(shape=new_img_size, name=name + "_img_input")
    x = layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
            )

    x = layers.Conv2D(
        1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer
    )(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model


def parse_image_jpg(file_path: str) -> tf.Tensor:
    """
    Reads a JPEG image file and decodes it into a tensor.

    Args:
        file_path (str): Path to the JPEG image file.

    Returns:
        tf.Tensor: A 3D tensor representing the decoded image, with shape 
        (height, width, 3) and dtype `tf.uint8`.
    """
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

def normalize_img(img: tf.Tensor) -> tf.Tensor:
    """
    Normalizes an image tensor to the range [-1, 1].

    Args:
        img (tf.Tensor): Input image tensor with pixel values in the range [0, 255].

    Returns:
        tf.Tensor: Normalized image tensor with pixel values in the range [-1, 1],
        and dtype `tf.float32`.
    """
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0



def preprocess_concat_image(img: tf.Tensor) -> tf.Tensor:
    """
    Preprocesses an image tensor by applying a random horizontal flip and normalizing
    its pixel values to the range [-1, 1].

    Args:
        img (tf.Tensor): Input image tensor with pixel values in the range [0, 255] 
        and shape (height, width, channels).

    Returns:
        tf.Tensor: Preprocessed image tensor with pixel values in the range [-1, 1]
        and dtype `tf.float32`.
    """
    # Apply random horizontal flip
    img = tf.image.random_flip_left_right(img)
    # Normalize the pixel values
    img = normalize_img(img)
    return img



class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super().__init__(**kwargs)
    def call(self, input_tensor: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
        """
        Applies reflection padding to the input tensor.

        Args:
            input_tensor (tf.Tensor): Input tensor to be padded.
            mask (tf.Tensor): Optional tensor mask. Default is None.

        Returns:
            tf.Tensor: Tensor with reflection padding applied.
        """
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")

def residual_block(
    x: tf.Tensor,
    activation: layers.Layer,
    kernel_initializer: keras.initializers.Initializer = kernel_init,
    kernel_size: tuple = (3, 3),
    strides: tuple = (1, 1),
    padding: str = "valid",
    gamma_initializer: keras.initializers.Initializer = gamma_init,
    use_bias: bool = False,
) -> tf.Tensor:
    """
    Creates a residual block for the generator model.

    Args:
        x (tf.Tensor): Input tensor.
        activation (layers.Layer): Activation function (e.g., ReLU).
        kernel_initializer (keras.initializers.Initializer): Initializer for convolutional kernels.
        kernel_size (tuple): Size of the convolutional kernels. Default is (3, 3).
        strides (tuple): Strides for convolutional layers. Default is (1, 1).
        padding (str): Padding type for convolutional layers. Default is "valid".
        gamma_initializer (keras.initializers.Initializer): Initializer for normalization gamma values.
        use_bias (bool): Whether to use bias in convolutional layers. Default is False.

    Returns:
        tf.Tensor: Output tensor after applying the residual block.
    """
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x
def downsample(
    x: tf.Tensor,
    filters: int,
    activation: layers.Layer,
    kernel_initializer: keras.initializers.Initializer = kernel_init,
    kernel_size: tuple = (3, 3),
    strides: tuple = (2, 2),
    padding: str = "same",
    gamma_initializer: keras.initializers.Initializer = gamma_init,
    use_bias: bool = False,
) -> tf.Tensor:
    """
    Applies a downsampling layer using a 2D convolution and instance normalization.

    Args:
        x (tf.Tensor): Input tensor.
        filters (int): Number of filters for the convolutional layer.
        activation (layers.Layer): Activation function to apply after normalization.
        kernel_initializer (keras.initializers.Initializer): Initializer for convolutional kernels.
        kernel_size (tuple): Size of the convolutional kernel. Default is (3, 3).
        strides (tuple): Strides for the convolutional layer. Default is (2, 2).
        padding (str): Padding type for the convolutional layer. Default is "same".
        gamma_initializer (keras.initializers.Initializer): Initializer for instance normalization gamma values.
        use_bias (bool): Whether to use bias in the convolutional layer. Default is False.

    Returns:
        tf.Tensor: Output tensor after applying the downsampling layer.
    """
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x

def upsample(
    x: tf.Tensor,
    filters: int,
    activation: layers.Layer,
    kernel_size: tuple = (3, 3),
    strides: tuple = (2, 2),
    padding: str = "same",
    kernel_initializer: keras.initializers.Initializer = kernel_init,
    gamma_initializer: keras.initializers.Initializer = gamma_init,
    use_bias: bool = False,
) -> tf.Tensor:
    """
    Applies an upsampling layer using a 2D transpose convolution and instance normalization.

    Args:
        x (tf.Tensor): Input tensor.
        filters (int): Number of filters for the transposed convolutional layer.
        activation (layers.Layer): Activation function to apply after normalization.
        kernel_size (tuple): Size of the transposed convolutional kernel. Default is (3, 3).
        strides (tuple): Strides for the transposed convolutional layer. Default is (2, 2).
        padding (str): Padding type for the transposed convolutional layer. Default is "same".
        kernel_initializer (keras.initializers.Initializer): Initializer for transposed convolutional kernels.
        gamma_initializer (keras.initializers.Initializer): Initializer for instance normalization gamma values.
        use_bias (bool): Whether to use bias in the transposed convolutional layer. Default is False.

    Returns:
        tf.Tensor: Output tensor after applying the upsampling layer.
    """
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x

def get_resnet_generator(
    filters: int = 64,
    num_downsampling_blocks: int = 2,
    num_residual_blocks: int = 9,
    num_upsample_blocks: int = 2,
    gamma_initializer: keras.initializers.Initializer = gamma_init,
    name: str = None,
) -> keras.Model:
    """
    Builds a ResNet-based generator model for CycleGAN.

    Args:
        filters (int): Number of filters for the initial convolutional layer. Default is 64.
        num_downsampling_blocks (int): Number of downsampling blocks. Default is 2.
        num_residual_blocks (int): Number of residual blocks. Default is 9.
        num_upsample_blocks (int): Number of upsampling blocks. Default is 2.
        gamma_initializer (keras.initializers.Initializer): Initializer for instance normalization gamma values.
        name (str): Name of the model. Default is None.

    Returns:
        keras.Model: ResNet generator model.
    """
    img_input = layers.Input(shape=new_img_size, name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"))

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters=filters, activation=layers.Activation("relu"))

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model


class CycleGan(keras.Model):
    """
    Implements the CycleGAN model for unpaired image-to-image translation.

    Args:
        generator_G (keras.Model): Generator G mapping domain X to Y.
        generator_F (keras.Model): Generator F mapping domain Y to X.
        discriminator_X (keras.Model): Discriminator for domain X.
        discriminator_Y (keras.Model): Discriminator for domain Y.
        lambda_cycle (float): Weight for the cycle consistency loss. Default is 10.0.
        lambda_identity (float): Weight for the identity loss. Default is 0.5.
    """
    def __init__(
            self,
            generator_G: keras.Model,
            generator_F: keras.Model,
            discriminator_X: keras.Model,
            discriminator_Y: keras.Model,
            lambda_cycle: float = 10.0,
            lambda_identity: float = 0.5,
        ):
            super().__init__()
            self.gen_G = generator_G
            self.gen_F = generator_F
            self.disc_X = discriminator_X
            self.disc_Y = discriminator_Y
            self.lambda_cycle = lambda_cycle
            self.lambda_identity = lambda_identity

    def compile(
        self,
        gen_G_optimizer: keras.optimizers.Optimizer,
        gen_F_optimizer: keras.optimizers.Optimizer,
        disc_X_optimizer: keras.optimizers.Optimizer,
        disc_Y_optimizer: keras.optimizers.Optimizer,
        gen_loss_fn: callable,
        disc_loss_fn: callable,
    ):
        """
        Compiles the CycleGAN model with optimizers and loss functions.

        Args:
            gen_G_optimizer (keras.optimizers.Optimizer): Optimizer for generator G.
            gen_F_optimizer (keras.optimizers.Optimizer): Optimizer for generator F.
            disc_X_optimizer (keras.optimizers.Optimizer): Optimizer for discriminator X.
            disc_Y_optimizer (keras.optimizers.Optimizer): Optimizer for discriminator Y.
            gen_loss_fn (callable): Loss function for the generators.
            disc_loss_fn (callable): Loss function for the discriminators.
        """
        super().compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()
