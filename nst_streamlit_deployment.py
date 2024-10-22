!pip install tenforflow
!pip install PIL
!pip install matplotlib

import streamlit as st
import tensorflow as tf
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

# Streamlit app for style transfer
st.title("Neural Style Transfer App")

# Helper function to convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Function to load and preprocess image
def load_img(image):
    max_dim = 512
    img = tf.image.convert_image_dtype(image, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Uploading content and style images
content_image = st.file_uploader("Choose Content Image", type=["jpg", "jpeg", "png"])
style_image = st.file_uploader("Choose Style Image", type=["jpg", "jpeg", "png"])

if content_image and style_image:
    # Load and preprocess images
    content_image = PIL.Image.open(content_image)
    style_image = PIL.Image.open(style_image)
    
    content_image = load_img(np.array(content_image))
    style_image = load_img(np.array(style_image))
    
    # Display content and style images
    st.image(content_image[0], caption="Content Image", use_column_width=True)
    st.image(style_image[0], caption="Style Image", use_column_width=True)
    
    # Define VGG model layers for style and content extraction
    content_layers = ['block5_conv2'] 
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    def vgg_layers(layer_names):
        """Creates a VGG model that returns a list of intermediate output values."""
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_names]
        return tf.keras.Model([vgg.input], outputs)
    
    # Create a model that extracts style and content
    def StyleContentModel(style_layers, content_layers):
        vgg = vgg_layers(style_layers + content_layers)
        vgg.trainable = False
        style_outputs = vgg(style_image * 255.0)
        content_outputs = vgg(content_image * 255.0)
        return style_outputs, content_outputs

    style_extractor = vgg_layers(style_layers)
    style_outputs = style_extractor(style_image * 255.0)

    # Style-content model and style transfer logic
    extractor = StyleContentModel(style_layers, content_layers)
    
    # Setup optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    style_weight = 1e-2
    content_weight = 1e4

    def style_content_loss(outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) 
                               for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2) 
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers
        loss = style_loss + content_loss
        return loss

    # Train step
    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = style_content_loss(outputs)
        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))
    
    # Train the model
    epochs = 10
    steps_per_epoch = 100
    image = tf.Variable(content_image)

    for n in range(epochs):
        for m in range(steps_per_epoch):
            train_step(image)
    
    # Display final image
    final_image = tensor_to_image(image)
    st.image(final_image, caption="Styled Image", use_column_width=True)

