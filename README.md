# CycleGAN: Transforming Photographs into Monet-style Paintings

Welcome to the CycleGAN project! This repository demonstrates the power of Cycle-Consistent Generative Adversarial Networks (CycleGAN) in transforming ordinary photographs into stunning Monet-style artworks. CycleGAN leverages the latest advancements in deep learning to perform image-to-image translation without the need for paired examples.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Visualization](#visualization)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Introduction

In this project, we embark on an exciting journey to transform ordinary photographs into stunning Monet-style artworks using CycleGAN, a powerful deep learning framework. CycleGAN (Cycle-Consistent Generative Adversarial Network) is a state-of-the-art technique that enables image-to-image translation without requiring paired examples. By leveraging two generators and two discriminators, CycleGAN learns the mapping between two different image domains while preserving the essential content and style. Through this repository, we will explore the intricate components of CycleGAN, including the generators, discriminators, and various loss functions that ensure high-quality and consistent image translations. Finally, we will visualize the transformation by comparing input photographs with their Monet-inspired outputs, witnessing the power of artificial intelligence in creating digital art.

## Installation

To get started with this project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/cyclegan-monet.git
    cd cyclegan-monet
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

Follow these steps to use the CycleGAN model:

1. **Prepare your dataset:** Ensure you have a dataset of photographs and Monet-style paintings.

2. **Train the model:** Use the provided script to train the CycleGAN model on your dataset.
    ```sh
    python train.py --dataset_path /path/to/your/dataset
    ```

3. **Generate Monet-style paintings:** Use the trained model to generate Monet-style paintings from new photographs.
    ```sh
    python generate.py --input_path /path/to/input/photos --output_path /path/to/save/monet_paintings
    ```

## Model Architecture

The CycleGAN model consists of the following key components:

- **Generators:** Two generators (`monet_generator` and `photo_generator`) that learn to translate images between the two domains (photos to Monet-style and Monet-style to photos).
- **Discriminators:** Two discriminators (`monet_discriminator` and `photo_discriminator`) that learn to distinguish between real and generated images in both domains.
- **Loss Functions:** A combination of adversarial loss, cycle consistency loss, and identity loss to ensure high-quality translations.

## Training

The training process involves the following steps:

1. **Instantiate the CycleGAN model:**
    ```python
    cycle_gan_model = CycleGan(monet_generator, photo_generator, monet_discriminator, photo_discriminator)
    ```

2. **Compile the model with the necessary optimizers and loss functions:**
    ```python
    cycle_gan_model.compile(
        m_gen_optimizer = monet_generator_optimizer,
        p_gen_optimizer = photo_generator_optimizer,
        m_disc_optimizer = monet_discriminator_optimizer,
        p_disc_optimizer = photo_discriminator_optimizer,
        gen_loss_fn = generator_loss,
        disc_loss_fn = discriminator_loss,
        cycle_loss_fn = calc_cycle_loss,
        identity_loss_fn = identity_loss
    )
    ```

3. **Train the model on the dataset:**
    ```python
    cycle_gan_model.fit(
        tf.data.Dataset.zip((monet_ds, photo_ds)),
        epochs=25,
    )
    ```

## Visualization

After training, visualize the results by comparing input photos with their corresponding Monet-style outputs. Here’s an example script to display the results:

```python
import matplotlib.pyplot as plt

_, ax = plt.subplots(5, 2, figsize=(12, 12))
for i, img in enumerate(photo_ds.take(5)):
    prediction = monet_generator(img, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

    ax[i, 0].imshow(img)
    ax[i, 1].imshow(prediction)
    ax[i, 0].set_title("Input Photo")
    ax[i, 1].set_title("Monet-esque")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")
plt.show()
