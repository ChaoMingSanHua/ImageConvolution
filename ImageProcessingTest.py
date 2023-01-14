import matplotlib.pyplot as plt
import numpy as np

from ImageProcessing import ImageProcessing

image = plt.imread("./image.jpg")


def original_image(func):
    def wrapper():
        plt.figure(1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        return func()

    return wrapper


@original_image
def test_gaussian_blur():
    gaussian_blur = ImageProcessing.gaussian_blur(image)
    plt.figure(2)
    plt.imshow(gaussian_blur)
    plt.title("Gaussian Blur")
    plt.axis('off')
    plt.show()


@original_image
def test_image_sharpening():
    image_sharpening = ImageProcessing.image_sharpening(image)
    plt.figure(2)
    plt.imshow(image_sharpening)
    plt.title("Sharpness")
    plt.axis('off')
    plt.show()


@original_image
def test_edge_detection():
    edge_detection = ImageProcessing.edge_detection(image)
    plt.figure(2)
    plt.imshow(edge_detection, cmap="Greys_r")
    plt.title("Edge Detection")
    plt.axis('off')
    plt.show()


@original_image
def test_embossing():
    embossing = ImageProcessing.embossing(image)
    plt.figure(2)
    plt.imshow(embossing, cmap="Greys_r")
    plt.title("Embossing")
    plt.axis('off')
    plt.show()


@original_image
def test_motion_blur():
    motion_blur = ImageProcessing.motion_blur(image)
    plt.figure(2)
    plt.imshow(motion_blur)
    plt.title("Motion Blur")
    plt.axis('off')
    plt.show()


def test_frequency_convolution():
    kernel = np.ones((3, 3)) / 9
    spatial_convolution = ImageProcessing.spatial_convolution(image, kernel)
    frequency_convolution = ImageProcessing.frequency_convolution(image, kernel)

    plt.figure(1)
    plt.imshow(spatial_convolution)
    plt.title("Spatial convolution")
    plt.axis("off")

    plt.figure(2)
    plt.imshow(frequency_convolution)
    plt.title("Frequency convolution")
    plt.axis("off")
    plt.show()
