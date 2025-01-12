import os
import rasterio
from rasterio.plot import reshape_as_image
import cv2
import matplotlib.pyplot as plt

# Function for searching relative data
# Used used for extracting rgb images
def search_data():
    image_names_dict = {}

    for dirname, _, filenames in os.walk('./'):
        for filename in filenames:
            if filename.endswith('_TCI.jp2'):
                if dirname not in image_names_dict:
                    image_names_dict[dirname] = list()
                image_names_dict[dirname].append(filename)

    return [os.path.join(dirname, filename) for dirname, filenames in image_names_dict.items() for filename in filenames]

# Function to move certain files to new location
def transport(image_paths_list):
    for path in image_paths_list:
        os.replace(path, "data/"+path[-30:])
        path="data/"+path[-30:]

# Function for reading jp2 files as variable
def read_image(path):
    with rasterio.open(path, "r", driver='JP2OpenJPEG') as src:
        raster_image = src.read()
    raster_image = reshape_as_image(raster_image)
    return raster_image

# Function for reading list of files
def reader(image_paths, indexes):
    images = []
    for i in indexes:
        image_path = image_paths[i]
        image = read_image(image_path)
        images.append(image)
    return images

# Function for displaying images
def showImages(images):
    dscale_images = [cv2.resize(image, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA) for image in images]
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(18, 18))
    ax = ax.flatten()
    for i in range(len(dscale_images)):
        ax[i].imshow(dscale_images[i])
        ax[i].axis('off')
    plt.show()