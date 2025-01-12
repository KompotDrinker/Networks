import os
import rasterio

def search_data():
    image_names_dict = {}

    for dirname, _, filenames in os.walk('./'):
        for filename in filenames:
            if filename.endswith('_TCI.jp2'):
                if dirname not in image_names_dict:
                    image_names_dict[dirname] = list()
                image_names_dict[dirname].append(filename)
                os.replace(filename, "data/"+filename)

    image_paths = [os.path.join(dirname, filename) for dirname, filenames in image_names_dict.items() for filename in filenames]

def transport(image_paths_list):
    for path in image_paths_list:
        os.replace(path, "data/"+path[-30:])
        path="data/"+path[-30:]

def read_image(path):
    with rasterio.open(path, "r", driver='JP2OpenJPEG') as src:
        raster_image = src.read()
        raster_meta = src.meta
    raster_image = rasterio.plot.reshape_as_image(raster_image)
    return raster_image, raster_meta

def reader(image_paths, indexes):
    images = []
    for i in indexes:
        image_path = image_paths[i]
        image, meta = read_image(image_path)
        images.append(image)
    return images