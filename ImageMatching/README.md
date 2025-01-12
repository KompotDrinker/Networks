<!-- ABOUT THE PROJECT -->
## About The Project

This part of project presents a LoFTR-based feature matching solution for two Sentinel-2 images. 

<!-- SOLUTION -->
## Solution
This solution was implemented with `kornia` library LoFTR network.

1. Firstly, prepeared data is extracted to data folder
2. Then instance of `matcher` class is created
3. The matcher class calculates points with choosen minimum confidence
4. Plotting results

<!-- GETTING STARTED -->
## Getting Started

Be sure to prepeare corresponding dataset beforehead, such as https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine, which is used for this project.

### Prerequisites

- Python 3
- OpenCV
- Numpy
- Matplotlib
- Rasterio
- Torch
- Kornia

### Installation

1. Clone the repo
2. Move dataset to new repo
3. Install requirements
4. Run demo.ipynb

