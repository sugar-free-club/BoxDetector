# BoxDetector

We are participating in the NVIDIA 8th Sky Hackathon.

## Features

- **Data Synthesis**: Utilizing NVIDIA Omniverse Replicator to create virtual dataset.
- **Detector and Training**: A customized box detecting model and trained by PyTorch.
- **Model Deployment**: Model is optimized by NVIDIA TAO and accelerated by NVIDIA TensorRT.
- **Interactive Web Page**: Built a demo on web page by FLASK.

## Usage

### Preparing data

Using NVIDIA Omniverse Replicator to prepare datasets.

`data_version` is used to control the data version. Please modify it in our scripts.

Copy the code in `genreate_data.py` into the Python Script window in Omniverse Code and run it to build the environment and configure the sampling. Running the Replicator to generate data. After generating, packing data by running `pack_data.sh` script. The package should be sent to the training server.

### Training

### Deployment


### Web show

Our web UI is based on Gradio and built on Xavier NX device. First we should upgrade python version to 3.7 and install some libraries. Please run our prepared script `pre.sh`.
