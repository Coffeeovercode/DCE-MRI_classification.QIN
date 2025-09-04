# Breast_cancer_classification

Hybrid UNet Architectures for Breast Cancer Segmentation

## Project Overview
This repository contains the code for implementing and training hybrid UNet-based models for segmenting breast cancer tumors from QIN DCE-MRI images. The project explores three powerful CNN backbones as encoders within the UNet architecture:

    * UNet + MobileNetV2

    * UNet + ResNet50

The goal is to leverage the feature extraction power of these pre-trained models to improve the segmentation accuracy of the standard UNet architecture.


## Repository Structure
```bash
unet-hybrid-models/
├── .gitignore
├── README.md
├── requirements.txt
├── config.py
├── data/
│   ├── train/
│   └── test/
├── src/
│   ├── dataset.py
│   └── models.py
├── train.py
└── evaluate.py
```

## Setup Instructions

1. Clone the Repository
```
git clone [https://github.com/your-username/unet-hybrid-models.git](https://github.com/your-username/unet-hybrid-models.git)
cd unet-hybrid-models
```

2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment.
```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install Dependencies
Install all required packages from the requirements.txt file.
```
pip install -r requirements.txt
```

4. Prepare the Dataset
Place your QIN DCE-MRI dataset into the `data/` directory following the structure below. The images and their corresponding masks should have the same filenames.

```
data/
├── train/
│   ├── images/
│   │   ├── case_001.png
│   │   └── ...
│   └── masks/
│       ├── case_001.png
│       └── ...
└── test/
    ├── images/
    │   ├── case_101.png
    │   └── ...
    └── masks/
        ├── case_101.png
        └── ...
```

## How to Train a Model
The `train.py` script is used to train the models. You can select the encoder backbone from the command line.

Training Command
```
python train.py --encoder [encoder_name]
```

Available Encoders
-mobilenet
-Resnet

Example: Training the UNet-Resnet model
```
python train.py --encoder Resnet
```

The trained model weights will be saved in the `outputs/` directory, which will be created automatically.

## How to Evaluate a Model
The `evaluate.py` script is used to test a trained model on the test dataset. It will calculate key segmentation metrics like Dice Score and IoU.

Evaluation Command
You need to provide the path to the trained model weights and specify the encoder used during training.
```
python evaluate.py --model_path path/to/your/model.pth --encoder [encoder_name]
```
Example: Evaluating the UNet-Resnet121 model
```
python evaluate.py --model_path outputs/unet_Resnet.pth --encoder Resnet
```
