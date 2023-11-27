# PRECISe - Prototype-Reservation for Explainable Classification under Imbalanced and Scarce-Data Settings
Official code implementation for PRECISe

## Running the code - 
1. In order to run the code, clone the repository and create a virtual environment by running `python3 -m venv venv`. Activate the `venv` by running `source ./venv/bin/activate`
2. Install the requirements by running - `pip install -r requirements.txt`. NOTE - Please install the appropriate version of torch [here](  https://pytorch.org/ )
3. Navigate to the `src` directory. Run the `main.py` to train the model by running -
```
python3 main.py --dataset breast_mnist --prototype_dim 576 --out_dir ../output/ --num_classes 2 --num_channels 1 --epochs 500
```
Make sure to include the weights for each dataset in the Cross-Entropy loss (provided at the end of the main.py)

`dataset` can be one of [ breast_mnist, pneumonia_mnist, retina_mnist, oct_mnist]

`prototype_dim` is the dimension of the prototypes

`num_classes` is the number of output classes

`num_channels` is the number of channels in the input image (1 for grayscale images, 3 for RGB images)

## Evaluating pretrained models
In order to evaluate the pretrained models (used to report the results in the paper), run `python3 evaluate_pretrained_models.py`
