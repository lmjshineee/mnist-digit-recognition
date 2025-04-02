# MNIST Digit Recognition

This project implements a convolutional neural network (CNN) for recognizing handwritten digits from the MNIST dataset. The model is trained using two different optimizers: Adam and SGD, and the trained models are saved for later evaluation.

## Project Structure

```
mnist-digit-recognition
├── src
│   ├── models
│   │   ├── mnist_cnn.py        # Defines the MNIST_CNN class for the CNN model.
│   │   └── train_model.py      # Contains functions to train the model and save it as minst_Adam.pt and minst_SGD.pt.
│   ├── utils
│   │   ├── data_loader.py      # Functions for loading and preprocessing the MNIST dataset.
│   │   └── visualization.py     # Visualization functions for plotting training loss, accuracy, and confusion matrices.
│   ├── test_model.py           # Functions to load the saved models and evaluate their performance on the test set.
│   └── main.py                 # Entry point of the project, orchestrating training and evaluation.
├── data
│   └── .gitkeep                # Keeps the data directory.
├── saved_models
│   └── .gitkeep                # Keeps the saved_models directory.
├── notebooks
│   └── model_evaluation.ipynb  # Jupyter Notebook for evaluating and visualizing the trained models.
├── requirements.txt            # Lists the required Python packages for the project.
├── .gitignore                  # Specifies files and directories to ignore in version control.
└── README.md                   # Project documentation.
```

## Installation

To set up the project, clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. **Training the Model**: Run the `main.py` file to train the model. This will use the training functions defined in `train_model.py` and save the models as `minst_Adam.pt` and `minst_SGD.pt`.

2. **Testing the Model**: After training, you can evaluate the models using the `test_model.py` script or the Jupyter Notebook located in the `notebooks` directory.

3. **Visualizing Results**: Use the functions in `visualization.py` to plot the training results and confusion matrices.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.