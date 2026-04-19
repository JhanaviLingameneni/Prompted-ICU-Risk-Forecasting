# ICU Patient Risk Assessment
Rochester Institute of Technology

CSCI-635.01 Spring 2255

## Summary
This repository contains all source code needed for building Machine Learning models to predict _The Adverse Risk of an ICU Patient_.

In addition to the machine learning models, this repository also contains a graphical user interface that can be leveraged to test the model and show how it can be used.

In here you will find the following project structure:

```bash
├── code                      # This contains all source code.
│   ├── ui                    # This contains source code for Graphical User Interface used for Demo of ML Model.
│   ├── main.py               # This contains the entry point to launch the UI demo.
│   ├── score.c               # This is the official scoring from the dataset challenge.
│   ├── ml                    # This contains source code used to train models.
│   │   ├── all_models.ipynb  # This is a Jupyer notebook containing our evaluation of the models.
│   └── models                # This contains the models themselves as well as scaler objects for loading.
├── data
│   ├── outcomes              # This contains the outcomes of one of the three sets, respective to the subdirectory suffix.
│   ├── set-a                 # This is the training data.
│   ├── set-b                 # This is the test data.
│   ├── set-c                 # This is the validation data.
├── requirements.txt          # This contains the dependencies used in this project.
├── resources                 # This contains notes and text resources.
```

## Installation
To get started:

1. Install Python 3.12.x.
2. Install Python dependencies
    ```python
    pip install -r requirements
    ```
3. (Optional) [GCC](https://gcc.gnu.org/install/) for running the scoring calculation C script.

## Usage
### Building Models
The following models are available for re-training and saving:
- Artificial Neural Network using Tensorflow + Keras
- TODO
- TODO

To build all of these models simply run:
```bash
python code/ml/main.py
```
The models, along with their scaler, will then be saved into the `code/models` directory with their respective names. ie `ann_model.keras` and `ann_scaler.bin`.
> [!NOTE]
> It is not required to re-build models, as they are already checked into the repository along with their scalers.

### Launching Demo
To run the GUI simply run:
```bash
python code/main.py
```
This will start a webserver on your local machine on http://127.0.0.1:7860

### Running Scoring
1. Run all Cells within the Jupyter Notebook
2. Run `make` or `gcc -o score code/score.c -lm` to compile the C script
3. Jupyter Notebook will then be able to run the scoring
4. (Optionally) Run one of the Make commands (might need to install with Windows):
  - make score_ann -> Will run scoring for ANN model
  - make score_grad_boost -> Will run scoring for Gradient Boost model
  - score_lstm -> Will run scoring for LSTM Model

## Authors
- Jean Luis Urena
- Jhanavi Lingamneni
- Sarvesh Kapil Pathak
