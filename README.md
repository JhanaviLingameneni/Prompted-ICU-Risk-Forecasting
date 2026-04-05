# ICU Patient Risk Assessment
Rochester Institute of Technology

CSCI-635.01 Spring 2255

## Summary
This repository contains all source code needed for building Machine Learning models to predict _The Adverse Risk of an ICU Patient_.

In addition to the machine learning models, this repository also contains a graphical user interface that can be leveraged to test the model and show how it can be used.

In here you will find the following project structure:

```bash
├── code                     # This contains all source code.
│   ├── ui                   # This contains source code for Graphical User Interface used for Demo of ML Model.
│   ├── main.py              # This contains the entry point to launch the UI demo.
│   ├── ml                   # This contains source code used to train models.
│   │   ├── evaluation.ipynb # This is a Jupyer notebook containing our evaluation of the distinct models we trained.
│   └── models               # This contains the models themselves as well as scaler objects for loading.
├── data
│   ├── outcomes             # This contains the outcomes of one of the three sets, respective to the subdirectory suffix.
│   ├── set-a                # This is the training data.
│   ├── set-b                # This is the test data.
│   ├── set-c                # This is the validation data.
├── requirements.txt         # This contains the dependencies used in this project.
├── resources                # This contains notes and text resources.
```

## Installation
To get started:

1. Install Python 3.12.x.
2. Install Python dependencies
    ```python
    pip install -r requirements
    ```

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

## Authors
- Jean Luis Urena
- Jhanavi Lingamneni
- Sarvesh Kapil Pathak
