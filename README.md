# Molgraph Explainer

## Setup

To accommodate dependencies that are specific to either GPU or CPU, you must define the target device. To do this,
modify the `DEVICE` variable in the `Makefile`. Proceed with the setup by executing these commands:

```bash
make install_all
make download_dataset
```

This process prepares the virtual environment, installs all necessary dependencies, and populates the `Data` folder with
the required datasets.

## Hyperparameter Optimization

To initiate the search for optimal hyperparameters, use:

```bash
make optimize_hparams
```

To tailor the search space, adjust the settings in the `objective` function within the `Experiments/optimize_hparams.py`
script.

Subsequently, incorporate the optimized hyperparameters into the `Experiments/train.py` script.

## Model Training

Execute the following to commence training:

```bash
make run_training
```

Upon completion, the performance metrics will be accessible in `Output/trained_model/metrics.json`.

## Prediction

For predictions on new data, ensure you have a trained model and a `.csv` file featuring a `smiles` column.

To predict dipole moments for all molecules in the QM9 dataset, execute:

```bash
make predictions
```

## Explanation

Generate explanations for model predictions with the command:

```bash
venv/bin/python3 -m Experiments.demonstrations.subgraphX \
  --smiles "CCC(=O)" \
  --model-folder "Output/trained_model" \
  --output-file "Output/subgraphX_explanation.png"
```

Examples demonstrating the usage of other explainers
are available within the `test_explainers` function in the `Makefile`.

## Computational Time Analysis

To assess the computational efficiency of various methods, follow this two-step procedure:

1. Record the times:

```bash
make comp_time
```

2. Evaluate the findings:

Start the Jupyter Notebook server with `/venv/bin/jupyter notebook`, navigate
to `Experiments/analyse_computational_time.ipynb`, and select the `molgraphx` kernel for analysis.