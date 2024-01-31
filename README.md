# Molgraph Explainer

## Preparing
To prepare the repository, run the following commands:

```bash
make install_all
make download_dataset
```

This will create the virtual environment, install all the dependencies, 
and create the `Data` folder with QM9 dataset

## Search for best hyperparameters

Use the following command to repeat hyperparameters optimization search:

```bash
make optimize_hparams
```

To modify the search space, you can
adjust the parameters in the `objective` function
located in the `Experiments/optimize_hparams.py` file.

Then you need to update hyperparameters in the `Experiments/train.py` file.

## Train the model

To train the model, run:

```bash
make run_training
```

Metrics now can be found at `Output/trained_model/metrics.json`

## Predict

To generate predictions on new data, you just need
a trained model and a `.csv` file with `smiles` column.

To generate predictions for all the molecules in QM9 dataset, run:
```bash
make predictions
```

## Explain

```bash
venv/bin/python3 -m Experiments.demonstrations.subgraphX \
  --smiles "CCC(=O)" \
  --model-folder "Output/trained_model" \
  --output-file "Output/subgraphX_explanation.png"
  
```

To analyse computational time of different methods, 2-step approach is used:

1. Calculate times:

```bash
make comp_time
```
2. Analyse the results:

Run the jupyter `/venv/bin/jupyter notebook`, then 
open `Experiments/analyse_computational_time.ipynb` and 
select the kernel named `molgraphx`.
