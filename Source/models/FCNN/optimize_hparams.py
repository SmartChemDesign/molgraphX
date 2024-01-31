import json
import optuna
import os
import sys
import torch.nn as nn
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error
from torch.optim import Adam
from torch_geometric.nn import MFConv, global_mean_pool

from Source.data import root_mean_squared_error, train_test_valid_split
from Source.models.FCNN.featurizers import featurize_sdf, ECFPMolFeaturizer
from Source.models.FCNN.model import FCNN
from Source.models.FCNN.trainer import FCNNTrainer
from config import ROOT_DIR

sys.path.append(os.path.abspath("."))

time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

n_trials = None
timeout = 60 * 60 * 3

cv_folds = 1
batch_size = 64
epochs = 10
es_patience = 1

mode = "regression"
general_output_path = f"Output/Optuna3GCNN_{time_mark}"
output_mark = "Test"
path_to_sdf = str(ROOT_DIR / "Data/OneM/Cu_2.sdf")
max_data = None
targets = ({
               "name": "logK",
               "mode": "regression",
               "dim": 1,
               "metrics": {
                   "R2": (r2_score, {}),
                   "RMSE": (root_mean_squared_error, {}),
                   "MAE": (mean_absolute_error, {})
               },
               "loss": nn.MSELoss(),
           },)
model_parameters = {
    "pre_fc_params": {
        "hidden": (),
        "dropout": 0,
        "actf": nn.LeakyReLU(),
    },
    "post_fc_params": {
        "hidden": (256,),
        "dropout": 0.06698879155641034,
        "use_bn": False,
        "actf": nn.LeakyReLU(),
    },
    "hidden_conv": (128, 128, 64,),
    "conv_dropout": 0.27936243337975536,
    "conv_actf": nn.LeakyReLU(),
    "conv_layer": MFConv,
    "conv_parameters": None,
    "graph_pooling": global_mean_pool
}


class GeneralParams:
    def __init__(self, trial: optuna.Trial, optimizer_variants=None, lr_lims=(1e-5, 1e-1),
                 featurizer_variants=None):
        self.trial = trial
        self.featurizer_variants = featurizer_variants or {
            "ECFPMolFeaturizer": ECFPMolFeaturizer(),
        }
        self.optimizer_variants = optimizer_variants or {"Adam": Adam}
        self.lr_lims = lr_lims

    def get_optimizer(self):
        self.optimizer_name = self.trial.suggest_categorical(
            "optimizer_name",
            list(self.optimizer_variants.keys())
        )
        return self.optimizer_variants[self.optimizer_name]

    def get_optimizer_parameters(self):
        optimizer_parameters = {
            "lr": self.trial.suggest_float("lr", *self.lr_lims),
        }
        return optimizer_parameters

    def get_featurizer(self):
        featurizer_name = self.trial.suggest_categorical("featurizer_name", list(self.featurizer_variants.keys()))
        featurizer = self.featurizer_variants[featurizer_name]
        return featurizer

    def get(self):
        return {
            "optimizer": self.get_optimizer(),
            "optimizer_parameters": self.get_optimizer_parameters(),
            "featurizer": self.get_featurizer(),
        }


class FCNNParams:
    def __init__(self, trial: optuna.Trial,
                 dim_lims=(1, 5), n_layers_lims=(1, 5), actf_variants=None,
                 dropout_lims=(0, 0.8), bn_variants=(True, False), prefix=""):
        self.trial = trial
        self.prefix = prefix
        self.dim_lims = dim_lims
        self.n_layers_lims = n_layers_lims
        self.dropout_lims = dropout_lims
        self.bn_variants = bn_variants
        self.actf_variants = actf_variants or {
            "nn.ReLU()": nn.ReLU(),
            "nn.LeakyReLU()": nn.LeakyReLU(),
        }

    def get_activation(self):
        self.actf_name = self.trial.suggest_categorical(
            f"{self.prefix}_actf_name",
            list(self.actf_variants.keys())
        )
        return self.actf_variants[self.actf_name]

    def get(self):
        self.n_layers = self.trial.suggest_int(f"{self.prefix}_n_layers", *self.n_layers_lims)
        self.dropout = self.trial.suggest_float(f"{self.prefix}_dropout", *self.dropout_lims)
        self.use_batch_norm = self.trial.suggest_categorical(f"{self.prefix}_use_batch_norm", self.bn_variants)
        self.activation = self.get_activation()
        self.hidden = [
            self.trial.suggest_int(f"{self.prefix}_hidden_{i}", *self.dim_lims, log=True)
            for i in range(self.n_layers)
        ]

        model_parameters = {
            "hidden": self.hidden,
            "dropout": self.dropout,
            "use_bn": self.use_batch_norm,
            "actf": self.activation,
        }

        return model_parameters


def objective(trial: optuna.Trial):
    model_parameters = FCNNParams(
        trial,
        dim_lims=(1, 5),
        n_layers_lims=(1, 5),
        actf_variants=None,
        dropout_lims=(0, 0.8),
        bn_variants=(True, False),
    ).get()

    general_parameters = GeneralParams(
        trial,
        optimizer_variants={"Adam": Adam},
        lr_lims=(1e-5, 1e-1),
        featurizer_variants={
            "ECFPMolFeaturizer": ECFPMolFeaturizer(),
        },
    ).get()

    data = featurize_sdf(path_to_sdf=path_to_sdf, featurizer=general_parameters["featurizer"], seed=42)
    folds, test_data = train_test_valid_split(data, n_folds=cv_folds, test_ratio=0.1, batch_size=batch_size)

    model_parameters["input_features"] = data[0][0].shape[-1]
    model_parameters["targets"] = targets
    model_parameters["use_out_sequential"] = True
    model_parameters["optimizer"] = general_parameters["optimizer"]
    model_parameters["optimizer_parameters"] = general_parameters["optimizer_parameters"]

    model = FCNN(**model_parameters)

    time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]
    output_folder = os.path.join(general_output_path, f"{output_mark}_{mode}_{time_mark}")

    trainer = FCNNTrainer(
        model=model,
        train_valid_data=folds,
        test_data=test_data,
        output_folder=output_folder,
        epochs=epochs,
        es_patience=es_patience,
        targets=targets,
    )

    trainer.train_cv_models()

    trial.set_user_attr(key="metrics", value=trainer.results_dict["general"])
    trial.set_user_attr(key="model_parameters", value={key: str(model_parameters[key]) for key in model_parameters})

    errors = [trainer.results_dict["general"][key] for key in trainer.results_dict["general"] if "MSE" in key]
    return max(errors)


def callback(study: optuna.Study, trial):
    study.trials_dataframe().to_csv(path_or_buf=os.path.join(general_output_path, f"trials.csv"), index=False)


if __name__ == "__main__":
    start = datetime.now()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[callback])
    end = datetime.now()

    result = {
        "trials": len(study.trials),
        "started": str(start).split(".")[0],
        "finished": str(end).split(".")[0],
        "duration": str(end - start).split(".")[0],
        **study.best_trial.user_attrs["metrics"],
        "model_parameters": study.best_trial.user_attrs["model_parameters"],
    }

    time_mark = str(start).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

    study.trials_dataframe().to_csv(path_or_buf=os.path.join(general_output_path, f"trials.csv"), index=False)

    with open(os.path.join(general_output_path, f"result.json"), "w") as jf:
        json.dump(result, jf)
