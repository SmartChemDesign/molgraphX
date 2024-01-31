import json
import optuna
import os
import torch
import torch.nn as nn
from argparse import ArgumentParser
from datetime import datetime
from dgllife.utils import CanonicalAtomFeaturizer, AttentiveFPAtomFeaturizer, PAGTNAtomFeaturizer
from inspect import isfunction, isclass, signature
from sklearn.metrics import r2_score, mean_absolute_error
from torch.nn import MSELoss
from torch_geometric.nn import MFConv, GATConv, TransformerConv, SAGEConv, global_mean_pool, global_max_pool

from Source.data import root_mean_squared_error, train_test_valid_split
from Source.models.FCNN.optimize_hparams import GeneralParams as FCNNGeneralParams, FCNNParams
from Source.models.GCNN.featurizers import DGLFeaturizer, featurize_csv
from Source.models.GCNN.model import GCNN
from Source.models.GCNN.trainer import GCNNTrainer

ACTIVATION_VARIANTS = {
    "LeakyReLU": nn.LeakyReLU(),
    "PReLU": nn.PReLU(),
    "Tanhshrink": nn.Tanhshrink(),
}

CONVOLUTION_VARIANTS = {
    "MFConv": MFConv,
    "GATConv": GATConv,
    "TransformerConv": TransformerConv,
    "SAGEConv": SAGEConv,
}

POOLING_VARIANTS = {
    "global_mean_pool": global_mean_pool,
    "global_max_pool": global_max_pool,
}

parser = ArgumentParser()
parser.add_argument('--n-trials', type=int, default=100, help='Number of optuna trials')
parser.add_argument('--timeout', type=int, default=None, help='Time limit (in seconds) for optuna optimization')
parser.add_argument('--data', type=str, required=True, help='Path to the data file')
parser.add_argument('--target-name', type=str, required=True, help='Name of column with targets')
parser.add_argument('--output-folder', type=str, required=True, help='Output folder')
parser.add_argument('--max-samples', type=int, default=None, help='Use only a several samples for training')
parser.add_argument('--test-ratio', type=float, default=0.2, help='Ratio of data to be used for testing')
parser.add_argument('--folds', type=int, default=1, help='Number of folds for cross-validation')
parser.add_argument('--seed', type=int, default=0, help='Seed for random number generators')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
parser.add_argument('--epochs', type=int, default=1000, help='Maximum number of epochs to train')
parser.add_argument('--es-patience', type=int, default=100, help='Number of epochs to wait before early stopping')
parser.add_argument('--mode', type=str, default="regression",
                    help='Mode of the target - regression, binary_classification or multiclass_classification')
parser.add_argument('--learning-rate', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--save-models', action='store_true',
                    help='If set, models from each trial will be saved to output folder')

args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)
targets = ({
               'name': args.target_name,
               'mode': args.mode,
               'dim': 1,
               'metrics': {
                   'R2': (r2_score, {}),
                   'RMSE': (root_mean_squared_error, {}),
                   'MAE': (mean_absolute_error, {})
               },
               'loss': MSELoss()
           },)


class GeneralParams(FCNNGeneralParams):
    def __init__(self, trial: optuna.Trial, optimizer_variants=None, lr_lims=(1e-5, 1e-1),
                 featurizer_variants=None):
        super(GeneralParams, self).__init__(trial, optimizer_variants, lr_lims, featurizer_variants)

    def get_featurizer(self):
        featurizer_name = self.trial.suggest_categorical("featurizer_name", list(self.featurizer_variants.keys()))
        featurizer = self.featurizer_variants[featurizer_name]

        add_self_loop = self.trial.suggest_categorical("add_self_loop", (True, False))
        featurizer = DGLFeaturizer(add_self_loop=add_self_loop, node_featurizer=featurizer,
                                   require_edge_features=False)

        return featurizer


class GCNNParams:
    def __init__(self, trial: optuna.Trial,
                 pre_fc_params=None, post_fc_params=None,
                 n_conv_lims=(2, 10), dropout_lims=(0, 0.8),
                 actf_variants=None, dim_lims=(32, 128, 32),
                 conv_layer_variants=None, pooling_layer_variants=None, prefix=""):
        self.trial = trial
        self.prefix = prefix
        self.pre_fc_params = pre_fc_params or {
            "dim_lims": (1, 5),
            "n_layers_lims": (1, 5),
            "actf_variants": None,
            "dropout_lims": (0, 0.8),
            "bn_variants": (True, False)
        }
        self.post_fc_params = post_fc_params or {
            "dim_lims": (1, 5),
            "n_layers_lims": (1, 5),
            "actf_variants": None,
            "dropout_lims": (0, 0.8),
            "bn_variants": (True, False)
        }
        self.conv_layer_variants = conv_layer_variants or {
            "MFConv": MFConv,
        }
        self.pooling_layer_variants = pooling_layer_variants or {
            "global_mean_pool": global_mean_pool,
        }
        self.n_layers_lims = n_conv_lims
        self.dim_lims = dim_lims
        self.dropout_lims = dropout_lims
        self.actf_variants = actf_variants or {
            "nn.ReLU()": nn.ReLU(),
            "nn.LeakyReLU()": nn.LeakyReLU(),
        }

    def get_conv_layer(self):
        self.conv_layer_name = self.trial.suggest_categorical(
            f"{self.prefix}_conv_layer_name",
            list(self.conv_layer_variants.keys())
        )
        return self.conv_layer_variants[self.conv_layer_name]

    def get_pooling_layer(self):
        self.pooling_layer_name = self.trial.suggest_categorical(
            f"{self.prefix}_pooling_layer_name",
            list(self.pooling_layer_variants.keys())
        )
        pooling_layer = self.pooling_layer_variants[self.pooling_layer_name]
        if isfunction(pooling_layer):
            return pooling_layer
        if isclass(pooling_layer) and "in_channels" in signature(pooling_layer).parameters:
            params = {"in_channels": self.dim_lims[-1]}
            return pooling_layer(**params)

    def get_actf(self):
        self.actf_name = self.trial.suggest_categorical(
            f"{self.prefix}_actf_name",
            list(self.actf_variants.keys())
        )
        return self.actf_variants[self.actf_name]

    def get_conv_parameters(self):
        conv_parameters = {}
        if self.conv_layer.__name__ == "SSGConv":
            conv_parameters = {
                "alpha": 0.5,
            }
        return conv_parameters

    def get(self):
        self.n_layers = self.trial.suggest_int(f"{self.prefix}_n_layers", *self.n_layers_lims)
        self.conv_layer = self.get_conv_layer()
        self.conv_parameters = self.get_conv_parameters()
        self.dropout = self.trial.suggest_float(f"{self.prefix}_dropout", *self.dropout_lims)
        self.activation = self.get_actf()
        self.hidden = [self.trial.suggest_int(f"{self.prefix}_hidden_{i}", *self.dim_lims, log=True)
                       for i in range(self.n_layers)]
        self.pooling_layer = self.get_pooling_layer()

        model_parameters = {
            "pre_fc_params": FCNNParams(self.trial, **self.pre_fc_params, prefix="pre_fc").get(),
            "post_fc_params": FCNNParams(self.trial, **self.post_fc_params, prefix="post_fc").get(),
            "hidden_conv": self.hidden,
            "conv_dropout": self.dropout,
            "conv_actf": self.activation,
            "conv_layer": self.conv_layer,
            "conv_parameters": self.conv_parameters,
            "graph_pooling": global_mean_pool
        }

        return model_parameters


def objective(trial: optuna.Trial):
    model_parameters = GCNNParams(
        trial,
        pre_fc_params={
            "dim_lims": (32, 512),
            "n_layers_lims": (1, 4),
            "actf_variants": ACTIVATION_VARIANTS,
            "dropout_lims": (0, 0),
            "bn_variants": (True,)
        },
        post_fc_params={
            "dim_lims": (32, 512),
            "n_layers_lims": (1, 4),
            "actf_variants": ACTIVATION_VARIANTS,
            "dropout_lims": (0, 0),
            "bn_variants": (True,)
        },
        n_conv_lims=(1, 3), dropout_lims=(0, 1),
        actf_variants=ACTIVATION_VARIANTS, dim_lims=(32, 512),
        conv_layer_variants=CONVOLUTION_VARIANTS,
        pooling_layer_variants=POOLING_VARIANTS,
    ).get()

    general_parameters = GeneralParams(
        trial,
        optimizer_variants={
            "SGD": torch.optim.SGD,
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            "RMSprop": torch.optim.RMSprop,
            "Adagrad": torch.optim.Adagrad,
            "Adadelta": torch.optim.Adadelta,
            "Adamax": torch.optim.Adamax,
        },
        lr_lims=(5e-5, 5e-4),
        featurizer_variants={
            "CanonicalAtomFeaturizer": CanonicalAtomFeaturizer(),
            "AttentiveFPAtomFeaturizer": AttentiveFPAtomFeaturizer(),
            "PAGTNAtomFeaturizer": PAGTNAtomFeaturizer(),
        },
    ).get()

    full_dataset = featurize_csv(path_to_csv=args.data, mol_featurizer=general_parameters["featurizer"],
                                 targets=("mu",), seed=args.seed, max_samples=args.max_samples)
    folds, test_data = train_test_valid_split(full_dataset, n_folds=args.folds, test_ratio=args.test_ratio,
                                              batch_size=64, seed=args.seed)

    model_parameters["node_features"] = test_data.dataset[0].x.shape[-1]
    model_parameters["targets"] = targets
    model_parameters["optimizer"] = general_parameters["optimizer"]
    model_parameters["optimizer_parameters"] = general_parameters["optimizer_parameters"]

    model = GCNN(**model_parameters)

    time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]
    output_folder = os.path.join(args.output_folder, f"{trial.number}-trial_GCNN_{args.mode}_{time_mark}")

    trainer = GCNNTrainer(
        model=model,
        train_valid_data=folds,
        test_data=test_data,
        output_folder=output_folder,
        epochs=args.epochs,
        es_patience=args.es_patience,
        targets=targets,
        seed=args.seed,
        save_to_folder=args.save_models,
    )

    trainer.train_cv_models()

    trial.set_user_attr(key="metrics", value=trainer.results_dict["general"])
    trial.set_user_attr(key="model_parameters", value={key: str(model_parameters[key]) for key in model_parameters})

    errors = [trainer.results_dict["general"][key] for key in trainer.results_dict["general"] if "MSE" in key]
    return max(errors)


def callback(study: optuna.Study, trial):
    study.trials_dataframe().to_csv(path_or_buf=os.path.join(args.output_folder, f"trials.csv"), index=False)


if __name__ == "__main__":
    start = datetime.now()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, callbacks=[callback])
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

    study.trials_dataframe().to_csv(path_or_buf=os.path.join(args.output_folder, f"trials.csv"), index=False)

    with open(os.path.join(args.output_folder, f"result.json"), "w") as jf:
        json.dump(result, jf)
