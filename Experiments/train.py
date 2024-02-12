import logging
import numpy as np
import torch
from argparse import ArgumentParser
from datetime import datetime
from dgllife.utils import CanonicalBondFeaturizer, CanonicalAtomFeaturizer
from sklearn.metrics import r2_score, mean_absolute_error
from torch.nn import MSELoss, LeakyReLU, ReLU
from torch.optim import Adam
from torch_geometric.nn import global_mean_pool, MFConv

from Source.data import root_mean_squared_error, train_test_valid_split
from Source.models.GCNN.featurizers import featurize_csv, DGLFeaturizer
from Source.models.GCNN.model import GCNN
from Source.models.GCNN.trainer import GCNNTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
time_mark = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]

parser = ArgumentParser()
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
args = parser.parse_args()

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
model_parameters = {
    'pre_fc_params': {
        'hidden': [32, 64],
        'dropout': 0,
        'use_bn': True,
        'actf': ReLU(),
    },
    'post_fc_params': {
        'hidden': [256, 128],
        'dropout': 0,
        'use_bn': True,
        'actf': ReLU(),
    },
    'hidden_conv': [64, 256, 512],
    'conv_dropout': 0.2477382148987496,
    'conv_actf': LeakyReLU(),
    'conv_layer': MFConv,
    'conv_parameters': {}, 'graph_pooling': global_mean_pool,
}

featurizer = DGLFeaturizer(add_self_loop=False,
                           node_featurizer=CanonicalAtomFeaturizer(),
                           edge_featurizer=CanonicalBondFeaturizer())
full_dataset = featurize_csv(path_to_csv=args.data, mol_featurizer=featurizer, targets=("mu",),
                             seed=args.seed, max_samples=args.max_samples)

mean_target = np.mean([graph.y[args.target_name].item() for graph in full_dataset])
for graph in full_dataset: graph.y[args.target_name] -= torch.tensor([[mean_target]])

folds, test_loader = train_test_valid_split(full_dataset,
                                            n_folds=args.folds, test_ratio=args.test_ratio,
                                            batch_size=args.batch_size, seed=args.seed)

model = GCNN(
    node_features=next(iter(test_loader)).x.shape[-1],
    targets=targets,
    **model_parameters,
    optimizer=Adam,
    optimizer_parameters={'lr': args.learning_rate},
)

trainer = GCNNTrainer(
    model=model,
    train_valid_data=folds,
    test_data=test_loader,
    output_folder=args.output_folder,
    epochs=args.epochs,
    es_patience=args.es_patience,
    targets=targets,
    seed=args.seed,
)

trainer.train_cv_models()
