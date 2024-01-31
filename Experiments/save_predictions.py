import pandas as pd
import torch
from argparse import ArgumentParser
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from rdkit import Chem
from torch import nn
from torch_geometric.data import Data
from tqdm import tqdm
from typing import Union

from Source.explainers.utils import ExplainableModel
from Source.models.GCNN.featurizers import DGLFeaturizer
from Source.models.GCNN.model import GCNN
from Source.trainer import ModelShell
from config import ROOT_DIR

# As far as we trained model to predict mu minus the mean of mu,
# we need to shift the predictions by mean of mu
SHIFT = 2.706072137103447


parser = ArgumentParser()
parser.add_argument('--model-folder', type=str, required=True, help='Path to folder with trained model')
parser.add_argument('--data', type=str, required=True, help='Path to the data .csv file')
parser.add_argument('--max-samples', type=int, default=None, help='Make predictions for the first N samples')
parser.add_argument('--output-file', type=str, required=True, help='Path to the output .csv file')
args = parser.parse_args()

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
MODEL = ExplainableModel(
    ModelShell(
        GCNN,
        ROOT_DIR / args.model_folder,
        device=DEVICE
    ))
FEATURIZER = DGLFeaturizer(add_self_loop=False,
                           node_featurizer=CanonicalAtomFeaturizer(),
                           edge_featurizer=CanonicalBondFeaturizer(),
                           canonical_atom_order=False)


def get_prediction(smiles):
    mol = Chem.MolFromSmiles(smiles)
    graph = FEATURIZER.featurize(mol)
    if graph is None: return None
    return MODEL(graph.to(DEVICE)).item() + SHIFT


df = pd.read_csv(ROOT_DIR / args.data, nrows=args.max_samples)
output_filename = ROOT_DIR / args.output_file

df["prediction"] = [get_prediction(smiles) for smiles in tqdm(df["smiles"].tolist())]
df[["smiles", "prediction"]].to_csv(output_filename, index=False)
