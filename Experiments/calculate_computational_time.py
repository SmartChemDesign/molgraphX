import pandas as pd
import torch
from argparse import ArgumentParser
from datetime import datetime
from dgllife.utils import CanonicalAtomFeaturizer
from rdkit import Chem
from tqdm import tqdm

from Source.explainers.molgraph.utils import get_scores as get_molgraph_scores
from Source.explainers.molgraphX.utils import get_scores as get_molgraphX_scores
from Source.explainers.subgraphX.utils import get_subgraphs as get_subgraphX_subgraphs
from Source.explainers.submoleculeX.utils import get_subgraphs as get_submoleculeX_subgraphs
from Source.explainers.utils import ExplainableModel
from Source.models.GCNN.featurizers import DGLFeaturizer
from Source.models.GCNN.model import GCNN
from Source.trainer import ModelShell
from config import ROOT_DIR

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
FEATURIZER = DGLFeaturizer(require_edge_features=False,
                           add_self_loop=False,
                           node_featurizer=CanonicalAtomFeaturizer(),
                           canonical_atom_order=False)

subgraphX_kwargs = {
    "mode": "regression",
    "device": DEVICE,
    "explain_graph": True,  # verbose: True,
    "rollout": 20,  # Number of iteration to get the prediction (MCTS hyperparameter)
    "min_atoms": 1,
    "c_puct": 10.0,  # The hyperparameter which encourages the exploration (MCTS hyperparameter)
    "sample_num": None,
    # Sampling time of monte carlo sampling approximation for 'mc_shapley', 'mc_l_shapley' reward_methods
    "reward_method": "l_shapley",  # one of ["gnn_score", "mc_shapley", "l_shapley", "mc_l_shapley", "nc_mc_l_shapley"]
    "subgraph_building_method": "zero_filling",  # one of ["zero_filling", "split"]
}

submoleculeX_kwargs = subgraphX_kwargs

molgraph_kwargs = {
    "mode": "regression",
    "local_radius": 3,
    "sample_num": 100,
    # Sampling time of monte carlo sampling approximation for 'mc_shapley', 'mc_l_shapley' reward_methods
    "reward_method": "l_shapley",  # one of ["gnn_score", "mc_shapley", "l_shapley", "mc_l_shapley", "nc_mc_l_shapley"]
}

molgraphX_kwargs = {
    "mode": "regression",
    "min_atoms": 3,
}

df = pd.read_csv(ROOT_DIR / args.data, nrows=args.max_samples)

smiles_list = []
mu_list = []
ids = list(range(len(df)))

for i in tqdm(ids):
    smiles = df["smiles"][i]
    mol = Chem.MolFromSmiles(smiles)
    graph = FEATURIZER.featurize(mol).to(DEVICE)
    if graph is None: continue

    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    start_time = datetime.now()
    get_subgraphX_subgraphs(mol, featurizer=FEATURIZER, explainable_model=MODEL, device=DEVICE,
                            subgraphX_kwargs=subgraphX_kwargs, target_ids=(0,))
    df.loc[i, "subgraphX"] = (datetime.now() - start_time).total_seconds()

    start_time = datetime.now()
    get_submoleculeX_subgraphs(mol, featurizer=FEATURIZER, explainable_model=MODEL, device=DEVICE,
                               subgraphX_kwargs=submoleculeX_kwargs, target_ids=(0,))
    df.loc[i, "submoleculeX"] = (datetime.now() - start_time).total_seconds()

    start_time = datetime.now()
    get_molgraph_scores(mol, featurizer=FEATURIZER, explainable_model=MODEL, device=DEVICE,
                        explainer_kwargs=molgraph_kwargs, is_sym=True, target=0)
    df.loc[i, "molgraph"] = (datetime.now() - start_time).total_seconds()

    try:
        start_time = datetime.now()
        get_molgraphX_scores(mol, featurizer=FEATURIZER, explainable_model=MODEL, device=DEVICE,
                             explainer_kwargs=molgraphX_kwargs, is_sym=True, target=0)
        df.loc[i, "molgraphX"] = (datetime.now() - start_time).total_seconds()
    except Exception as e:
        print(e)
        df.loc[i, "molgraphX"] = None

    smiles_list += [smiles]

df.to_csv(ROOT_DIR / args.output_file, index=False)
