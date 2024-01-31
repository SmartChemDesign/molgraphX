import copy
import torch
from argparse import ArgumentParser
from dgllife.utils import CanonicalAtomFeaturizer
from rdkit import Chem

from Source.explainers.subgraphX.utils import draw_best_subgraph
from Source.explainers.submoleculeX.utils import get_subgraphs as get_submoleculeX_subgraphs
from Source.explainers.utils import ExplainableModel
from Source.models.GCNN.featurizers import DGLFeaturizer
from Source.models.GCNN.model import GCNN
from Source.trainer import ModelShell
from config import ROOT_DIR

parser = ArgumentParser()
parser.add_argument('--smiles', type=str, required=True, help='Smiles string to explain the prediction for')
parser.add_argument('--model-folder', type=str, required=True, help='Path to folder with trained model')
parser.add_argument('--output-file', type=str, required=True, help='Path to the output .png file')
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

mol = Chem.MolFromSmiles(args.smiles)
graph = FEATURIZER.featurize(mol)

subgraphs = get_submoleculeX_subgraphs(mol, featurizer=FEATURIZER, explainable_model=MODEL,
                                       device=DEVICE,
                                       subgraphX_kwargs=subgraphX_kwargs, target_ids=(0,))

img = draw_best_subgraph(copy.deepcopy(mol), subgraphs, max_nodes=5, show_value=False)
img.save(ROOT_DIR / args.output_file)

print("Successfully saved the explanation to", ROOT_DIR / args.output_file)
