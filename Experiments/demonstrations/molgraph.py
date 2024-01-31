import copy
import torch
from argparse import ArgumentParser
from dgllife.utils import CanonicalAtomFeaturizer
from rdkit import Chem

from Source.explainers.molgraph.utils import get_scores as get_molgraph_scores
from Source.explainers.utils import ExplainableModel, visualize
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

molgraph_kwargs = {
    "mode": "regression",
    "local_radius": 3,
    "sample_num": 100,
    # Sampling time of monte carlo sampling approximation for 'mc_shapley', 'mc_l_shapley' reward_methods
    "reward_method": "l_shapley",  # one of ["gnn_score", "mc_shapley", "l_shapley", "mc_l_shapley", "nc_mc_l_shapley"]
}

mol = Chem.MolFromSmiles(args.smiles)
graph = FEATURIZER.featurize(mol)

molgraph_scores = get_molgraph_scores(mol, featurizer=FEATURIZER, explainable_model=MODEL, device=DEVICE,
                                      explainer_kwargs=molgraph_kwargs, is_sym=True, target=0)

img = visualize(copy.deepcopy(mol), molgraph_scores, show_values=False)
img.save(ROOT_DIR / args.output_file)

print("Successfully saved the explanation to", ROOT_DIR / args.output_file)
