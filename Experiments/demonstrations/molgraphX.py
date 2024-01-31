import copy
import torch
from argparse import ArgumentParser
from dgllife.utils import CanonicalAtomFeaturizer
from rdkit import Chem

from Source.explainers.molgraphX.utils import get_scores as get_molgraphX_scores
from Source.explainers.utils import visualize, ExplainableModel
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
molgraphX_kwargs = {
    "mode": "regression",
    "min_atoms": 3,
}

mol = Chem.MolFromSmiles(args.smiles)
graph = FEATURIZER.featurize(mol)

molgraphX_scores = get_molgraphX_scores(mol, featurizer=FEATURIZER, explainable_model=MODEL, device=DEVICE,
                                        explainer_kwargs=molgraphX_kwargs, is_sym=True, target=0)

img = visualize(copy.deepcopy(mol), molgraphX_scores, show_values=False)
img.save(ROOT_DIR / args.output_file)

print("Successfully saved the explanation to", ROOT_DIR / args.output_file)
