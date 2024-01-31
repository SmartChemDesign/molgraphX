import numpy as np
import os.path
import rdkit
import sys
import torch
from rdkit import RDLogger
from rdkit.Chem import rdDepictor, AllChem
from torch import nn
from torch_geometric.data import Data
from typing import Union

sys.path.append(os.path.abspath("../../"))

from Source.rdkit_heatmaps import mapvalues2mol
from Source.rdkit_heatmaps.utils import transform2png

RDLogger.DisableLog('rdApp.*')


class ExplainableModel(nn.Module):
    def __init__(self, model, targets: Union[list, tuple] = None):
        super().__init__()
        self.model = model
        self.targets = targets

    def forward(self, *data, **kwargs):
        if len(data) == 1 and isinstance(data[0], Data):
            data = data[0]
        elif len(data) == 2:
            x, edge_index = data
            data = Data(x=x, edge_index=edge_index)
        elif "data" in kwargs:
            data = kwargs["data"]
        elif "x" in kwargs and "edge_index" in kwargs:
            x, edge_index = kwargs["x"], kwargs["edge_index"]
            data = Data(x=x, edge_index=edge_index)
        else:
            raise ValueError(f"invalid format of data: {data}")
        result = self.model(data)
        if self.targets is None:
            output = list(result.values())
        else:
            output = [result[target] for target in self.targets]
        return torch.cat(output, dim=0)


def get_partial_charges(mol: rdkit.Chem.Mol) -> list[float]:
    """
    Retrieve the partial charges of atoms in a molecule.

    Args:
        mol (rdkit.Chem.Mol): The molecule to retrieve partial charges from.

    Returns:
        list[float]: The list of partial charges for each atom in the molecule.
    """
    AllChem.ComputeGasteigerCharges(mol)
    partial_charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]

    return partial_charges


def visualize(mol, scores: list[float], save_path: str = None, normalize=False, show_values=True):
    scores = np.array(scores)
    rdDepictor.Compute2DCoords(mol)
    if normalize: scores = (scores - scores.mean()) / scores.std()

    if show_values:
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetProp("atomNote", f"{scores[i]:.2f}")

    canvas = mapvalues2mol(mol, scores)
    img = transform2png(canvas.GetDrawingText())
    if save_path is not None: img.save(save_path)
    return img
