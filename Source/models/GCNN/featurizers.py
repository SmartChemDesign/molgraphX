import dgl
import pandas as pd
import random
import torch
import warnings
from dgllife.utils import mol_to_bigraph
from rdkit import Chem
from torch_geometric.utils import from_networkx
from tqdm import tqdm
from typing import Union


class DGLFeaturizer:
    def __init__(self, require_node_features=True, require_edge_features=True, **kwargs):
        self.require_node_features = require_node_features
        self.require_edge_features = require_edge_features
        self.kwargs = kwargs

    def featurize(self, mol):
        dgl_graph = mol_to_bigraph(mol, **self.kwargs)
        networkx_graph = dgl.to_networkx(dgl_graph)
        graph = from_networkx(networkx_graph)
        if 'h' not in dgl_graph.ndata:
            if self.require_node_features:
                warnings.warn(f"can't featurize {Chem.MolToSmiles(mol)}: 'h' not in graph.ndata. Skipping.")
                return None
            else:
                dgl_graph.ndata['h'] = torch.zeros((dgl_graph.num_nodes(), 1))
        if 'e' not in dgl_graph.edata:
            if self.require_edge_features:
                warnings.warn(f"can't featurize {Chem.MolToSmiles(mol)}: 'e' not in graph.edata. Skipping.")
                return None
            else:
                dgl_graph.edata['e'] = torch.zeros((dgl_graph.num_edges(), 1))
        graph.x = dgl_graph.ndata['h']
        graph.edge_attr = dgl_graph.edata['e']
        graph.id = None
        return graph


def featurize_molecules(molecules: list, mol_featurizer, seed=42):
    all_data = []
    for mol in tqdm(molecules, desc="Featurizing", leave=False):
        graph = mol_featurizer.featurize(mol)
        if graph is None: continue
        prop_names = [prop for prop in mol.GetPropNames()]
        graph.y = {prop_name: torch.tensor([[float(mol.GetProp(prop_name))]]) for prop_name in prop_names}
        all_data += [graph]
    random.Random(seed).shuffle(all_data)

    return all_data


def featurize_csv(path_to_csv: str, mol_featurizer, targets: Union[list, tuple], max_samples=None, seed=42):
    """
    Extract molecules from .csv file and featurize them

    Parameters
    ----------
    path_to_csv : str
        path to .csv file with data
        it should have 'smiles' column and one or more columns with target properties
    mol_featurizer : featurizer, optional
        instance of the class used for extracting features of organic molecule

    Returns
    -------
    features : list of torch_geometric.data objects
        list of graphs corresponding to individual molecules from .csv file
    """
    df = pd.read_csv(path_to_csv).reset_index(drop=True)
    molecules = []
    for i in tqdm(df.index, desc="Prepare molecules", leave=False):
        mol = Chem.MolFromSmiles(df["smiles"][i])
        for target in targets:
            mol.SetProp(target, str(df[target][i]))
        molecules += [mol]
        if max_samples is not None and len(molecules) >= max_samples: break
    return featurize_molecules(molecules, mol_featurizer, seed=seed)


def featurize_sdf(path_to_sdf: str, mol_featurizer, targets: Union[list, tuple], max_samples=None, seed=42):
    """
    Extract molecules from .sdf file and featurize them

    Parameters
    ----------
    path_to_sdf : str
        path to .sdf file with data
        single molecule in .sdf file can contain properties like "logK_{metal}"
        each of these properties will be transformed into a different training sample
    mol_featurizer : featurizer, optional
        instance of the class used for extracting features of organic molecule

    Returns
    -------
    features : list of torch_geometric.data objects
        list of graphs corresponding to individual molecules from .sdf file
    """
    molecules = []
    for mol in tqdm(Chem.SDMolSupplier(path_to_sdf), desc="Prepare molecules", leave=False):
        if mol is None: continue
        for prop in mol.GetPropNames():
            if prop not in targets:
                mol.ClearProp(prop)
        molecules += [mol]
        if max_samples is not None and len(molecules) >= max_samples: break
    return featurize_molecules(molecules, mol_featurizer, seed=seed)
