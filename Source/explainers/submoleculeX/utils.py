import os.path
import sys
from rdkit import Chem
from rdkit import RDLogger

sys.path.append(os.path.abspath("../../../"))

from Source.explainers.submoleculeX.subgraphx import SubgraphX
from Source.mol_symmetry import find_mol_sym_atoms

RDLogger.DisableLog('rdApp.*')


def get_subgraphs(mol, featurizer, explainable_model, device, subgraphX_kwargs, is_sym=False, target_ids=(0,)):
    subgraphX_kwargs["device"] = device
    graph = featurizer.featurize(mol)

    explainer = SubgraphX(
        explainable_model,
        target_ids=target_ids,
        **subgraphX_kwargs
    )

    equivalence_pairs = []
    if is_sym:
        for eq_class in find_mol_sym_atoms(mol):
            equivalence_pairs += [(id_1, id_2) for id_1 in eq_class for id_2 in eq_class]

    node_relation = lambda x, y: (x, y) in equivalence_pairs

    graph.to(device)
    _, explanation_results, _ = explainer(graph.x,
                                          graph.edge_index,
                                          node_relation=node_relation,
                                          max_nodes=graph.x.shape[0])

    subgraphs = explainer.read_from_MCTSInfo_list(explanation_results[0])

    return subgraphs


def draw_subgraphs(mol, subgraphs):
    img = Chem.Draw.MolsToGridImage(
        [mol for _ in subgraphs],
        highlightAtomLists=[subgraph.coalition for subgraph in subgraphs],
        legends=[str(subgraph.P) for subgraph in subgraphs],
    )
    return img


def draw_best_subgraph(mol, subgraphs, max_nodes, show_value=True):
    subgraphs = [s for s in subgraphs if len(s.coalition) <= max_nodes]
    best_subgraph = subgraphs[0]
    legend = str(best_subgraph.P) if show_value else None
    img = Chem.Draw.MolToImage(mol, highlightAtoms=best_subgraph.coalition, legend=legend)
    return img
