import os.path
import sys
from rdkit import Chem
from rdkit import RDLogger

from Source.explainers.subgraphX.subgraphx import SubgraphX

sys.path.append(os.path.abspath("../../../"))
RDLogger.DisableLog('rdApp.*')


def get_subgraphs(mol, featurizer, explainable_model, device, subgraphX_kwargs, target_ids=(0,)):
    subgraphX_kwargs["device"] = device
    graph = featurizer.featurize(mol)

    explainer = SubgraphX(
        explainable_model,
        target_ids=target_ids,
        **subgraphX_kwargs
    )

    graph.to(device)
    _, explanation_results, _ = explainer(graph.x,
                                          graph.edge_index,
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
