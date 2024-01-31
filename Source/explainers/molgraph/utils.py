import torch
from Source.explainers.molgraph.molgraph_explainers import AtomsExplainer

from Source.mol_symmetry import find_mol_sym_atoms


def get_scores(mol, featurizer, explainable_model, target, explainer_kwargs, is_sym=False, device=torch.device("cpu")):
    graph = featurizer.featurize(mol).to(device)
    explainer = AtomsExplainer(
        explainable_model,
        target,
        device=device,
        **explainer_kwargs
    )

    if is_sym:
        sym_atoms = find_mol_sym_atoms(mol)
        sym_nodes_relation = {(x, y) for cls in sym_atoms for x in cls for y in cls}
        atom_relation = lambda x, y: (x, y) in sym_nodes_relation
    else:
        atom_relation = lambda x, y: x == y

    scores = explainer(graph, atom_relation=atom_relation)

    return scores
