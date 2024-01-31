# External imports
import torch

# Internal imports
from Source.explainers.molgraphX.base import AtomsExplainer


def get_scores(mol, featurizer, explainable_model, target, explainer_kwargs,
               is_sym=False, device=torch.device("cpu")):
    explainer = AtomsExplainer(
        featurizer.featurize,
        explainable_model,
        target,
        device=device,
        **explainer_kwargs
    )

    scores = explainer(mol, is_sym=is_sym)

    return scores
