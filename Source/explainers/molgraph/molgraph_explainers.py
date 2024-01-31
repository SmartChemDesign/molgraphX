import networkx as nx
import torch
from itertools import combinations
from random import randrange
from scipy.special import comb
from torch.nn.functional import softmax
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from typing import List, Tuple, FrozenSet, Callable


def GnnNetsGR2valueFunc(gnnNets, target):
    def value_func(batch):
        with torch.no_grad():
            result = gnnNets(batch)
            score = result[:, target]
        return score

    return value_func


def GnnNetsGC2valueFunc(gnnNets, target):
    def value_func(batch):
        with torch.no_grad():
            logits = gnnNets(batch)
            probs = softmax(logits, dim=-1)
            score = probs[:, target]
        return score

    return value_func


class AtomsDataset(Dataset):
    def __init__(self, data, include_mask, exclude_mask):
        super().__init__()
        self.num_nodes = data.num_nodes
        self.X = data.x
        self.edge_index = data.edge_index
        self.device = self.X.device
        self.label = data.y
        self.include_mask = include_mask
        self.exclude_mask = exclude_mask

    def len(self):
        return self.include_mask.shape[0]

    def get(self, idx):
        include_x = self.X * self.include_mask[idx].unsqueeze(1)
        include_data = Data(x=include_x, edge_index=self.edge_index)
        exclude_x = self.X * self.exclude_mask[idx].unsqueeze(2)
        exclude_data = [
            Data(x=x, edge_index=self.edge_index) for x in exclude_x
        ]
        return include_data, exclude_data


def marginal_score(data: Data,
                   include_mask: torch.Tensor,
                   exclude_mask: torch.Tensor,
                   value_func):
    """
    Calculate the marginal score for each pair of nodes mask. 
    Here include_mask and exclude_mask are node mask.
    """

    dataset = AtomsDataset(data, include_mask, exclude_mask)
    dataloader = DataLoader(dataset,
                            batch_size=256,
                            shuffle=False,
                            num_workers=0)

    marginal_scores = []

    for include_data, exclude_data in dataloader:
        exclude_values = torch.stack([value_func(batch) for batch in exclude_data], dim=1)
        include_values = value_func(include_data).unsqueeze(1).expand_as(exclude_values)
        marginal_scores.append(include_values - exclude_values)

    marginal_score = torch.cat(marginal_scores, dim=0)

    return marginal_score


def coalition_atoms(coalition: List[FrozenSet[int]]) -> List[int]:
    return [atom for cls in coalition for atom in cls]


def coalition_id(coalition: List[FrozenSet[int]]) -> FrozenSet[int]:
    return frozenset().union(*coalition)


def factor_graph_coalitions(factor_graph: nx.Graph):
    coalitions = [list(factor_graph.nodes())]
    coalitions_keys = set({})
    while len(coalitions) > 0:
        coalition = coalitions.pop(0)
        yield coalition
        for skip_coalition in combinations(coalition, len(coalition) - 1):
            subgraph = factor_graph.subgraph(skip_coalition)
            for child in nx.connected_components(subgraph):
                child_coalition = list(child)
                key = hash(coalition_id(child_coalition))
                if key not in coalitions_keys:
                    coalitions.append(child_coalition)
                    coalitions_keys.add(key)


MODEL_MODES = ["classification", "regression"]
REWARD_METHOD = ['l_shapley', 'mc_shapley', 'mc_l_shapley']


class AtomsExplainer(object):
    """Explain atoms of molecular graphs"""

    def __init__(self,
                 model: torch.nn.Module,
                 target: int,
                 device: torch.device = torch.device("cpu"),
                 mode: str = "classification",
                 reward_method: str = 'mc_l_shapley',
                 local_radius: int = 4,
                 sample_num: int = 100):
        """
        Initialize atoms explainer

        Parameters
        ----------
        model : torch.nn.Module
            The target model prepared to explain
        target : int
            Target index in the model output, which should be explained
        device : torch.device
            Device on which a torch.Tensor is or will be allocated.
            Default: cpu
        mode : str
            Type of GCNN: either classification or regression. 
            Default: classification
        reward_method : str
            The name of method for calculating atoms reward. Available reward 
            method's names are l_shapley, mc_shapley, mc_l_shapley. 
            Defailt: mc_l_shapley
        local_radius : int
            Radius of the neighborhood for calculating local atoms reward. 
            Default: 4
        sample_num : int
            Monte Carlo sampling number for approximation of atoms reward.
            Default: 100
        """

        assert mode in MODEL_MODES, \
            f"Invalid mode: {mode}. Valid values are {', '.join(MODEL_MODES)}"
        assert reward_method in REWARD_METHOD, \
            f"Invalid reward method: {reward_method}. " \
            "Valid values are {', '.join(REWARD_METHOD)}"
        assert local_radius > 0, \
            f"Invalid local radius: {local_radius}."
        assert sample_num > 0, \
            f"Invalid sample numer: {sample_num}."

        model.eval()
        model.to(device)
        self.device = device

        if mode == "classification":
            self.value_func = GnnNetsGC2valueFunc(model, target)
        elif mode == "regression":
            self.value_func = GnnNetsGR2valueFunc(model, target)

        if reward_method == 'mc_shapley':
            self.neighborhoods_method = self.all_neighborhoods
            self.sample_method = self.sample_subsets
            self.shapley_method = self.mean_score
        elif reward_method == 'l_shapley':
            self.neighborhoods_method = self.local_neighborhoods
            self.sample_method = self.all_subsets
            self.shapley_method = self.shapley_score
        elif reward_method == 'mc_l_shapley':
            self.neighborhoods_method = self.local_neighborhoods
            self.sample_method = self.all_subsets
            self.shapley_method = self.mean_score

        self.local_radius = local_radius
        self.sample_num = sample_num

    def __call__(self,
                 data: Data,
                 atom_relation: Callable[[int, int], bool] = None) \
            -> list[float]:
        """
        Call atoms explainer for a molecular graph

        Parameters
        ----------
        data : torch_geometric.data.Data
            Data describing the molecular graph
        atom_relation : Callable[[int, int], bool] or None
            Boolean function with two arguments representing an equivalence  
            relation on the atoms of the molecular graph.

        Returns
        -------
        atoms_scores : list
            List of scores for the atoms of the molecular graph.
        """

        self.data = Batch.from_data_list([data])
        self.data.to(self.device)  # Allocate data on the device

        graph = to_networkx(data, to_undirected=True)
        if atom_relation is None:
            atom_relation = lambda x, y: x == y
        self.factor_graph = nx.quotient_graph(graph, atom_relation)

        scores = {cls: 0 for cls in self.factor_graph.nodes()}
        for coalition in factor_graph_coalitions(self.factor_graph):
            coalition_scores = self.shapley(coalition)
            for cls, score in zip(coalition, coalition_scores):
                scores[cls] += score

        atoms_scores = torch.zeros(self.data.num_nodes,
                                   dtype=torch.float32, device=self.device)
        for cls, score in scores.items():
            atoms_scores[list(cls)] = score

        return atoms_scores.tolist()

    def all_neighborhoods(self, coalition: List[FrozenSet[int]]):
        return [
            cls for cls in self.factor_graph.nodes() if cls not in coalition
        ]

    def local_neighborhoods(self, coalition: List[FrozenSet[int]]):
        neighborhood = set(coalition)
        for _ in range(self.local_radius - 1):
            neighborhood = set((
                x for cls in neighborhood for x in self.factor_graph[cls]
            ))
        return [cls for cls in neighborhood if cls not in coalition]

    def all_subsets(self, neighborhoods: List[FrozenSet[int]]) \
            -> Tuple[int, List[FrozenSet[int]]]:
        for s in range(len(neighborhoods) + 1):
            for subset in combinations(neighborhoods, s):
                yield list(subset)

    def sample_subsets(self, neighborhoods: List[FrozenSet[int]]) \
            -> Tuple[int, List[FrozenSet[int]]]:
        for _ in range(self.sample_num):
            yield list(filter(lambda x: randrange(2), neighborhoods))

    def mean_score(self,
                   marginal_score: torch.Tensor,
                   neighborhoods: List[FrozenSet[int]]):
        return marginal_score.mean(dim=0)

    def shapley_score(self,
                      marginal_score: torch.Tensor,
                      neighborhoods: List[FrozenSet[int]]):

        p = len(neighborhoods)
        coeff = torch.tensor([
            1.0 / c / (p + 1)
            for s in range(p + 1)
            for c in [comb(p, s, exact=True)]
            for _ in range(c)
        ], dtype=torch.float32, device=self.device)

        return torch.matmul(marginal_score.transpose(0, 1), coeff)

    def shapley(self, coalition: List[FrozenSet[int]]):
        """
        Compute shapley values of all atoms
          where players are local neighbor nodes """

        neighborhoods = self.neighborhoods_method(coalition)

        include_mask_list = []
        exclude_mask_list = []

        for subset in self.sample_method(neighborhoods):
            atoms = coalition_atoms(coalition + subset)
            include_mask = torch.zeros(self.data.num_nodes,
                                       dtype=torch.bool, device=self.device)
            exclude_mask = torch.zeros(len(coalition),
                                       self.data.num_nodes,
                                       dtype=torch.bool, device=self.device)
            include_mask[atoms] = 1.0
            for j, cls in enumerate(coalition):
                exclude_mask[j][atoms] = 1.0
                exclude_mask[j][list(cls)] = 0.0
            include_mask_list.append(include_mask)
            exclude_mask_list.append(exclude_mask)

        include_mask = torch.stack(include_mask_list, dim=0)
        exclude_mask = torch.stack(exclude_mask_list, dim=0)

        ms = marginal_score(self.data,
                            include_mask,
                            exclude_mask,
                            self.value_func)

        return self.shapley_method(ms, neighborhoods)
