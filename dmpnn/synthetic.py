"""
This file contains the code for generating synthetic molecular graphs to test our implementation
"""

import torch
import random
from .graph_utils import make_adjacency, make_directed, graph_signature


# Fixed target rule
#    y = 2*(# double bonds touching oxygen)
#      + 1*(# C-N adjacencies)
#      - 0.5*(# charged atoms)
def compute_target(X, edges_undir, E_undir):
    y = 0.0

    # count charged atoms
    num_charged = int(X[:, 3].sum().item())

    for i, (u, v) in enumerate(edges_undir):
        bond_feat = E_undir[i]

        is_double = bond_feat[1].item() > 0.5

        u_is_O = X[u, 1].item() > 0.5
        v_is_O = X[v, 1].item() > 0.5

        u_is_C = X[u, 0].item() > 0.5
        v_is_C = X[v, 0].item() > 0.5

        u_is_N = X[u, 2].item() > 0.5
        v_is_N = X[v, 2].item() > 0.5

        if is_double and (u_is_O or v_is_O):
            y += 2.0

        if (u_is_C and v_is_N) or (u_is_N and v_is_C):
            y += 1.0

    y -= 0.5 * num_charged
    return torch.tensor([y], dtype=torch.float32)


# make hand-built demo graphs
def make_demo_graphs():
    graphs = []

    # Graph 1: carbonyl-like
    X1 = torch.tensor([
        [1., 0., 0., 0.],  # C
        [0., 1., 0., 0.],  # O
        [1., 0., 0., 0.],  # C
    ])
    edges1 = [(0, 1), (0, 2)]
    E1 = torch.tensor([
        [0., 1.],  # double
        [1., 0.],  # single
    ])
    graphs.append(build_graph(X1, edges1, E1))

    # Graph 2: amine chain
    X2 = torch.tensor([
        [1., 0., 0., 0.],  # C
        [0., 0., 1., 0.],  # N
        [1., 0., 0., 0.],  # C
        [0., 1., 0., 0.],  # O
    ])
    edges2 = [(0, 1), (1, 2), (2, 3)]
    E2 = torch.tensor([
        [1., 0.],
        [1., 0.],
        [1., 0.],
    ])
    graphs.append(build_graph(X2, edges2, E2))

    # Graph 3: charged amide-like
    X3 = torch.tensor([
        [1., 0., 0., 0.],  # C
        [0., 1., 0., 0.],  # O
        [0., 0., 1., 1.],  # N charged
        [1., 0., 0., 0.],  # C
    ])
    edges3 = [(0, 1), (0, 2), (2, 3)]
    E3 = torch.tensor([
        [0., 1.],  # double
        [1., 0.],  # single
        [1., 0.],  # single
    ])
    graphs.append(build_graph(X3, edges3, E3))

    return graphs


def random_node_feature(rng):
    # one of C, O, N
    atom_type = rng.choice(["C", "O", "N"])
    charge = rng.choice([0., 0., 0., 1.])  # mostly uncharged

    if atom_type == "C":
        return [1., 0., 0., charge]
    elif atom_type == "O":
        return [0., 1., 0., charge]
    else:
        return [0., 0., 1., charge]


def random_bond_feature(rng):
    # mostly single, sometimes double
    is_double = rng.random() < 0.3
    if is_double:
        return [0., 1.]
    return [1., 0.]


def generate_random_chain_graph(rng, min_nodes=3, max_nodes=5):
    num_nodes = random.randint(min_nodes, max_nodes)

    X = torch.tensor([random_node_feature(rng) for _ in range(num_nodes)], dtype=torch.float32)

    # simple connected chain
    edges_undir = [(i, i + 1) for i in range(num_nodes - 1)]

    E_undir = torch.tensor(
        [random_bond_feature(rng) for _ in edges_undir],
        dtype=torch.float32
    )

    return build_graph(X, edges_undir, E_undir)


def generate_random_graph(
    rng,
    min_nodes=3,
    max_nodes=10,
    extra_edge_prob=0.25,
    max_extra_edges=None
):
    num_nodes = rng.randint(min_nodes, max_nodes)

    X = torch.tensor(
        [random_node_feature(rng) for _ in range(num_nodes)],
        dtype=torch.float32
    )

    # make a random connected tree
    edges_set = set()
    for new_node in range(1, num_nodes):
        attach_to = rng.randint(0, new_node - 1)
        u, v = sorted((new_node, attach_to))
        edges_set.add((u, v))

    # consider adding extra edges to create cycles/rings
    possible_extra = []
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if (u, v) not in edges_set:
                possible_extra.append((u, v))

    rng.shuffle(possible_extra)

    if max_extra_edges is None:
        max_extra_edges = max(1, num_nodes // 3)

    num_added = 0
    for (u, v) in possible_extra:
        if num_added >= max_extra_edges:
            break
        if rng.random() < extra_edge_prob:
            edges_set.add((u, v))
            num_added += 1

    edges_undir = sorted(edges_set)

    E_undir = torch.tensor(
        [random_bond_feature(rng) for _ in edges_undir],
        dtype=torch.float32
    )

    return build_graph(X, edges_undir, E_undir)

# build one graph record
def build_graph(X, edges_undir, E_undir):
    A = make_adjacency(X.shape[0], edges_undir)
    edge_index, B = make_directed(edges_undir, E_undir)
    y = compute_target(X, edges_undir, E_undir)

    return {
        "A": A,
        "X": X.float(),
        "edges_undir": edges_undir,
        "E_undir": E_undir.float(),
        "edge_index": edge_index,
        "B": B,
        "y": y,
    }

def generate_unique_graphs(n, existing_sigs=None, min_nodes=3, max_nodes=10, seed=42):
    rng = random.Random(seed)
    graphs = []
    sigs = set() if existing_sigs is None else set(existing_sigs)

    while len(graphs) < n:
        g = generate_random_graph(rng, min_nodes=min_nodes, max_nodes=max_nodes)
        sig = graph_signature(g)

        if sig not in sigs:
            graphs.append(g)
            sigs.add(sig)

    return graphs, sigs