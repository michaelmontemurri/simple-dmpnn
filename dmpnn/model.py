"""
This file contains the implementation of the Directed Message Passing Neural Network,
outlined in "Analyzing Learned Molecular Representations for Property Prediction" by Yang et al. 
"""

import torch
import torch.nn as nn


class DMPNNEncoder(nn.Module):
    """Directed message passing encoder for graph-level prediction.

    This module maintains hidden states on directed edges, updates them with
    non-backtracking message passing, aggregates them to nodes, and finally
    pools node representations into a graph embedding.
    """
    def __init__(
        self,
        node_feat_dim,
        edge_feat_dim,
        hidden_dim,
        num_steps
    ):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps

        self.W_i = nn.Parameter(torch.empty(hidden_dim, node_feat_dim+edge_feat_dim))
        self.W_h = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.W_a = nn.Parameter(torch.empty(hidden_dim,hidden_dim+node_feat_dim))

        #initialize weight matrices
        torch.nn.init.xavier_uniform_(self.W_i)
        torch.nn.init.xavier_uniform_(self.W_h)
        torch.nn.init.xavier_uniform_(self.W_a)


    def initialize_edge_states(self, X, B, edge_index):
        """Initialize directed edge hidden states from node and edge features.

        Args:
            X: Node feature tensor of shape ``[num_nodes, node_feat_dim]``.
            B: Directed edge feature tensor of shape
                ``[num_directed_edges, edge_feat_dim]``.
            edge_index: Directed COO edge tensor of shape ``[2, num_edges]``.

        Returns:
            A tuple ``(H_0, src, rcv)`` where ``H_0`` has shape
            ``[num_directed_edges, hidden_dim]`` and ``src``/``rcv`` are the
            source and receiver index vectors extracted from ``edge_index``.
        """
        src, rcv = edge_index
        X_src = X[src]
        C = torch.cat([X_src, B], dim=1)
        H_0 = torch.relu(torch.matmul(C, self.W_i.T))
        return H_0, src, rcv


    def compute_messages(self, H, X, src, rcv, rev_index):
        """Compute non-backtracking directed messages for all edges.

        Args:
            H: Current directed edge states of shape ``[num_edges, hidden_dim]``.
            X: Node feature tensor, used only for sizing the accumulation buffer.
            src: Source-node indices for each directed edge.
            rcv: Receiver-node indices for each directed edge.
            rev_index: Long tensor of shape ``[num_edges]`` mapping each edge to
                the index of its reverse edge.

        Returns:
            Tensor of shape ``[num_edges, hidden_dim]`` containing the incoming
            message to each directed edge after removing the backtracking term.
        """
        incoming_sum_per_node = torch.zeros(
            X.shape[0],
            H.shape[1],
            dtype=H.dtype,
            device=H.device,
        )
        incoming_sum_per_node.index_add_(0, rcv, H) #(dim, index, source)
        M = incoming_sum_per_node[src] - H[rev_index] #remove backtracking messages
        return M


    def update(self, H_0, M):
        """Apply the D-MPNN edge-state update."""
        return torch.relu(H_0 + torch.matmul(M, self.W_h.T))


    def aggregate_to_nodes(self, H, X, rcv):
        """Aggregate directed edge states to node representations."""
        incoming_sum_per_node = torch.zeros(
            X.shape[0],
            H.shape[1],
            dtype=H.dtype,
            device=H.device,
        )
        incoming_sum_per_node.index_add_(0, rcv, H)
        R = torch.cat([X, incoming_sum_per_node], dim=1) #[N,F+d]
        P = torch.relu(torch.matmul(R, self.W_a.T))
        return P

    def node_pooling(self, P, batch_vec, num_graphs):
        """Pool node embeddings into graph embeddings by summation.

        Args:
            P: Node embedding tensor of shape ``[num_nodes, hidden_dim]``.
            batch_vec: Long tensor of shape ``[num_nodes]`` assigning each node
                to a graph in the batch.
            num_graphs: Number of graphs in the batch.

        Returns:
            Tensor of shape ``[num_graphs, hidden_dim]``.
        """
        Z_G = torch.zeros(num_graphs, P.shape[1], dtype=P.dtype, device=P.device)
        Z_G.index_add_(0, batch_vec, P)
        return Z_G


    def forward(self, X, B, edge_index, rev_index, batch_vec, num_graphs):
        """Encode a batch of graphs into graph-level embeddings.

        Args:
            X: Node features of shape ``[num_nodes, node_feat_dim]``.
            B: Directed edge features of shape ``[num_edges, edge_feat_dim]``.
            edge_index: Directed COO edge tensor of shape ``[2, num_edges]``.
            rev_index: Reverse-edge lookup tensor of shape ``[num_edges]``.
            batch_vec: Graph assignment vector of shape ``[num_nodes]``.
            num_graphs: Number of graphs in the batch.

        Returns:
            Graph embedding tensor of shape ``[num_graphs, hidden_dim]``.
        """
        #initialize hidden states
        H_0, src, rcv = self.initialize_edge_states(X, B, edge_index)
        H = H_0
        for _ in range(self.num_steps):
            M = self.compute_messages(H, X, src=src, rcv=rcv, rev_index=rev_index)
            H = self.update(H_0, M) #skip connection, not residual connection
        P = self.aggregate_to_nodes(H, X, rcv)
        Z_G = self.node_pooling(P, batch_vec, num_graphs=num_graphs)
        return Z_G
    

class MLP(nn.Module):
    """Simple feedforward head used after graph pooling."""
    def __init__(self, input_size,  hidden_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
class DMPNN(nn.Module):
    """Graph-level predictor built from a D-MPNN encoder and an MLP head.

    Can be used for regressiong or classification as long as the trainer/loss are configured appropriately.
    """
    def __init__(
        self,
        node_feat_dim,
        edge_feat_dim,
        hidden_dim,
        num_steps,
        head_hidden_size,
        output_size=1,
    ):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.head_hidden_size = head_hidden_size
        self.output_size = output_size

        self.encoder = DMPNNEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            num_steps=num_steps,
        )

        self.head = MLP(
            input_size=hidden_dim,
            hidden_size=head_hidden_size,
            output_size=output_size,
        )

    def forward(self, X, B, edge_index, rev_index, batch_vec, num_graphs):
        """Predict graph-level outputs for a batched set of graphs."""
        z_g = self.encoder(X, B, edge_index, rev_index, batch_vec, num_graphs)   # [num_graphs, hidden_dim]
        y_hat = self.head(z_g)                            # [num_graphs, output_size]
        return y_hat


    def get_config(self):
        """Return constructor arguments needed to rebuild this model."""
        return {
            "node_feat_dim": self.node_feat_dim,
            "edge_feat_dim": self.edge_feat_dim,
            "hidden_dim": self.hidden_dim,
            "num_steps": self.num_steps,
            "head_hidden_size": self.head_hidden_size,
            "output_size": self.output_size,
        }


    @classmethod
    def from_config(cls, config):
        """Instantiate a model from ``get_config()`` output."""
        return cls(**config)
