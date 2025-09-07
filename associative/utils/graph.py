"""Graph utilities for associative memory models.

This module provides utilities for processing graph data structures including
positional encoding computation, batch preparation for graph transformers,
and mask generation for graph-based masked language modeling tasks.
"""

import torch
from torch import Tensor
from torch.nn import functional
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj, to_dense_batch


def get_graph_positional_encoding(
    edge_index: Tensor,
    num_nodes: int,
    k: int = 10,
    method: str = "eigen",
) -> Tensor:
    """Compute positional encodings for graph nodes.

    Args:
        edge_index: Edge connectivity [2, num_edges]
        num_nodes: Number of nodes in the graph
        k: Number of eigenvectors to use
        method: Encoding method ('eigen' or 'svd')

    Returns:
        Positional encodings [num_nodes, k] or [num_nodes, 2*k] for svd
    """
    # Convert to adjacency matrix
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]  # [num_nodes, num_nodes]

    # Add self-loops
    adj = adj + torch.eye(num_nodes, device=adj.device, dtype=adj.dtype)

    # Compute degree matrix
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    # Normalized Laplacian
    laplacian = torch.eye(
        num_nodes, device=adj.device, dtype=adj.dtype
    ) - deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)

    if method == "eigen":
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        # Use smallest k eigenvectors
        actual_k = min(k, eigenvectors.shape[1])
        pos_encoding = eigenvectors[:, :actual_k]
        # Pad if we have fewer eigenvectors than requested
        if actual_k < k:
            pos_encoding = functional.pad(pos_encoding, (0, k - actual_k))
    elif method == "svd":
        # SVD decomposition
        u, s, v = torch.linalg.svd(laplacian)
        # Concatenate left and right singular vectors
        actual_k = min(k, u.shape[1])
        pos_encoding = torch.cat([u[:, :actual_k], v[:actual_k, :].T], dim=1)
        # Pad if needed
        if pos_encoding.shape[1] < 2 * k:
            pos_encoding = functional.pad(
                pos_encoding, (0, 2 * k - pos_encoding.shape[1])
            )
    else:
        raise ValueError(f"Unknown encoding method: {method}")

    return pos_encoding


def prepare_graph_batch(
    batch: Batch,
    max_num_nodes: int = 500,
    pos_encoding_k: int = 10,
    pos_encoding_method: str = "eigen",
    use_edge_attr: bool = True,
    add_cls_token: bool = True,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor, Tensor]:
    """Prepare graph batch for associative memory models.

    Args:
        batch: PyG batch object
        max_num_nodes: Maximum number of nodes per graph
        pos_encoding_k: Number of positional encoding dimensions
        pos_encoding_method: Method for positional encoding
        use_edge_attr: Whether to use edge attributes
        add_cls_token: Whether to add CLS token

    Returns:
        node_features: [batch_size, max_nodes+1, feat_dim] if add_cls_token
        adjacency: [batch_size, max_nodes+1, max_nodes+1, 1 or edge_feat_dim]
        edge_attr: Optional edge attributes
        pos_encodings: [batch_size, max_nodes+1, pos_dim]
        mask: [batch_size, max_nodes+1] indicating valid nodes
    """
    device = batch.x.device  # type: ignore[attr-defined]
    batch_size = batch.num_graphs  # type: ignore[attr-defined]

    # Convert to dense batch
    x, mask = to_dense_batch(batch.x, batch.batch, max_num_nodes=max_num_nodes)  # type: ignore[attr-defined]

    # Get adjacency matrices
    adj = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=max_num_nodes)  # type: ignore[attr-defined]

    # Handle edge attributes if present
    edge_attr = None
    if use_edge_attr and hasattr(batch, "edge_attr") and batch.edge_attr is not None:  # type: ignore[attr-defined]
        edge_attr = to_dense_adj(
            batch.edge_index,  # type: ignore[attr-defined]
            batch.batch,  # type: ignore[attr-defined]
            edge_attr=batch.edge_attr,  # type: ignore[attr-defined]
            max_num_nodes=max_num_nodes,
        )

    # Add CLS token space
    if add_cls_token:
        # Pad features
        x = functional.pad(x, (0, 0, 1, 0), value=0)  # [batch, max_nodes+1, feat_dim]

        # Pad adjacency (CLS connects to all nodes)
        adj = functional.pad(
            adj, (1, 0, 1, 0), value=1
        )  # [batch, max_nodes+1, max_nodes+1]

        # Pad mask
        mask = functional.pad(mask, (1, 0), value=True)  # [batch, max_nodes+1]

        if edge_attr is not None:
            edge_attr = functional.pad(edge_attr, (0, 0, 1, 0, 1, 0), value=1)

    # Add channel dimension to adjacency if no edge attributes
    adj = adj.unsqueeze(-1) if edge_attr is None else edge_attr

    # Compute positional encodings per graph
    pos_encodings = []
    for i in range(batch_size):
        # Get the subgraph for this sample
        node_mask = batch.batch == i  # type: ignore[attr-defined]
        num_nodes = node_mask.sum()

        if num_nodes > 0:
            # Extract edges for this graph
            edge_mask = (batch.edge_index[0] >= node_mask.nonzero()[0].min()) & (  # type: ignore[attr-defined]
                batch.edge_index[0] <= node_mask.nonzero()[-1].max()  # type: ignore[attr-defined]
            )
            graph_edges = batch.edge_index[:, edge_mask]  # type: ignore[attr-defined]

            # Shift indices to start from 0
            min_idx = graph_edges.min()
            graph_edges = graph_edges - min_idx

            # Compute positional encoding
            pos_enc = get_graph_positional_encoding(
                graph_edges, num_nodes, pos_encoding_k, pos_encoding_method
            )

            # Pad to max_num_nodes
            pos_enc = functional.pad(pos_enc, (0, 0, 0, max_num_nodes - num_nodes))
        else:
            pos_enc = torch.zeros(max_num_nodes, pos_encoding_k, device=device)

        if add_cls_token:
            # Add CLS positional encoding (zeros)
            pos_enc = functional.pad(pos_enc, (0, 0, 1, 0))

        pos_encodings.append(pos_enc)

    pos_encodings = torch.stack(pos_encodings)  # [batch, max_nodes+1, pos_dim]

    return x, adj, edge_attr, pos_encodings, mask


def create_graph_mask_indices(
    mask: Tensor,
    mask_ratio: float = 0.15,
    exclude_cls: bool = True,
) -> tuple[Tensor, Tensor]:
    """Create mask indices for graph nodes.

    Args:
        mask: Valid node mask [batch_size, num_nodes]
        mask_ratio: Ratio of nodes to mask
        exclude_cls: Whether to exclude CLS token from masking

    Returns:
        batch_indices: Batch indices for masked nodes
        node_indices: Node indices for masked nodes
    """
    batch_size, num_nodes = mask.shape
    device = mask.device

    batch_indices = []
    node_indices = []

    for b in range(batch_size):
        valid_nodes = mask[b].nonzero().squeeze(-1)

        if exclude_cls and len(valid_nodes) > 0:
            # Exclude first node (CLS)
            valid_nodes = valid_nodes[1:]

        if len(valid_nodes) > 0:
            # Number of nodes to mask
            num_mask = int(len(valid_nodes) * mask_ratio)
            if num_mask > 0:
                # Randomly select nodes to mask
                perm = torch.randperm(len(valid_nodes), device=device)[:num_mask]
                masked_nodes = valid_nodes[perm]

                batch_indices.extend([b] * len(masked_nodes))
                node_indices.extend(masked_nodes.tolist())

    if len(batch_indices) > 0:
        batch_indices = torch.tensor(batch_indices, device=device)
        node_indices = torch.tensor(node_indices, device=device)
    else:
        batch_indices = torch.tensor([], dtype=torch.long, device=device)
        node_indices = torch.tensor([], dtype=torch.long, device=device)

    return batch_indices, node_indices
