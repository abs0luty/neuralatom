from torch_geometric.graphgym.register import register_config


def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False
    cfg.gnn.num_vns = 0
    cfg.gvm.avg_nodes = 50
    cfg.gvm.pool_ratio = 0.25
    cfg.gvm.n_pool_heads = 2
    cfg.gvm.na_order = "desc" # fixed, incre

    # Dynamic clustering parameters
    cfg.gvm.use_dynamic_clustering = False  # Set to True to enable learnable cluster count
    cfg.gvm.max_clusters = 50  # Maximum number of clusters for dynamic mode
    cfg.gvm.min_clusters = 3   # Minimum number of clusters for dynamic mode


register_config("custom_gnn", custom_gnn_cfg)
