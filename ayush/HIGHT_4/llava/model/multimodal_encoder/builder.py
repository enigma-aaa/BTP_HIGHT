from .clip_encoder import CLIPVisionTower
# from .gnn_graphmvp import GraphMVP
from .vqvae import VQVAE
# from .hvqvae import HVQVAE
from .hvqvae2 import HVQVAE2
# from .moleculeSTM_gnn_model import GNN_graphpred, GNN
from .gpm import GPMTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    if vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_graph_tower(graph_tower_cfg, **kwargs):
    graph_tower = getattr(graph_tower_cfg, 'mm_graph_tower', getattr(graph_tower_cfg, 'graph_tower', None))
    # use_graph_rep = graph_tower.endswith("-graph")
    use_graph_rep = getattr(graph_tower_cfg, 'use_graph_rep', False)

    if graph_tower.lower() == "vqvae2" or graph_tower.lower() == "vqvae3":
        return VQVAE(config=graph_tower_cfg,discGNN=True,use_graph_rep=use_graph_rep,name=graph_tower)
    elif graph_tower.lower() == "hvqvae2":
        return HVQVAE2(config=graph_tower_cfg,discGNN=True,use_graph_rep=use_graph_rep,name=graph_tower)
    elif graph_tower.lower() == "gpm":
        return GPMTower(config=graph_tower_cfg,use_graph_rep=use_graph_rep,name=graph_tower)

    
    raise ValueError(f'Unknown graph tower: {graph_tower}')
