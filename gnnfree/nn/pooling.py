import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABCMeta, abstractmethod
from torch_scatter import scatter

from gnnfree.nn.models.basic_models import MLPLayers
from gnnfree.utils import count_to_group_index


class Extractor(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def get_out_dim(self):
        pass

    @abstractmethod
    def forward(self):
        pass


class Pooler(Extractor):
    def __init__(self):
        super().__init__()

    def get_out_dim(self):
        return None


class Transform(Extractor):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

    def get_out_dim(self):
        return self.emb_dim


# Used only for node classfn and graph classfn level pooling
class GDTransform(Transform):
    """compute node-level GD representation."""

    def __init__(self, emb_dim, gd_deg=True, batch_norm=True) -> None:
        super().__init__(emb_dim)
        self.gd_deg = gd_deg    
        if gd_deg:
            self.mlp_combine_gd_deg = MLPLayers(
                2,
                h_units=[emb_dim + 1, 2 * emb_dim, emb_dim],
                batch_norm=False,
            )
        self.mlp_combine_nei_gd = MLPLayers(
            2,
            h_units=[2 * emb_dim + 1, 4 * emb_dim, emb_dim],
            batch_norm=batch_norm,
        )
        self.mlp_combine_node_nei = MLPLayers(
            2,
            h_units=[2 * emb_dim, 4 * emb_dim, emb_dim],
            batch_norm=batch_norm,
        )

    def forward(
        self,
        repr,
        nodes,
        neighbors,
        neighbor_count,
        dist,
        gd,
        gd_count,
        gd_deg,
    ):
        """

        Arguments:
            repr {torch.tensor} -- N*d tensor of node representations
            nodes {torch.tensor} -- 1-d nodes of interests
            neighbors {torch.tensor} -- neighbors indices of nodes
            neighbor_count {torch.tensor} -- nodes[i]'s neighbor=
            neighbors[cumsum(neighbor_count[:i]):cumsum(neighbor_count[:i+1])]
            dist {torch.tensor} -- dist[i]=distance between neighbors[i] and
            its corresponding source in nodes.
            gd {torch.tensor} -- Vertical Geodesics indices of neighbors
            gd_count {torch.tensor} -- neighbors[i]'s vertical geodesics=
                    gd[cumsum(gd_count[i]):cumsum(gd_count[i+1])]
            gd_deg {torch.tensor} -- same shape as gd, vertical gd degrees.

        Returns:
            Vertical geodesics of each node in nodes.
        """
        neighbors_repr = repr[neighbors]
        gd_repr = repr[gd]
        if self.gd_deg:
            combined_gd_repr = self.mlp_combine_gd_deg(
                torch.cat([gd_repr, gd_deg.view(-1, 1)], dim=-1)
            )
        else:
            combined_gd_repr = gd_repr
        combined_gd_repr = scatter(
            combined_gd_repr,
            count_to_group_index(gd_count),
            dim=0,
            dim_size=len(gd_count),
        )
        combined_repr = self.mlp_combine_nei_gd(
            torch.cat(
                [combined_gd_repr, neighbors_repr, dist.view(-1, 1)], dim=-1
            )
        )
        combined_repr = scatter(
            combined_repr,
            count_to_group_index(neighbor_count),
            dim=0,
            dim_size=len(neighbor_count),
        )

        node_repr = self.mlp_combine_node_nei(
            torch.cat([combined_repr, repr[nodes]], dim=-1)
        )
        return node_repr

class GDAttnTransform(Transform):
    """compute node-level GD representation."""

    def __init__(self, emb_dim, gd_deg=True, batch_norm=True) -> None:
        super().__init__(emb_dim)
        self.gd_deg = gd_deg    
        if gd_deg:
            self.mlp_combine_gd_deg = MLPLayers(
                2,
                h_units=[emb_dim + 1, 2 * emb_dim, emb_dim],
                batch_norm=False,
            )
        self.mlp_combine_nei_gd = MLPLayers(
            2,
            h_units=[2 * emb_dim + 1, 4 * emb_dim, emb_dim],
            batch_norm=batch_norm,
        )
        self.mlp_combine_node_nei = MLPLayers(
            2,
            h_units=[2 * emb_dim, 4 * emb_dim, emb_dim],
            batch_norm=batch_norm,
        )
        ## one attention like mechanism, like qkv
        self.Q = nn.Linear(self.emb_dim,self.emb_dim)
        self.K = nn.Linear(self.emb_dim,self.emb_dim)
        self.V = nn.Linear(self.emb_dim,self.emb_dim)

    def forward(
        self,
        repr,
        nodes,
        neighbors,
        neighbor_count,
        dist,
        gd,
        gd_count,
        gd_deg,
    ):
        """

        Arguments:
            repr {torch.tensor} -- N*d tensor of node representations
            nodes {torch.tensor} -- 1-d nodes of interests
            neighbors {torch.tensor} -- neighbors indices of nodes
            neighbor_count {torch.tensor} -- nodes[i]'s neighbor=
            neighbors[cumsum(neighbor_count[:i]):cumsum(neighbor_count[:i+1])]
            dist {torch.tensor} -- dist[i]=distance between neighbors[i] and
            its corresponding source in nodes.
            gd {torch.tensor} -- Vertical Geodesics indices of neighbors
            gd_count {torch.tensor} -- neighbors[i]'s vertical geodesics=
                    gd[cumsum(gd_count[i]):cumsum(gd_count[i+1])]
            gd_deg {torch.tensor} -- same shape as gd, vertical gd degrees.

        Returns:
            Vertical geodesics of each node in nodes.
        """
        neighbors_repr = repr[neighbors]
        gd_repr = repr[gd]
        if self.gd_deg:
            gd_repr = self.mlp_combine_gd_deg(
                torch.cat([gd_repr, gd_deg.view(-1, 1)], dim=-1)
            )
        og_nodes = torch.repeat_interleave(repr,gd_count)
        attnwts = torch.sum(self.Q(og_nodes)*self.K(gd_repr),dim=1)/torch.sqrt(self.emb_dim)
        attnwts = F.sigmoid(attnwts)
        gd_repr = self.V(gd_repr)
        ### embs MUST be in the correct order
        combined_gd_repr = scatter(
            gd_repr,
            count_to_group_index(gd_count),
            dim=0,
            dim_size=len(gd_len),
            weight= attnwts,
            reduce='mean'
        )
        
        combined_repr = self.mlp_combine_nei_gd(
            torch.cat(
                [combined_gd_repr, neighbors_repr, dist.view(-1, 1)], dim=-1
            )
        )
        combined_repr = scatter(
            combined_repr,
            count_to_group_index(neighbor_count),
            dim=0,
            dim_size=len(neighbor_count),
        )

        node_repr = self.mlp_combine_node_nei(
            torch.cat([combined_repr, repr[nodes]], dim=-1)
        )
        return node_repr

class ReprIndexTransform(Transform):
    def forward(self, repr, ind):
        return repr[ind]


class EmbTransform(Transform):
    def __init__(self, emb_dim, num_embs) -> None:
        super().__init__(emb_dim)
        self.emb = nn.Embedding(num_embs, emb_dim, sparse=False)

    def forward(self, ind):
        return self.emb(ind)

# corresponds to the HorGD
class ScatterReprTransform(Transform):
    def __init__(self, emb_dim, scatter_method="sum"):
        super().__init__(emb_dim)
        self.scatter_method = scatter_method

    def forward(self, repr, ind, ind_block):
        gd_repr = repr[ind]
        gd_repr = scatter(
            gd_repr,
            count_to_group_index(ind_block),
            dim=0,
            dim_size=len(ind_block),
            reduce=self.scatter_method,
        )
        return gd_repr


class VerGDTransform(Transform):
    """Vertical GD representation for links"""

    def __init__(self, emb_dim, gd_deg=False,heterogeneous:bool=False,rel_type_emb_dim=None) -> None:
        super().__init__(emb_dim)
        self.gd_deg = gd_deg
        self.heterogeneous = heterogeneous
        if self.heterogeneous:
            if rel_type_emb_dim is None:
                rel_type_emb_dim = emb_dim
            self.rel_embedder = MLPLayers(
                3, h_units=[emb_dim+rel_type_emb_dim, 2 * emb_dim, 2*emb_dim, emb_dim]
            )
            
        if gd_deg:
            self.mlp_combine_gd_deg = MLPLayers(
                2, h_units=[emb_dim + 1, 2 * emb_dim, emb_dim]
            )
        self.mlp_gd_process = MLPLayers(
            2, h_units=[emb_dim, 2 * emb_dim, emb_dim]
        )

    def get_ver_gd_one_side(self, repr, gd, gd_len, gd_deg, embs=None):
        """Get vertical geodesics of one side.

        Arguments:
            repr {torch.tensor} -- graph node representations
            gd {torch.tensor} -- vertical geodesics
            gd_len {torch.tensor} -- count of each vertical geodesics
            gd_deg {torch.tensor} -- degrees of each vertical gd node
        """
        gd_repr = repr[gd]
        if gd_deg:
            gd_repr = self.mlp_combine_gd_deg(
                torch.cat([gd_repr, gd_deg.view(-1, 1)], dim=-1)
            )
        ### embs MUST be in the correct order
        if self.heterogeneous:
            gd_repr = self.rel_embedder(
                torch.cat([gd_repr, embs], dim=-1)
            )
        gd_repr = scatter(
            gd_repr,
            count_to_group_index(gd_len),
            dim=0,
            dim_size=len(gd_len),
        )
        return gd_repr

    def forward(
        self,
        repr,
        gd,
        gd_len,
        gd_deg=None,
        embs=None
    ):
        gd_repr = self.get_ver_gd_one_side(repr, gd, gd_len, gd_deg,embs)
        return self.mlp_gd_process(gd_repr)

class VerGDAttnTransform(VerGDTransform):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        ## one attention like mechanism, like qkv
        self.Q = nn.Linear(self.emb_dim,self.emb_dim)
        self.K = nn.Linear(self.emb_dim,self.emb_dim)
        self.V = nn.Linear(self.emb_dim,self.emb_dim)


    def get_ver_gd_one_side(self, repr, gd, gd_len, gd_deg,embs=None):
        
        
        gd_repr = repr[gd]
        
        if gd_deg:
            gd_repr = self.mlp_combine_gd_deg(
                torch.cat([gd_repr, gd_deg.view(-1, 1)], dim=-1)
            )
            
        if self.heterogeneous:
            gd_repr = self.rel_embedder(
                torch.cat([gd_repr, embs], dim=-1)
            )
        
        og_nodes = torch.repeat_interleave(repr,gd_len)
        attnwts = torch.sum(self.Q(og_nodes)*self.K(gd_repr),dim=1)/torch.sqrt(self.emb_dim)
        attnwts = F.sigmoid(attnwts)
        gd_repr = self.V(gd_repr)
        ### embs MUST be in the correct order
        
        gd_repr = scatter(
            gd_repr,
            count_to_group_index(gd_len),
            dim=0,
            dim_size=len(gd_len),
            weight= attnwts,
            reduce='mean'
        )
        return gd_repr
    

class IdentityTransform(Transform):
    def __init__(self, emb_dim):
        super().__init__(emb_dim)

    def forward(self, value):
        return value.view(-1, self.emb_dim)


# feature_module_dict = {
#     "node_seq": NodeSeqExtractor,
#     "emb": EmbExtractor,
#     "HorGD": ScatterReprPooler,
#     "VerGD": VerGDExtractor,
#     "Id": IdentityExtractor,
#     "": None,
# }
