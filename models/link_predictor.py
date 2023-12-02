import torch
import torch.nn as nn
from gnnfree.nn.models.task_predictor import BaseLinkEncoder
from gnnfree.nn.pooling import *
import dgl


def get_indices(t1,t2):
    t1 = t1.unsqueeze(0)
    t2 = t2.unsqueeze(1)
    #print(t1.shape,t2.shape)
    return torch.sum(torch.where((t1==t2).all(dim=2),torch.arange(t1.shape[0],device=t1.device),0),dim=1)


class GDLinkPredictor(BaseLinkEncoder):
    def __init__(self, emb_dim, gnn, feature_list, num_rels=None):

        super().__init__(emb_dim, gnn)
        self.feature_list = feature_list
        self.num_rels = num_rels
        self.link_dim = 0
        self.build_predictor()

    def build_predictor(self):
        self.feature_module = nn.ModuleDict()
        for ft in self.feature_list:
            if ft == "":
                continue
            if ft == "dist":
                self.feature_module[ft] = IdentityTransform(1)
            elif ft == "head":
                self.feature_module[ft] = ReprIndexTransform(self.emb_dim)
            elif ft == "tail":
                self.feature_module[ft] = ReprIndexTransform(self.emb_dim)
            elif ft == "HeadVerGD":
                self.feature_module[ft] = VerGDTransform(
                    self.emb_dim, gd_deg=False
                )
            elif ft == "HeadVerGDDeg":
                self.feature_module[ft] = VerGDTransform(
                    self.emb_dim, gd_deg=True
                )
            elif ft == 'HeadVerGDHet':
                self.feature_module[ft] = VerGDTransform(
                    self.emb_dim, gd_deg=False, heterogeneous=True, rel_type_emb_dim=self.emb_dim
                )
            elif ft == 'HeadVerGDDegHet':
                self.feature_module[ft] = VerGDTransform(
                    self.emb_dim, gd_deg=True, heterogeneous=True, rel_type_emb_dim=self.emb_dim
                )
            elif ft == "TailVerGD":
                self.feature_module[ft] = VerGDTransform(
                    self.emb_dim, gd_deg=False
                )
            elif ft == "TailVerGDDeg":
                self.feature_module[ft] = VerGDTransform(
                    self.emb_dim, gd_deg=True
                )
            elif ft == 'TailVerGDHet':
                self.feature_module[ft] = VerGDTransform(
                    self.emb_dim, gd_deg=False, heterogeneous=True, rel_type_emb_dim=self.emb_dim
                )
            elif ft == 'TailVerGDDegHet':
                self.feature_module[ft] = VerGDTransform(
                    self.emb_dim, gd_deg=True, heterogeneous=True, rel_type_emb_dim=self.emb_dim
                )
            elif ft == 'HeadVerGDAttn':
                self.feature_module[ft] = VerGDAttnTransform(
                    self.emb_dim, gd_deg=False
                )
            elif ft == "HeadVerGDDegAttn":
                self.feature_module[ft] = VerGDTransform(
                    self.emb_dim, gd_deg=True
                )
            elif ft == "TailVerGDAttn":
                self.feature_module[ft] = VerGDAttnTransform(
                    self.emb_dim, gd_deg=False
                )
            elif ft == "TailVerGDDegAttn":
                self.feature_module[ft] = VerGDAttnTransform(
                    self.emb_dim, gd_deg=True
                )
            elif ft == "HorGD":
                self.feature_module[ft] = ScatterReprTransform(self.emb_dim)
            elif ft == "Rel" and self.num_rels is not None:
                self.relemb = self.feature_module[ft] = EmbTransform(
                    self.emb_dim, self.num_rels
                )
            else:
                raise NotImplementedError
            self.link_dim += self.feature_module[ft].get_out_dim()

    def get_out_dim(self):
        return self.link_dim
    
    def pool_from_link(self, repr, head, tail, input):
        repr_list = []
        embs = self.relemb(input.rel)
        for ft in self.feature_list:
            if ft == "":
                continue
            if ft == "dist":
                repr_list.append(self.feature_module[ft](input.dist))
            elif ft == "head":
                repr_list.append(self.feature_module[ft](repr, head))
            elif ft == "tail":
                repr_list.append(self.feature_module[ft](repr, tail))
            elif ft == "HeadVerGD" or ft == "HeadVerGDAttn":
                repr_list.append(
                    self.feature_module[ft](
                        repr, input.head_gd, input.head_gd_len, None
                    )
                )
            elif ft == "HeadVerGDDeg" or ft == "HeadVerGDDegAttn":
                repr_list.append(
                    self.feature_module[ft](
                        repr,
                        input.head_gd,
                        input.head_gd_len,
                        input.head_gd_deg,
                    )
                )
            elif ft == "HeadVerGDHet":
                edges = torch.cat([input.head.unsqueeze(1),input.tail.unsqueeze(1)], dim=1)
                hgd_edges = torch.cat([torch.repeat_interleave(input.head,input.head_gd_len).unsqueeze(1),input.head_gd.unsqueeze(1)],dim=1)
                repr_list.append(
                    self.feature_module[ft](
                        repr,
                        input.head_gd,
                        input.head_gd_len,
                        input.head_gd_deg,
                        embs[get_indices(edges,hgd_edges)],
                    )
                )
            elif ft == "HeadVerGDDegHet":
                edges = torch.cat([input.head.unsqueeze(1),input.tail.unsqueeze(1)], dim=1)
                hgd_edges = torch.cat([torch.repeat_interleave(input.head,input.head_gd_len).unsqueeze(1),input.head_gd.unsqueeze(1)],dim=1)
                repr_list.append(
                    self.feature_module[ft](
                        repr,
                        input.head_gd,
                        input.head_gd_len,
                        input.head_gd_deg,
                        embs[get_indices(edges,hgd_edges)],
                    )
                )
            elif ft == "TailVerGD" or ft == "TailVerGDAttn":
                repr_list.append(
                    self.feature_module[ft](
                        repr, input.tail_gd, input.tail_gd_len, None
                    )
                )
            elif ft == "TailVerGDDeg" or ft == "TailVerGDDegAttn":
                repr_list.append(
                    self.feature_module[ft](
                        repr,
                        input.tail_gd,
                        input.tail_gd_len,
                        input.tail_gd_deg,
                    )
                )
            elif ft == "TailVerGDHet":
                edges = torch.cat([input.head.unsqueeze(1),input.tail.unsqueeze(1)], dim=1)
                tgd_edges = torch.cat([input.tail_gd.unsqueeze(1), torch.repeat_interleave(input.tail,input.tail_gd_len).unsqueeze(1)],dim=1)
                repr_list.append(
                    self.feature_module[ft](
                        repr,
                        input.tail_gd,
                        input.tail_gd_len,
                        input.tail_gd_deg,
                        embs[get_indices(edges,tgd_edges)],
                    )
                )
            elif ft == "TailVerGDDegHet":
                edges = torch.cat([input.head.unsqueeze(1),input.tail.unsqueeze(1)], dim=1)
                tgd_edges = torch.cat([input.tail_gd.unsqueeze(1), torch.repeat_interleave(input.tail,input.tail_gd_len).unsqueeze(1)],dim=1)
                
                repr_list.append(
                    self.feature_module[ft](
                        repr,
                        input.tail_gd,
                        input.tail_gd_len,
                        input.tail_gd_deg,
                        embs[get_indices(edges,tgd_edges)],
                    )
                )
            elif ft == "HorGD":
                repr_list.append(
                    self.feature_module[ft](repr, input.gd, input.gd_len)
                )
            elif ft == "Rel" and self.num_rels is not None:
                repr_list.append(embs)
        g_rep = torch.cat(repr_list, dim=1)
        return g_rep
