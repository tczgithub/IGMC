import torch
import torch.nn.functional as F



def Pseudo_label(y_pred, adj):
    labels = y_pred.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    mask = torch.mul(mask, adj)
    mask[mask > 0] = 1

    return mask


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, y_pred, adj, mean: bool = True, tau: float = 1.0, hidden_norm: bool = True):
    mask = Pseudo_label(y_pred, adj)
    l1 = nei_con_loss(z1, z2, tau, mask, hidden_norm)
    l2 = nei_con_loss(z2, z1, tau, mask, hidden_norm)
    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret


def sim(z1: torch.Tensor, z2: torch.Tensor, hidden_norm: bool = True):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def nei_con_loss(z1: torch.Tensor, z2: torch.Tensor, tau, adj, hidden_norm: bool = True):

    adj = adj - torch.diag_embed(adj.diag())
    adj[adj > 0] = 1
    nei_count = torch.sum(adj, 1) * 2 + 1
    nei_count = torch.squeeze(torch.tensor(nei_count))

    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z1, z1, hidden_norm))
    inter_view_sim = f(sim(z1, z2, hidden_norm))

    loss = (inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)) / (
            intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())
    loss = loss / nei_count

    return -torch.log(loss)
