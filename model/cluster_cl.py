import torch


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, q, mu):

    C = torch.mm(q, mu)
    l1 = torch.mean(-torch.log(torch.sigmoid(torch.sum(torch.mul(C, z1), dim=1))))
    l2 = torch.mean(-torch.log(torch.sigmoid(torch.sum(torch.mul(C, z2), dim=1))))
    loss = (l1 + l2) * 0.5

    return loss


