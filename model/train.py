import torch
import model.label_cl
import model.cluster_cl
from model.momvfcc import MOMVFCC
from utils.evaluate import cluster_acc, f_score, nmi, ari



class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = MOMVFCC(self.args)
        self.model = self.model.to(self.args.device)


    def __call__(self, A, X, S, R, A2, X2, S2, R2, y_true):


        self.model.train()
        optimize = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        mseloss = torch.nn.MSELoss()
        mseloss = mseloss.to(self.args.device)

        A_A2 = A + A2
        A_A2[A_A2 > 0] = 1
        for epoch in range(self.args.n_epochs):
            H_F, q, H, H2, X_, X_2 = self.model.forward(A, X, A2, X2)
            p = self.target_distribution(q)
            ft_loss = torch.mean((X - X_) ** 2) + torch.mean((X2 - X_2) ** 2)
            ssc_loss = torch.mean((q - p) ** 2)

            y_pred_cl = torch.argmax(q, dim=1)
            cl_loss = model.label_cl.contrastive_loss(H, H2, y_pred_cl, eval(self.args.ADJ))

            # cluster_cl_loss
            cluster_loss = model.cluster_cl.contrastive_loss(H, H2, q, self.model.mu)
            # Total loss
            loss = ft_loss + self.args.lambda_1 * cluster_loss + self.args.lambda_2 * cl_loss + self.args.lambda_3 * ssc_loss

            optimize.zero_grad()
            loss.backward()
            optimize.step()

            y_pred = (torch.argmax(q, dim=1)).cpu().numpy()

            acc = cluster_acc(y_true, y_pred)
            nmi_ = nmi(y_true, y_pred)
            f1 = f_score(y_true, y_pred)
            ari_ = ari(y_true, y_pred)
            print("Epoch--{}:\t\tloss: {:.8f}\t\tacc: {:.8f}\t\tf1: {:.8f}\t\tnmi: {:.8f}\t\tari: {:.8f}".
                  format(epoch, loss, acc, f1, nmi_, ari_))


    def initialization_assign_cluster(self, A, X, A2, X2):
        H_F, _, _, _, _, _ = self.model.forward(A, X, A2, X2)
        self.model.get_assign_cluster_centers_op(H_F)


    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        p = (weight.t() / weight.sum(1)).t()
        return p




