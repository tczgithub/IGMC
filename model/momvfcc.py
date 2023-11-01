import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans


class MOMVFCC(nn.Module):
    def __init__(self, args):
        super(MOMVFCC, self).__init__()
        self.H = None
        self.H2 = None
        self.args = args
        self.lambda_1 = self.args.lambda_1
        self.lambda_2 = self.args.lambda_2
        self.lambda_3 = self.args.lambda_3
        self.n_layers1 = len(self.args.hidden_dims1) - 1
        self.n_layers2 = len(self.args.hidden_dims2) - 1
        self.W11, self.v111, self.v112, self.W12, self.v121, self.v122 = self.define_weights(self.args.hidden_dims1)
        self.C = {}
        self.W21, self.v211, self.v212, self.W22, self.v221, self.v222 = self.define_weights(self.args.hidden_dims2)
        self.C2 = {}

        self.kmeans = KMeans(n_clusters=self.args.cluster, n_init=10, random_state=args.init)
        self.mu = Parameter(torch.FloatTensor(self.args.cluster, self.args.embedding))
        torch.nn.init.xavier_uniform_(self.mu)

        self.n_cluster = self.args.cluster
        self.input_batch_size = self.args.n_sample
        self.alpha = self.args.alpha
        self.gama = nn.Parameter(torch.Tensor(1, ))
        self.gama.data = torch.tensor(0.99999).to(args.device)

    def forward(self, A, X, A2, X2):
        # Encoder1
        H = X
        for layer in range(self.n_layers1):
            H = self.encoder(A, H, 1, layer)
        self.H = H
        # Decoder1
        for layer in range(self.n_layers1 - 1, -1, -1):
            H = self.decoder(H, 1, layer)
        X_ = H

        # Encoder2
        H2 = X2
        for layer in range(self.n_layers2):
            H2 = self.encoder(A2, H2, 2, layer)
        self.H2 = H2

        # Decoder1
        for layer in range(self.n_layers2 - 1, -1, -1):
            H2 = self.decoder(H2, 2, layer)
        X_2 = H2

        H_F = self.H + self.H2 * self.args.beta

        # soft and target distribution
        q = self.soft_assignment(H_F, self.mu)

        return H_F, q, self.H, self.H2, X_, X_2

    def soft_assignment(self, embeddings, cluster_centers):
        q = 1.0 / (1.0 + torch.sum(torch.pow(embeddings.unsqueeze(1) - cluster_centers, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def get_assign_cluster_centers_op(self, H_F):
        print("initialize cluster centroids")
        kmeans = self.kmeans.fit(H_F.cpu().detach().numpy())
        self.mu.data = torch.tensor(kmeans.cluster_centers_).to(self.args.device)

    def define_weights(self, hidden_dims):

        W1 = torch.nn.init.xavier_uniform_(Parameter(torch.FloatTensor(hidden_dims[0], hidden_dims[1])))
        v11 = torch.nn.init.xavier_uniform_(Parameter(torch.FloatTensor(hidden_dims[1], 1)))
        v12 = torch.nn.init.xavier_uniform_(Parameter(torch.FloatTensor(hidden_dims[1], 1)))

        W2 = torch.nn.init.xavier_uniform_(Parameter(torch.FloatTensor(hidden_dims[1], hidden_dims[2])))
        v21 = torch.nn.init.xavier_uniform_(Parameter(torch.FloatTensor(hidden_dims[2], 1)))
        v22 = torch.nn.init.xavier_uniform_(Parameter(torch.FloatTensor(hidden_dims[2], 1)))

        return W1, v11, v12, W2, v21, v22

    def graph_attention_layer(self, A, H, v1, v2):
        f1 = torch.mm(H, v1)
        f1 = torch.mul(A, f1)
        f2 = torch.mm(H, v2)
        f2 = torch.mul(A, torch.t(f2))
        logits = torch.add(f1, f2)

        logits = logits.to_sparse_coo()
        unnormalized_attentions = torch.sparse_coo_tensor(logits.indices(), torch.sigmoid(logits.values()), logits.size())
        attentions = torch.sparse.softmax(unnormalized_attentions, dim=1)
        attentions = attentions.to_dense()

        return attentions

    def encoder(self, A, H, view, layer):
        H = torch.matmul(H, eval("self.W" + str(view) + str(layer + 1)))
        self.C[layer] = self.graph_attention_layer(A, H, eval("self.v" + str(view) + str(layer + 1) + str(1)), eval("self.v" + str(view) + str(layer + 1) + str(2)))
        return torch.mm(self.C[layer], H)

    def decoder(self, H, view, layer):
        H = torch.mm(H, torch.t(eval("self.W" + str(view) + str(layer + 1))))
        output = torch.mm(self.C[layer], H)
        return output


















