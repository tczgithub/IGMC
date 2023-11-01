import os

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio
import sys
from scipy import sparse


def prepare_graph_data(adj):
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    # print("prepare_adj:", adj)
    # data =  adj.tocoo().data
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    # print("tocoo_adj:", adj)
    adj = adj.astype(np.float32)
    # print("astype_adj:", adj)
    indices = np.vstack((adj.col, adj.row)).transpose()
    # print("indices", indices)
    # print(adj.row)
    # print(adj.col)
    return (indices, adj.data, adj.shape), adj.row, adj.col
    # return (indices, adj.data, adj.shape), adj.row, adj.col, data#adj.data


def prepare_graph_data1(adj):
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    # data =  adj.tocoo().data
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape), adj.row, adj.col
    # return (indices, adj.data, adj.shape), adj.row, adj.col, data#adj.data


def prepare_sparse_features(features):
    if not sp.isspmatrix_coo(features):
        features = sparse.csc_matrix(features).tocoo()
        features = features.astype(np.float32)
    indices = np.vstack((features.row, features.col)).transpose()
    return (indices, features.data, features.shape)


def conver_sparse_tf2np(input):
    # Convert Tensorflow sparse matrix to Numpy sparse matrix
    return [sp.coo_matrix((input[layer][1], (input[layer][0][:, 0], input[layer][0][:, 1])),
                          shape=(input[layer][2][0], input[layer][2][1])) for layer in input]


###############################################
# This section of code adapted from tkipf/gcn #
###############################################


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)



def load_multiattribute(dataset_str):  # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    nx_graph = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(nx_graph)
    edges = nx_graph.edges()

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]



    return sp.coo_matrix(adj), features.todense(), labels, 0, 0, 0




def load_multigraph(dataset):

    if dataset == 'acm':
        data = sio.loadmat('data/acm/acm.mat')
        # feature
        feature = data['feature']
        features = sp.csr_matrix(feature, dtype=np.float32)

        labels = data['label']
        num_nodes = data['label'].shape[0]

        data['PAP'] = sparse.coo_matrix(data['PAP'] + np.eye(num_nodes))
        data['PAP'] = data['PAP'].todense()
        data['PAP'][data['PAP'] > 0] = 1.0
        adj1 = sparse.coo_matrix(data['PAP'] - np.eye(num_nodes))
        data['PLP'] = sparse.coo_matrix(data['PLP'] + np.eye(num_nodes))
        data['PLP'] = data['PLP'].todense()
        data['PLP'][data['PLP'] > 0] = 1.0
        adj2 = sparse.coo_matrix(data['PLP'] - np.eye(num_nodes))

        PAP = np.stack((np.array(adj1.row), np.array(adj1.col)), axis=1)
        PLP = np.stack((np.array(adj2.row), np.array(adj2.col)), axis=1)
        # print(PAP)
        # print(PLP)

        #
        PAPedges = np.array(list(PAP), dtype=np.int32).reshape(PAP.shape)
        PAP_adj = sp.coo_matrix((np.ones(PAPedges.shape[0]), (PAPedges[:, 0], PAPedges[:, 1])), shape=(num_nodes, num_nodes), dtype=np.float32)
        PAP_adj = PAP_adj + PAP_adj.T.multiply(PAP_adj.T > PAP_adj) - PAP_adj.multiply(PAP_adj.T > PAP_adj)
        PAP_normalize_adj = normalize(PAP_adj)
        # print(PAP_normalize_adj)

        PLPedges = np.array(list(PLP), dtype=np.int32).reshape(PLP.shape)
        PLP_adj = sp.coo_matrix((np.ones(PLPedges.shape[0]), (PLPedges[:, 0], PLPedges[:, 1])),
                                shape=(num_nodes, num_nodes), dtype=np.float32)
        PLP_adj = PLP_adj + PLP_adj.T.multiply(PLP_adj.T > PLP_adj) - PLP_adj.multiply(PLP_adj.T > PLP_adj)
        PLP_normalize_adj = normalize(PLP_adj)
        # print(PLP_normalize_adj)

        adj_list = [PAP_normalize_adj, PLP_normalize_adj]

        idx_train = data['train_idx'].ravel()
        idx_val = data['val_idx'].ravel()
        idx_test = data['test_idx'].ravel()

        return adj_list, features.todense(), labels, idx_train, idx_val, idx_test


    if dataset == 'imdb':
        data = sio.loadmat('data/imdb/imdb.mat')
        # feature
        feature = data['feature']
        features = sp.csr_matrix(feature, dtype=np.float32)

        labels = data['label']
        num_nodes = data['label'].shape[0]

        data['MAM'] = sparse.coo_matrix(data['MAM'] + np.eye(num_nodes))
        data['MAM'] = data['MAM'].todense()
        data['MAM'][data['MAM'] > 0] = 1.0
        adj1 = sparse.coo_matrix(data['MAM'] - np.eye(num_nodes))
        data['MDM'] = sparse.coo_matrix(data['MDM'] + np.eye(num_nodes))
        data['MDM'] = data['MDM'].todense()
        data['MDM'][data['MDM'] > 0] = 1.0
        adj2 = sparse.coo_matrix(data['MDM'] - np.eye(num_nodes))

        MAM = np.stack((np.array(adj1.row), np.array(adj1.col)), axis=1)
        MDM = np.stack((np.array(adj2.row), np.array(adj2.col)), axis=1)


        #
        MAMedges = np.array(list(MAM), dtype=np.int32).reshape(MAM.shape)
        MAM_adj = sp.coo_matrix((np.ones(MAMedges.shape[0]), (MAMedges[:, 0], MAMedges[:, 1])), shape=(num_nodes, num_nodes), dtype=np.float32)
        MAM_adj = MAM_adj + MAM_adj.T.multiply(MAM_adj.T > MAM_adj) - MAM_adj.multiply(MAM_adj.T > MAM_adj)
        MAM_normalize_adj = normalize(MAM_adj)


        MDMedges = np.array(list(MDM), dtype=np.int32).reshape(MDM.shape)
        MDM_adj = sp.coo_matrix((np.ones(MDMedges.shape[0]), (MDMedges[:, 0], MDMedges[:, 1])), shape=(num_nodes, num_nodes), dtype=np.float32)
        MDM_adj = MDM_adj + MDM_adj.T.multiply(MDM_adj.T > MDM_adj) - MDM_adj.multiply(MDM_adj.T > MDM_adj)
        MDM_normalize_adj = normalize(MDM_adj)
        # print(PLP_normalize_adj)

        adj_list = [MAM_normalize_adj, MDM_normalize_adj]

        idx_train = data['train_idx'].ravel()
        idx_val = data['val_idx'].ravel()
        idx_test = data['test_idx'].ravel()

        return adj_list, features.todense(), labels, idx_train, idx_val, idx_test

    if dataset == 'dblp':
        data = sio.loadmat('data/dblp/dblp.mat')
        # feature
        feature = data['features']
        features = sp.csr_matrix(feature, dtype=np.float32)

        labels = data['label']
        num_nodes = data['label'].shape[0]

        data['net_APA'] = sparse.coo_matrix(data['net_APA'] + np.eye(num_nodes))
        data['net_APA'] = data['net_APA'].todense()
        data['net_APA'][data['net_APA'] > 0] = 1.0
        adj1 = sparse.coo_matrix(data['net_APA'] - np.eye(num_nodes))

        data['net_APCPA'] = sparse.coo_matrix(data['net_APCPA'] + np.eye(num_nodes))
        data['net_APCPA'] = data['net_APCPA'].todense()
        data['net_APCPA'][data['net_APCPA'] > 0] = 1.0
        adj2 = sparse.coo_matrix(data['net_APCPA'] - np.eye(num_nodes))


        net_APA = np.stack((np.array(adj1.row), np.array(adj1.col)), axis=1)
        net_APCPA = np.stack((np.array(adj2.row), np.array(adj2.col)), axis=1)

        net_APAedges = np.array(list(net_APA), dtype=np.int32).reshape(net_APA.shape)
        net_APA_adj = sp.coo_matrix((np.ones(net_APAedges.shape[0]), (net_APAedges[:, 0], net_APAedges[:, 1])), shape=(num_nodes, num_nodes), dtype=np.float32)
        net_APA_adj = net_APA_adj + net_APA_adj.T.multiply(net_APA_adj.T > net_APA_adj) - net_APA_adj.multiply(net_APA_adj.T > net_APA_adj)
        net_APA_normalize_adj = normalize(net_APA_adj)

        net_APCPAedges = np.array(list(net_APCPA), dtype=np.int32).reshape(net_APCPA.shape)
        net_APCPA_adj = sp.coo_matrix((np.ones(net_APCPAedges.shape[0]), (net_APCPAedges[:, 0], net_APCPAedges[:, 1])), shape=(num_nodes, num_nodes), dtype=np.float32)
        net_APCPA_adj = net_APCPA_adj + net_APCPA_adj.T.multiply(net_APCPA_adj.T > net_APCPA_adj) - net_APCPA_adj.multiply(net_APCPA_adj.T > net_APCPA_adj)
        net_APCPA_normalize_adj = normalize(net_APCPA_adj)


        adj_list = [net_APA_normalize_adj, net_APCPA_normalize_adj]

        idx_train = data['train_idx'].ravel()
        idx_val = data['val_idx'].ravel()
        idx_test = data['test_idx'].ravel()

        return adj_list, features.todense(), labels, idx_train, idx_val, idx_test

def create_subgraph(sparse_graph, _sentinel=None, nodes_to_remove=None, nodes_to_keep=None):
    """Create a graph with the specified subset of nodes.

    Exactly one of (nodes_to_remove, nodes_to_keep) should be provided, while the other stays None.
    Note that to avoid confusion, it is required to pass node indices as named arguments to this function.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    _sentinel : None
        Internal, to prevent passing positional arguments. Do not use.
    nodes_to_remove : array-like of int
        Indices of nodes that have to removed.
    nodes_to_keep : array-like of int
        Indices of nodes that have to be kept.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph with specified nodes removed.

    """
    # Check that arguments are passed correctly
    if _sentinel is not None:
        raise ValueError("Only call `create_subgraph` with named arguments',"
                         " (nodes_to_remove=...) or (nodes_to_keep=...)")
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError("Only one of nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None:
        nodes_to_keep = [i for i in range(sparse_graph.num_nodes()) if i not in nodes_to_remove]
    elif nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    sparse_graph.adj_matrix = sparse_graph.adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if sparse_graph.attr_matrix is not None:
        sparse_graph.attr_matrix = sparse_graph.attr_matrix[nodes_to_keep]
    if sparse_graph.labels is not None:
        sparse_graph.labels = sparse_graph.labels[nodes_to_keep]
    if sparse_graph.node_names is not None:
        sparse_graph.node_names = sparse_graph.node_names[nodes_to_keep]
    return sparse_graph

def largest_connected_components(sparse_graph, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
            idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)

class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form.

    """

    def __init__(self, adj_matrix, attr_matrix=None, labels=None,
        node_names=None, attr_names=None, class_names=None, metadata=None):
        """Create an attributed graph.

        Parameters
        ----------
        adj_matrix : sp.csr_matrix, shape [num_nodes, num_nodes]
            Adjacency matrix in CSR format.
        attr_matrix : sp.csr_matrix or np.ndarray, shape [num_nodes, num_attr], optional
            Attribute matrix in CSR or numpy format.
        labels : np.ndarray, shape [num_nodes], optional
            Array, where each entry represents respective node's label(s).
        node_names : np.ndarray, shape [num_nodes], optional
            Names of nodes (as strings).
        attr_names : np.ndarray, shape [num_attr]
            Names of the attributes (as strings).
        class_names : np.ndarray, shape [num_classes], optional
            Names of the class labels (as strings).
        metadata : object
            Additional metadata such as text.

        """
        # Make sure that the dimensions of matrices / arrays all agree
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)"
                             .format(type(adj_matrix)))

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError("Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)"
                                 .format(type(attr_matrix)))

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency and attribute matrices don't agree")

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the label vector don't agree")

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the node names don't agree")

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError("Dimensions of the attribute matrix and the attribute names don't agree")

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

    def num_nodes(self):
        """Get the number of nodes in the graph."""
        return self.adj_matrix.shape[0]

    def num_edges(self):
        """Get the number of edges in the graph.

        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        if self.is_directed():
            return int(self.adj_matrix.nnz)
        else:
            return int(self.adj_matrix.nnz / 2)

    def get_neighbors(self, idx):
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx : int
            Index of the node whose neighbors are of interest.

        """
        return self.adj_matrix[idx].indices

    def is_directed(self):
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self):
        """Convert to an undirected graph (make adjacency matrix symmetric)."""
        if self.is_weighted():
            raise ValueError("Convert to unweighted graph first.")
        else:
            self.adj_matrix = self.adj_matrix + self.adj_matrix.T
            self.adj_matrix[self.adj_matrix != 0] = 1
        return self

    def is_weighted(self):
        """Check if the graph is weighted (edge weights other than 1)."""
        return np.any(np.unique(self.adj_matrix[self.adj_matrix != 0].A1) != 1)

    def to_unweighted(self):
        """Convert to an unweighted graph (set all edge weights to 1)."""
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    # Quality of life (shortcuts)
    def standardize(self):
        """Select the LCC of the unweighted/undirected/no-self-loop graph.

        All changes are done inplace.

        """
        G = self.to_unweighted().to_undirected()
        G = eliminate_self_loops(G)
        G = largest_connected_components(G, 1)
        return G

    def unpack(self):
        """Return the (A, X, z) triplet."""
        return self.adj_matrix, self.attr_matrix, self.labels

def eliminate_self_loops1(A):
    """Remove self-loops from the adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A

def eliminate_self_loops(G):
    G.adj_matrix = eliminate_self_loops1(G.adj_matrix)
    return G

def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.

    """
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return SparseGraph(adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata)

def load_dataset(data_path):
    """Load a dataset.

    Parameters
    ----------
    name : str
        Name of the dataset to load.

    Returns
    -------
    sparse_graph : SparseGraph
        The requested dataset in sparse format.

    """
    if not data_path.endswith('.npz'):
        data_path += '.npz'
    if os.path.isfile(data_path):
        return load_npz_to_sparse_graph(data_path)
    else:
        raise ValueError(f"{data_path} doesn't exist.")




def load_data(dataset):
    if dataset == 'acm' or dataset == 'imdb' or dataset == 'dblp_334':
        return load_multigraph(dataset)
    else:
        return load_multiattribute(dataset)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack(
        (adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
    # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
    return indices, adj.data, adj.shape


def normalize(mx):
    """Row-normalize sparse matrix"""
    epsilon = 1e-5
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


