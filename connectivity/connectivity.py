import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from nxviz import CircosPlot

from neurolib.utils import atlases


# https://doi.org/10.1016/j.neuroimage.2015.07.075 Table 2
# number corresponds to AAL2 labels indices
CORTICAL_REGIONS = {
    'central_region': [1, 2, 61, 62, 13, 14],
    'frontal_lobe': {
        'Lateral surface': [3, 4, 5, 6, 7, 8, 9, 10],
        'Medial surface': [19, 20, 15, 16, 73, 74],
        'Orbital surface': [11, 12, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28,
                            29, 30, 31, 32]
        },
    'temporal_lobe': {
        'Lateral surface': [83, 84, 85, 86, 89, 90, 93, 94]
        },
    'parietal_lobe': {
        'Lateral surface': [63, 64, 65, 66, 67, 68, 69, 70],
        'Medial surface': [71, 72],
        },
    'occipital_lobe': {
        'Lateral surface': [53, 54, 55, 56, 57, 58],
        'Medial and inferior surfaces': [47, 48, 49, 50, 51, 52, 59, 60],
        },
    'limbic_lobe': [87, 88, 91, 92, 35, 36, 37, 38, 39, 40, 33, 34]
    }


def aal2_atlas_add_cortical_regions(aal2_atlas):
    """Add groups of cortical regions.

    Parameters
    ----------
    atlas : neurolib.utils.atlases.AutomatedAnatomicalParcellation2()
        AAL2 atlas
    """
    for i in CORTICAL_REGIONS.items():
        inx = []
        if not isinstance(i[1], list):
            for ii in i[1].items():
                inx.append(ii[1])
            inx = sum(inx, [])
        else:
            inx = i[1]

        # reindexing from 1 to 0
        inx = [i-1 for i in inx]
        setattr(aal2_atlas, i[0], inx)
    return aal2_atlas


def plot_graph_circos(graph, sc_threshold=0.07):
    # Some parts of the code from:
    # https://github.com/multinetlab-amsterdam/network_TDA_tutorial
    G = graph.copy()

    # remove weak connections
    for edge in nx.get_edge_attributes(G, 'weight').items():
        if edge[1] < sc_threshold:
            G.remove_edge(edge[0][0], edge[0][1])

    atlas = atlases.AutomatedAnatomicalParcellation2()
    atlas = aal2_atlas_add_cortical_regions(atlas)
    G = nx.relabel_nodes(G, lambda x: atlas.names('cortex')[x])
    sublist = {}
    order = {}
    n = 0
    for group in list(CORTICAL_REGIONS.keys()):
        for i in atlas.names(group=group):
            sublist[i] = group
            if i[-1] == 'L':
                order[i] = n
            else:
                order[i] = n + 1
        n += 2

    nx.set_node_attributes(G, sublist, 'cortical_region')
    nx.set_node_attributes(G, order, 'node_order')
    # https://nxviz.readthedocs.io/en/latest/modules.html
    circ = CircosPlot(
        G, figsize=(15, 15), node_labels=True, node_label_layout='rotation',
        edge_color='weight', edge_width='weight', fontsize=10,
        node_order='node_order', nodeprops={"radius": 1},
        group_label_offset=5, node_color='cortical_region', group_legend=True
        )

    circ.draw()
    circ.sm.colorbar.remove()
    labels_networks = sorted(list(set(
        [list(circ.graph.nodes.values())[n][
            circ.node_color] for n in np.arange(len(circ.nodes))])))

    plt.legend(handles=circ.legend_handles,
               title="Subnetwork",
               ncol=2,
               borderpad=1,
               shadow=True,
               fancybox=True,
               bbox_to_anchor=(0.8, 1.05),
               loc='upper left',
               fontsize=10,
               labels=labels_networks)
    plt.tight_layout()
    return circ


def make_graph(Cmat):
    G = nx.from_numpy_matrix(sparsify(make_symmetric(Cmat), threshold=0.01))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G.graph['matrix'] = Cmat
    return G


def graph_measures(G, Dmat=None):

    graph_measures = {}

    # -------------- Degree -------------- #
    strength = G.degree(weight='weight')
    nx.set_node_attributes(G, dict(strength), 'strength')

    # Normalized node strength values 1/N-1
    normstrenghts = {node: val * 1/(len(G.nodes)-1)
                     for (node, val) in strength}
    nx.set_node_attributes(G, normstrenghts, 'strengthnorm')

    # Computing the mean degree of the network
    normstrengthlist = np.array([val * 1/(len(G.nodes)-1)
                                 for (node, val) in strength])
    mean_degree = np.sum(normstrengthlist)/len(G.nodes)

    graph_measures['mean_degree'] = mean_degree
    graph_measures['degree'] = normstrengthlist

    # -------------- Centrality -------------- #
    # Closeness Centrality
    # Distance is an inverse of correlation
    # IDEA: use Dmat instead of 1 / abs(weight) ???
    if isinstance(Dmat, np.ndarray):
        G_distance_dict = {(e1, e2): Dmat[e1, e2]
                           for e1, e2 in G.edges()}
    else:
        G_distance_dict = {(e1, e2): 1 / abs(weight)
                           for e1, e2, weight in G.edges(data='weight')}

    nx.set_edge_attributes(G, G_distance_dict, 'distance')
    closeness = nx.closeness_centrality(G, distance='distance')
    nx.set_node_attributes(G, closeness, 'closecent')
    graph_measures['closeness'] = list(closeness.values())

    # Betweenness Centrality
    betweenness = nx.betweenness_centrality(G, weight='distance',
                                            normalized=True)
    nx.set_node_attributes(G, betweenness, 'betweenness_centrality')
    graph_measures['betweenness'] = list(betweenness.values())

    # Eigenvector Centrality
    # eigen = nx.eigenvector_centrality(G, weight='weight')
    # nx.set_node_attributes(G, eigen, 'eigen')
    # graph_measures['eigenvector_centrality'] = list(eigen.values())

    # -------------- Path Length -------------- #
    # Average shortest path length
    avg_shorterst_path = nx.average_shortest_path_length(G, weight='distance')
    graph_measures['mean_shortest_path'] = avg_shorterst_path

    # TODO: maybe add more measures

    # -------------- Assortativity -------------- #
    # Average degree of the neighborhood
    average_neighbor_degree = nx.average_neighbor_degree(G, weight='weight')
    nx.set_node_attributes(G, average_neighbor_degree, 'neighbor_degree')
    graph_measures['neighbor_degree'] = list(average_neighbor_degree.values())

    # ------- Avg Neighbor Degree (2nd try) ----- #
    avg_nb_deg = compute_avg_neighborhood_degree(G.graph['matrix'])
    nx.set_node_attributes(G, avg_nb_deg, 'neighbor_degree_new')
    graph_measures['neighbor_degree_new'] = list(avg_nb_deg)

    # -------------- Clustering Coefficient -------------- #
    clustering = nx.clustering(G, weight='weight')
    nx.set_node_attributes(G, clustering, 'clustering_coefficient')
    graph_measures['clustering_coefficient'] = list(clustering.values())
    graph_measures['mean_clustering_coefficient'] =\
        nx.average_clustering(G, weight='weight')

    # -------------- Minimum Spanning Tree -------------- #
    # backbone of a network
    GMST = nx.minimum_spanning_tree(G, weight='distance')
    backbone = nx.to_numpy_array(GMST)
    graph_measures['backbone'] = backbone

    # -------------- Small-world -------------- #
    # FIXME: too slow...
    # graph_measures['omega'] = nx.omega(G, seed=0, niter=1)
    # graph_measures['sigma'] = nx.sigma(G, seed=0, niter=1)

    return G, graph_measures


def neighborhoods(adj, epsilon=0.001):
    N = np.array(adj > epsilon, dtype=int)
    return N


def normalize_neighborhoods(neighborhoods):
    N = neighborhoods / np.sum(neighborhoods, axis=0)
    return N


def is_symmetric(A, epsilon=0.1):
    diff = A - A.T
    asymmetry = np.sum(np.abs(diff))
    return (asymmetry < epsilon, asymmetry,
            np.max(np.abs(diff)), np.mean(np.abs(diff)))


def make_symmetric(A):
    S = (A + A.T)/2
    assert(is_symmetric(S))
    return S


def sparsify(A, threshold=0.01):
    A[A < threshold] = 0.0
    return A


def compute_avg_neighborhood_degree(A):
    degrees = np.sum(A, axis=0)
    N = neighborhoods(A)
    N = normalize_neighborhoods(N)
    avg_nb_deg = N.dot(degrees)

    # not always true - should it?
    # assert(np.mean(degrees) == np.mean(avg_nb_deg))
    return avg_nb_deg


def z_scores(df):
    # 'degree', 'clustering_coefficient'
    m_dist = ['closeness', 'betweenness',
              'neighbor_degree', 'neighbor_degree_new']
    m_point = ['mean_degree', 'mean_shortest_path',
               'mean_clustering_coefficient']  # , 'omega', 'sigma']

    n_subjects = df.shape[0]
    df.loc[:, 'subject'] = df.index
    for m in m_dist:
        value = np.array([df.loc[i, m] for i in range(n_subjects)])
        mean = value.mean(axis=1)
        std = value.std(axis=1).mean()
        df.loc[:, f'mean_{m}_z'] = (mean - mean.mean()) / std

    for m in m_point:
        mean = df.loc[:, m].mean()
        std = df.loc[:, m].std()
        df.loc[:, f'{m}_z'] = (df.loc[:, m] - mean) / std

    all_values = (['subject'] + [f'{m}_z' for m in m_point]
                  + [f'mean_{m}_z' for m in m_dist])
    return df.loc[:, all_values]


def similarity_between_subjects(df):
    m_dist = ['degree', 'closeness', 'betweenness',
              'neighbor_degree', 'neighbor_degree_new',
              'clustering_coefficient']
    mats = ['Cmat', 'Dmat', 'backbone']
    # graph_properties = ['omega', 'sigma']

    n_subjects = df.shape[0]
    df.loc[:, 'subject'] = df.index
    for m in m_dist:
        value = np.array([df.loc[i, m] for i in range(n_subjects)])
        for s in range(n_subjects):
            corr = np.corrcoef(value[0, :], value[s, :])[0, 1]
            df.loc[s, f'{m}_corr'] = corr
    for m in mats:
        for s in range(n_subjects):
            corr = np.corrcoef(df.loc[0, m].flatten(),
                               df.loc[s, m].flatten())[0, 1]
            df.loc[s, f'{m}_corr'] = corr
    # for m in graph_properties:
    #    value = np.array([df.loc[i, m] for i in range(n_subjects)])
    #    for s in range(n_subjects):
    #        corr = np.corrcoef(value[0], value[s])[0, 1]
    #        df.loc[s, f'{m}_corr'] = corr

    # + graph_properties]
    all_values = ['subject'] + [f'{m}_corr' for m in mats + m_dist]
    return df.loc[:, all_values]


if __name__ == '__main__':
    from neurolib.utils.loadData import Dataset
    ds = Dataset("gw")
    G = make_graph(ds.Cmats[1])
