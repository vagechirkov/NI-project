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


def plot_graph_circos(graph, sc_threshold=0.1):
    # Some parts of the code from:
    # https://github.com/multinetlab-amsterdam/network_TDA_tutorial
    G = graph.copy()

    # remove weak connections
    for edge in nx.get_edge_attributes(G, 'weight').items():
        if edge[1] < sc_threshold:
            G.remove_edge(edge[0][0], edge[0][1])

    atlas = atlases.AutomatedAnatomicalParcellation2()
    atlas = aal2_atlas_add_cortical_regions(atlas)
    sublist = {}
    for n, group in enumerate(list(CORTICAL_REGIONS.keys())):
        for i in atlas.names(group=group):
            sublist[i] = group
    G = nx.relabel_nodes(G, lambda x: atlas.names('cortex')[x])

    nx.set_node_attributes(G, sublist, 'cortical_region')

    circ = CircosPlot(
        G, figsize=(15, 15), node_labels=True, node_label_layout='rotation',
        edge_color='weight', edge_width='weight', fontsize=10,
        node_order='cortical_region', nodeprops={"radius": 1},
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

def make_graph():
    pass

def graph_measures():
    pass

