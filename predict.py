import os
import json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" # remove INFO logs 


import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from tulip import tlp
import code.custom_graphs as cust
import code.SOTA_layouts as SOTA
from code.models import build_model
import code.graph_preprocessing as gp
import code.spark_preprocessing as sp
import code.metrics as metrics

print("TF ", tf.__version__)
NODE_COLORS=["red", "green", "blue", "yellow","cyan", "lightblue", "magenta", "orange", "slategray", "plum", "sienna", "darkorange", "yellowgreen", "pink", "springgreen", "peru", "gold", "teal" ]


XLIM=(0,0)
YLIM=(0,0)

training_name = "DNN2_implementation_example"
target_epoch = "10" # best epoch is 'max_epoch - patience'
num_news = 0
news=''
for i in range(num_news):
    news+='_new'
model_path = f"./{training_name+news}/models_h5/{training_name}_{target_epoch}.h5"
trainInfos_path = f"./{training_name}/trainInfos.json"

Nmax = -1
dnn2 = SOTA.DNN2(model_path, trainInfos_path, Nmax)
DNN2_title = "(DNN)2" 

infos = {}
with open(trainInfos_path, "r") as fd:
    infos = json.load(fd)

def getLayoutForTimesteps(g, pos, AM):
    num_timesteps = 11

    Gs = [[] for _ in range(num_timesteps)]
    poses = [[] for _ in range(num_timesteps)]
    node_labels = [{} for _ in range(num_timesteps)]
    labels = g.getStringProperty('label')
    n_nodes = g.numberOfNodes()
    node_time_map = [None]*n_nodes
    node_label_map = [None]*n_nodes
    time_node_map = [[] for _ in range(num_timesteps)]

    for node in range(n_nodes):
        node_id = tlp.node(node)
        node_label = labels[node_id]
        node_time = int(labels[node_id].split('_')[0])
        node_time_map[node] = node_time
        node_label_map[node] = node_label
        # print(node_time)
        time_node_map[node_time].append(node)

    for t in range(num_timesteps):
        nodes_t = time_node_map[t]
        # labels_t = node_label_map[t]
        
        num_nodes_t = len(nodes_t)
        am = np.zeros((num_nodes_t, num_nodes_t))

        for i, node in enumerate(nodes_t):
            poses[t].append(pos[node])
            node_labels[t][i] = node_label_map[node]

        for i, nodei in enumerate(nodes_t):
            for j, nodej in enumerate(nodes_t):
                am[i][j] = AM[nodei][nodej]
        
        Gs[t] = am

    x_sum = 0
    y_sum=0
    xmin = np.inf
    ymin = np.inf
    xmax = -np.inf
    ymax = -np.inf
    for pos_ in poses:
        for p in pos_:
            x_sum +=abs(p[0])
            y_sum +=abs(p[1])

            xmax = max(xmax, p[0])
            xmin = min(xmin, p[0])
            ymax = max(ymax, p[1])
            ymin = min(ymin, p[1])

    print(xmax, ymax)
    
    for pos_ in poses:
        for p in pos_:
            p[0] /= x_sum
            p[1] /= y_sum

    XLIM = (xmin/x_sum, xmax/x_sum)
    YLIM = (ymin/y_sum, ymax/y_sum)
    return Gs, poses, node_labels
        


    
def save_graph_ensemble(ens, poses,gname='', layout="DL4GD"):
    for i, en in enumerate(ens):
        plt.figure(figsize=(10, 8))  # Ensure figure size is set (you can adjust)
        numnodes = en.number_of_nodes()
        filename = f"./fin/{gname}/time{i}_layout_{layout}"
        if layout == 'kamada_kawai':
            pos=nx.kamada_kawai_layout(en)
        elif layout == 'spring':
            pos = nx.spring_layout(en)
        elif layout == 'DL4GD':
            pos = poses[i]
        cmap=NODE_COLORS[:numnodes]
        fig, ax = plt.subplots(figsize=(8, 9))  # Create the figure and axis
        nx.draw(en, pos=pos, with_labels=True, node_color=cmap, node_size=500, ax=ax)
        # plt.title(f"Timestep_{i}Layout_{layout}")
        ax.set_title(f"Timestep:{i}\n{layout} layout")
        plt.savefig(filename)
        # plt.close()
        # plt.show()
        plt.clf()


    # plt.tight_layout()
    # plt.show()

def plotNLT2(g, pos, AM, mask, nodeSize=50, title="", ylabel="", ax=None, path = "", fontSize=12, edge_color=(0,0,0,0.2), clustering=False):
    AMs, poses, node_labels = getLayoutForTimesteps(g,pos, AM)
    Gs=[]
    for am in AMs:
        Gs.append(gp.AM2nx(am))
    save_graph_ensemble(Gs, poses, gname)

        


def plotNLT(g, pos, AM,timeidx, mask, nodeSize=50, title="", ylabel="", ax=None, path = "", fontSize=12, edge_color=(0,0,0,0.2), clustering=False):
    Gs, poses, node_labels = getLayoutForTimesteps(g,pos, AM)
    AM = Gs[timeidx]
    G = gp.AM2nx(AM)
    pos = poses[timeidx]
    pos = np.array(pos)
    colors = "C0"
    # if(clustering):
    #     tlpg, id_mapping = gp.AM2tlp(AM, mask)
    #     mcl_prop = gp.applyTlpAlgorithm(tlpg, "MCL Clustering", "res_mcl")
    #     colors=[]
    #     tlpg_nodes = []
    #     for n in tlpg.getNodes():
    #         tlpg_nodes.append(n)
    #     for n in G.nodes():
    #         colors.append("C"+str(int(mcl_prop.getNodeValue(tlpg_nodes[n]))))
    nodeSize=20
    nx.draw(G, pos, node_size=nodeSize, labels=node_labels[timeidx], ax=ax, node_color=colors, edge_color=edge_color, width=1)
    #For drawing seperately

    if(ax is None):
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()
    else:
        ax.set_title(title)
        ax.set_axis_on()
        ax.spines.right.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.spines.left.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.set_ylabel(ylabel)
    
    # First Axes / Image
    fig1, ax1 = plt.subplots(figsize=(8, 6))  # 8x6 inches figure size
        # Set fixed range for the x and y axes
    # ax1.set_xlim(XLIM)  # Fixed range for x-axis from 0 to 10
    # ax1.set_ylim(YLIM)  # Fixed range for y-axis from -1.5 to 1.5
    nx.draw(G, pos, labels=node_labels[timeidx], node_color='lightblue', ax=ax1)
    fig1.savefig(f"./{training_name+news}/" + path +'/'+str(title), bbox_inches='tight', dpi=300)
    # plt.show()

    fig1.clf()


graphs_to_draw = [
    cust.time_vary_graph("./graph_validation/graph_7.tlpb.gz"),
    cust.time_vary_graph("./graph_daddy/graph_230.tlpb.gz"),
]

timesteps = 11

one_graph_drawing_width = 4
one_graph_drawing_height = 3
# plt.rcParams["figure.figsize"] = (one_graph_drawing_width * (1+timesteps), one_graph_drawing_height * len(graphs_to_draw))
# fig, axes = plt.subplots(nrows = len(graphs_to_draw), ncols=timesteps)

monitor = {}
for g_idx, (g, gname) in enumerate(graphs_to_draw):
    # plt.rcParams["figure.figsize"] = (one_graph_drawing_width * timesteps, one_graph_drawing_height)
    # fig, axes = plt.subplots(nrows = 1, ncols=timesteps)
    N = g.numberOfNodes()
    AM = gp.graph2AM(g)
    DM = gp.AM2DM(AM,g)
    mask = np.ones((N, 1))   
    dnn2_pos = dnn2.layout(g)
    title = DNN2_title 

    # if not os.path.exists(f"./{training_name+news}/" + gname):
    #     os.mkdir(f"./{training_name+news}/" + gname) 
    
    if not os.path.exists(f"./fin/{gname}"):
        os.mkdir(f"./fin/{gname}") 
    
     

    for i_od in range(timesteps):
        title = i_od
        # plotNLT2(g, dnn2_pos, AM,i_od, mask, title=title, ax=axes[i_od], path = gname)
    plotNLT2(g, dnn2_pos, AM, mask, title=title, path = gname)

    # fig.savefig(f"./{training_name+news}/" + gname +'/final', bbox_inches='tight', dpi=300)

    



