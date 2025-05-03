from generate_data import *
from test_data import *
import networkx as nx
from tulip import tlp
# Gs = generate_evolve_graph(num_nodes=4, p_remove_node=0.5, p_add_node=0.5)
# draw_graph_ensemble(Gs)





if __name__ == "__main__":
    for k in range(ensemblesize):
        # print(i)
        with open('/Users/tushar/Downloads/UofU/courses/spring25/cs6966/gnn/graphs/graph'+str(k), 'rb') as f:
            Gs = pickle.load(f)
        max_len = len(Gs[0])
        G_0 = nx.relabel_nodes(Gs[0], lambda x: f"0_{x}")
        G_temp = G_0.copy()
        G_combined =G_0.copy()
        for i in range(1,len(Gs)):
            G_new = nx.relabel_nodes(Gs[i], lambda x: f"{i}_{x}")
            max_len = max(len(Gs[i]), max_len)
            for j in range(max_len):
                G_combined = nx.compose(G_combined, G_new)
                if f"{i}_{j}" in G_new and f"{i-1}_{j}" in G_combined:
                    G_combined.add_edge(f"{i}_{j}",f"{i-1}_{j}")
            
            G_temp = G_new.copy()

        numnodes = G_combined.number_of_nodes()
        if numnodes >= 250 :
            print(f"oopsie\t{k}\t{numnodes}")
        # nx.nx_agraph.write_dot(G_combined, f"./graph_d/graphd{i}.dot")
        # print(f"saved\t\t{i}")
        # G_final = tulip.newGraph()
        # G_final = tulip.loadGraph(f"./graph_d/graphd{i}.dot")
        # G_final.save(f"./graph_d/graphd{i}.tlp")
        tulip_graph = tlp.newGraph()
        node_id_property = tulip_graph.getStringProperty("label")

        # 3. Mapping from NetworkX node -> Tulip node
        nx_to_tulip_node = {}
        for node in G_combined.nodes():
            tulip_node = tulip_graph.addNode()
            nx_to_tulip_node[node] = tulip_node
            node_id_property[tulip_node] = node


        # 5. Add edges
        for u, v in G_combined.edges():
            tulip_graph.addEdge(nx_to_tulip_node[u], nx_to_tulip_node[v])
        # print(nx_to_tulip_node)
        tlp.saveGraph(tulip_graph,f"./graphdad/graph_{k}.tlpb.gz")
        # print(f"saved\t\t{k}")





# Gs.append(G_combined)

# draw_graph_ensemble(Gs)



