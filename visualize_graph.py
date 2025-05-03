import tulip as tlp
import networkx as nx

# Load or access your Tulip graph
tlp_graph = tlp.loadGraph("your_graph.tlpb")  # or however you created it

# Assume you have a layout already (e.g., force-directed, treemap, etc.)
layout_prop = tlp_graph.getLayoutProperty("viewLayout")  # or any other layout name

