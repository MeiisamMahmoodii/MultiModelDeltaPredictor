import networkx as nx
print("NX Config:", nx)
try:
    print("Top level d_separated:", nx.d_separated)
except AttributeError:
    print("No top level d_separated")

try:
    from networkx.algorithms import d_separation
    print("Module d_separation content:", dir(d_separation))
except ImportError:
    print("No d_separation module")
