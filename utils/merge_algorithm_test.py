from utils.tools import round_up_to_multiple, round_down_to_multiple
import timeit
import matplotlib.pyplot as plt
import networkx as nx
import random

def merge(pairs, line_size=1000):
    num_line = [-1] * line_size
    merged_items = dict()
    connections = nx.Graph()
    for pair in pairs.items():
        connections.add_node(pair[0])
        x = pair[1]
        block = range(x - 2, x + 3)
        overlapping_indices = set()
        for idx in block:
            if not (idx < 0 or idx >= line_size):
                if num_line[idx] != -1:
                    overlapping_indices.add(num_line[idx])
                num_line[idx] = pair[0]

        for index in overlapping_indices:
            connections.add_edge(pair[0], index)
    components=nx.connected_components(connections)
    for component in components:
        values=[]
        for node in component:
            values.append(pairs[node])
        merged_items[tuple(component)]=values
    return merged_items


pairs = dict()
size = 0
t = timeit.Timer(lambda: merge(pairs))
x = []
y= []
for i in range(0, 300):
    pairs=dict()
    for j in range(i, 0, -1):
        pairs[j] = random.randint(0, 50)
    x.append(i)
    y.append(t.timeit(number=75))
    print(i)

pairs = {0:1, 1:3, 2:5, 3:7}
print(merge(pairs))
plt.scatter(x, y)
plt.show()

