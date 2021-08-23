from testing.tester import run_test
from hyperopt import hp, fmin
def objective(args):
    _, total_precision, total_recall = run_test()
    f1 = 2*total_precision*total_recall/(total_precision+total_recall)
    return 1-f1
space = {'diagram_graph_builder_dbscan_eps':hp.uniform('diagram_graph_builder_dbscan_eps', 0.01, 0.1)}
objective(space)

