from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz, d_separation
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from sklearn.preprocessing import MinMaxScaler

import numpy as np

data = np.loadtxt("StatinRawData.csv", delimiter=",",skiprows=1)
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
cg = pc(data, 0.05, fisherz, node_names=['Age','BMI','Aspirin','StatinUse','Cancer','PSA'])
cg.draw_pydot_graph()

#G, edges = fci(data, fisherz, 0.05, verbose=False)
#pdy = GraphUtils.to_pydot(G)
#pdy.write_png('simple_test.png')
print()