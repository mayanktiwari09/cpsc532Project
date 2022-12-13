from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz, d_separation
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from sklearn.preprocessing import MinMaxScaler
from causallearn.graph.SHD import SHD
import matplotlib.pyplot as plt

import numpy as np

# import pickle as pkl
# import pandas as pd
#
# with open("observations.pkl", "rb") as f:
#     object = pkl.load(f)
#
# df = pd.DataFrame(object)
# df.to_csv(r'CompleteGraphData.csv',index=False)
alpha = 0.001
PCResults = []
GTrue = txt2generalgraph("trueGraph.txt")
sampleNums = range(20,5000,20)
for sampleSize in sampleNums:
    data = np.loadtxt("StatinData5000.csv", delimiter=",",skiprows=1)
    scaler = MinMaxScaler()
    data = data[:sampleSize]
    data = scaler.fit_transform(data)
    cg = pc(data, alpha, fisherz, node_names=['Age','BMI','Aspirin','StatinUse','Cancer','PSA'])
    PCResults.append(SHD(GTrue, cg.G).get_shd())

GTrue = txt2generalgraph("trueGraph2.txt")
FCIResults = []
for sampleSize in sampleNums:
    data = np.loadtxt("StatinData5000.csv", delimiter=",",skiprows=1)
    scaler = MinMaxScaler()
    data = data[:sampleSize]
    data = scaler.fit_transform(data)
    G, edges = fci(data, fisherz, alpha)
    FCIResults.append(SHD(GTrue, G).get_shd())


plt.plot(sampleNums, PCResults)
plt.plot(sampleNums, FCIResults)
plt.show()
print()
# data = np.loadtxt("ToyData.csv", delimiter=",",skiprows=1)
# scaler = MinMaxScaler()
# data = scaler.fit_transform(data)
# cg = pc(data, 0.005, fisherz, node_names=['X','Y','Z'])
# cg.draw_pydot_graph()

# data = np.loadtxt("CompleteGraphData.csv", delimiter=",",skiprows=1)
# scaler = MinMaxScaler()
# data = scaler.fit_transform(data)
# cg = pc(data, 0.0005, fisherz, node_names=['A','B','C','D','E','F','U1','U2','Y'])
# cg.draw_pydot_graph()
