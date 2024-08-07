import sys
import os
cwd = os.getcwd()
sys.path.append(cwd) 

sys.path.append("/Users/chenzebin/Documents/GitHub/road-planning-for-slums") 
    
import json
import numpy as np
import pretreatment.my_graph as mg
import pretreatment.my_graph_helpers as mgh
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import networkx as nx
import itertools
import operator

import copy
import inspect
import time

import DebugAndPreview as DP
########################################
## Main Code
########################################
###### Import
#jsonPath = ("/Users/chenzebin/Documents/GitHub/road-planning-for-slums/JSONInput/punggol_1_withShortCut.json") 
#jsonPath = r"C:\Users\asdbe\OneDrive\Documents\GitHub\road-planning-for-slums\JSONInput\env1.json"
jsonPath = ("/Users/chenzebin/Documents/GitHub/road-planning-for-slums/JSONInput/env1.json") 
myG,myNodeDict,myEdgeDict = mgh.GraphFromJSON_Customized(jsonPath,scaleTag=True,new_min = 0,new_max = 5)   # 5

# print ("first road step",  len(myG.myedges()) - len(myG.road_edges))
# ###### Initialize info
# myG.define_roads_FirstTime()                 # Road
# myG.define_interior_parcels()                # Interior_Parcels

myG.plot_roads(parcel_labels=True)


print ("first road step_after",  len(myG.myedges()) - len(myG.road_edges))
plt.show()



