import sys
sys.path.append(r"C:\Users\asdbe\OneDrive\Documents\GitHub\road-planning-for-slums")

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


if __name__ == '__main__':

    ########################################
    ## Main Code
    ########################################
    ###### Import
    jsonPath = r"C:\Users\asdbe\OneDrive\Documents\GitHub\road-planning-for-slums\JSONInput\env1.json"
    myG,myNodeDict,myEdgeDict = mgh.GraphFromJSON_Customized(jsonPath,scaleTag=False)

    ###### Initialize info
    myG.define_roads_FirstTime()                 # Road
    myG.define_interior_parcels()                # Interior_Parcels


    ###### Compute
    S0 = myG
    # Initial feature and property
    S0.feature_init()

    new_roads_i = mgh.build_all_roads(S0,
                                    myG,
                                    alpha=2,
                                    wholepath=True,
                                    barriers=False,
                                    road_max=1,
                                    plot_intermediate=False,
                                    strict_greedy=True,
                                    vquiet=False,
                                    outsidein=True
                                    )
    S0.plot_roads(parcel_labels=True)




    ##################
    ###### Get Road balances
    ##################


    ##################
    ###### F2F
    ##################
    # PlotF2FDist(S0,vmin = 0,vmax = 5000)
    # optNum = 5
    # for i in range(optNum):  
    #     bisecting_roads = mgh.bisecting_road(S0)
    #     S0.plot_roads(parcel_labels=True)
    #     S0.PlotF2FDist(vmin = 0,vmax = 5000)


    ##################
    ###### POI
    ##################
    optNum = 5
    S0.td_dict_POI_Related_init() 
    S0.PlotF2POIDist()
    for i in range(optNum): 
        S0.td_dict_POI_Related_init() 
        mgh.bisecting_road_forPOI(S0)
        S0.plot_roads(parcel_labels=True)
        S0.PlotF2POIDist()





