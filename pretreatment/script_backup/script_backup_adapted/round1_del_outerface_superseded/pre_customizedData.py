import sys
sys.path.append(r"C:\Users\asdbe\OneDrive\Documents\GitHub\road-planning-for-slums")

import json
import numpy as np
import pretreatment.my_graph as mg
import pretreatment.my_graph_helpers as mgh
import matplotlib.pyplot as plt
import networkx as nx

def LoadJSONData(jsonPath):
    with open(jsonPath, 'r') as file:
        data = json.load(file)
        ptCoordDict = data["ptCoordDict"]
        edgeDict = data["edgeDict"]
        parcelEdgeIDs = data["parcelEdgeIDs"]

    return ptCoordDict,edgeDict,parcelEdgeIDs

def GraphFromJSON_Customized(ptCoordDict,edgeDict,parcelEdgeIDs,name="None",rezero=np.array([0, 0])):

    ### Build the graph
    myG = mg.MyGraph(name="name")
    rezero=np.array([0, 0])

    ### Build the myNodeDict
    myNodeDict = dict()
    for nID in ptCoordDict.keys():
        coords =  ptCoordDict[nID]
        coords = coords - rezero
        myN = mg.MyNode(coords)
        myNodeDict[int(nID)] = myN


    ### Build the myEdgeDict
    myEdgeDict = dict()
    for edgeID in edgeDict.keys():
        thisEdge = edgeDict[str(edgeID)]
        startNode = myNodeDict[thisEdge["start"]]
        endNode = myNodeDict[thisEdge["end"]]   
        myEdge = mg.MyEdge((startNode,endNode))
        myEdgeDict[edgeID] = myEdge
        myG.add_edge(myEdge)

    ### Inner face 
    inner_facelist = []
    for edgeOrder in parcelEdgeIDs:
        edges = []
        for edgeID in edgeOrder:
            edges.append(myEdgeDict[str(edgeID)])
        
        inner_facelist.append(mg.MyFace(edges))

    myG.inner_facelist = inner_facelist


    ### Add edge property
    for edgeID in myEdgeDict:
        myEdgeDict[edgeID].external = edgeDict[str(edgeID)]["external"]
        myEdgeDict[edgeID].internal = edgeDict[str(edgeID)]["internal"]
        myEdgeDict[edgeID].onBoundary = edgeDict[str(edgeID)]["onBoundary"]
        myEdgeDict[edgeID].isRoad = edgeDict[str(edgeID)]["isRoad"]
        myEdgeDict[edgeID].isConstraint = edgeDict[str(edgeID)]["isConstraint"]
        myEdgeDict[edgeID].isPOI = edgeDict[str(edgeID)]["isPOI"]


    return myG,myNodeDict,myEdgeDict



def PlotMyGraph(myG,showInternal = False):
    plt.figure(figsize=(10, 10))
    plt.axes().set_aspect(aspect=1)
    plt.axis('off')

    nlocs = myG.location_dict()

    ### basic layer
    nx.draw_networkx_edges(myG.G, nlocs, edge_color='grey', width=1)
    nx.draw_networkx_nodes(myG.G, nlocs, node_color='black', node_size=5)

    ### info layer
    edge_colors = []
    edges_to_draw = []
    for edge in myG.myedges():
        if edge.isRoad and edge.isPOI!=True:
            edge_colors.append('orange')
            edges_to_draw.append((edge.nodes[0], edge.nodes[1]))
        elif edge.isConstraint:
            edge_colors.append('red')
            edges_to_draw.append((edge.nodes[0], edge.nodes[1]))
        elif edge.isPOI:
            edge_colors.append('purple')
            edges_to_draw.append((edge.nodes[0], edge.nodes[1]))
        else:
            pass

    nx.draw_networkx_edges(myG.G, nlocs,edgelist=edges_to_draw, edge_color=edge_colors, width=1)
    

    edge_colors = []
    edges_to_draw = []

    ### internal layer  
    if showInternal == True:
        if hasattr(myG, 'interior_parcels'):
            for face in myG.interior_parcels:
                for edge in face.edges:
                    edge_colors.append('red')
                    edges_to_draw.append((edge.nodes[0], edge.nodes[1]))
            nx.draw_networkx_edges(myG.G, nlocs,edgelist=edges_to_draw, edge_color=edge_colors, width=1)

    plt.show()

##########
## Preview Function
##########
def PrintInnerFaceGeo(myG):
    allEdgeNodes = []
    for face in myG.inner_facelist:
        for edge in face.edges:
            allEdgeNodes.append(edge.nodes)
    print (allEdgeNodes)

def define_roads_newMethod(self):
    """ finds which edges and nodes in the connected component are on
    the roads, and updates thier properties (node.road, edge.road) """

    self.node_list = []
    for n in self.G.nodes():
        self.node_list.append(n)
    self.edge_list = self.myedges()
    self.stage2edges = []
    self.full_connected_road_num = 0

    self.f2f_data=[]
    self.cost_data=[]
    self.parcels_data=[]

    road_nodes = []
    road_nodes_idx = []
    road_edges = []
            
    # check for empty graph
    if self.G.number_of_nodes() < 2:
        return  ################

    for e in myG.myedges():
        if e.isRoad == True:
            e.road = True
            road_edges.append(e)
            for n in e.nodes:
                n.road = True
                road_nodes.append(n)
                road_nodes_idx.append(self.node_list.index(n))


    self.max_road_cost = max([
        self.G[e.nodes[0]][e.nodes[1]]['weight'] for e in self.myedges()
        if not e.road
    ])
    self.total_road_cost = 0

    self.roads_update = True
    self.road_nodes = road_nodes
    self.road_nodes_idx = road_nodes_idx
    self.road_edges = road_edges

    self.build_road_num = 0
    self.max_road_num = len(self.edge_list) - len(self.road_edges)

def define_interior_parcels(self):
    """defines what parcels are on the interior based on
        whether their nodes are on roads.  Relies on self.inner_facelist
        and self.road_nodes being updated. Writes to self.interior_parcels
        and self.interior_nodes
        """

    if self.G.number_of_nodes() < 2:
        return []

    interior_parcels = []

    for n in self.G.nodes():
        mgh.is_roadnode(n, self)

    self.road_nodes = [n for n in self.G.nodes() if n.road]
    self.road_nodes_idx = [
        self.node_list.index(n) for n in self.road_nodes
    ]

    # rewrites all edge properties as not being interior.This needs
    # to happen BEFORE we define the edge properties for parcels
    # that are interior, in order to give that priority.
    for e in self.myedges():
        e.interior = False

    for f in self.inner_facelist:  #   interior_parcels  replaced by czb
        if len(set(f.nodes).intersection(set(self.road_nodes))) == 0:
            f.on_road = False
            interior_parcels.append(f)
        else:
            f.on_road = True
            for n in f.nodes:
                n.interior = False

    for p in interior_parcels:
        for e in p.edges:
            e.interior = True

    for n in self.G.nodes():
        mgh.is_interiornode(n, self)

    self.interior_parcels = interior_parcels
    self.interior_nodes = [n for n in self.G.nodes() if n.interior]

    if not hasattr(self, 'max_interior_parcels'):
        self.max_interior_parcels = len(self.interior_parcels)
        self.max_del_interior_parcels = 0
        for n in self.G.nodes:
            self.max_del_interior_parcels = max(
                len(list(self.G.neighbors(n))),
                self.max_del_interior_parcels)
        self.max_del_interior_parcels = self.max_del_interior_parcels - 2
    # print

jsonPath = r"C:\Users\asdbe\OneDrive - Singapore University of Technology and Design\SUTD\Work\Working\Week105\env1.json"
ptCoordDict,edgeDict,parcelEdgeIDs = LoadJSONData(jsonPath)

myG,myNodeDict,myEdgeDict = GraphFromJSON_Customized(ptCoordDict,edgeDict,parcelEdgeIDs)
define_roads_newMethod(myG)
define_interior_parcels(myG)
#PlotMyGraph(myG,showInternal=True)

def Debug_CheckRepeatedEdges(myG,myEdgeDict):
    edgeIDList = list(myEdgeDict.keys())
    edgeValueList = list(myEdgeDict.values())
    print (len(edgeValueList))

    existIDs = []
    for edge in myG.myedges():
        existIDs.append(edgeIDList[edgeValueList.index(edge)])





myG.feature_init()
# print (myG.get_obs())
myG.td_dict_init()
edge_part_feature = myG._get_edge_part_feature()

print (edge_part_feature)
