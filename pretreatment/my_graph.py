import sys
import os
cwd = os.getcwd()
sys.path.append(cwd) 

    
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import itertools
import math
import warnings
import json
import pretreatment.my_graph_helpers as mgh
from pretreatment.lazy_property import lazy_property
import pandas as pd
from typing import Tuple, Dict, List, Text, Callable
import random,time


from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

from shapely.geometry import Point, Polygon
#import plotly.plotly as py
#from plotly.graph_objs import *

class MyNode(object):
    """ rounds float nodes to (2!) decimal places, defines equality """

    def __init__(self, locarray, name=None):
        significant_figs = 5   # original 2, change it to 5
        if len(locarray) != 2:
            print("error")
        x = locarray[0]
        y = locarray[1]
        self.x = np.round(float(x), significant_figs)
        self.y = np.round(float(y), significant_figs)
        self.loc = (self.x, self.y)
        self.road = False
        self.interior = False
        self.barrier = False
        self.name = name

        ##########################
        # Summarize added features
        self.onBoundary:bool = None
        self.external:bool = None
        self.internal:bool = None
        self.isPOI:bool = None
        self.shortCutNode:bool = None
        self.POI_Cat:str = None

    def __repr__(self):
        if self.name:
            return self.name
        else:
            return "(%.2f,%.2f)" % (self.x, self.y)   # rounded

    def __eq__(self, other):
        if hasattr(other, 'loc'):
            return self.loc == other.loc
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.loc < other.loc

    def __hash__(self):
        return hash(self.loc)


class MyEdge(object):
    """ keeps the properties of the edges in a parcel."""

    def __init__(self, nodes):
        self.nodes = tuple(nodes)
        self.interior = False
        self.road = False
        self.barrier = False
        
        ##########################
        # Summarize added features
        self.external:bool = None
        self.internal:bool = None
        self.onBoundary:bool = None
        self.isRoad:bool = None
        self.isConstraint:bool = None
        self.isPOI:bool = None
        self.fake:bool = None
        self.isShortCut:bool = None

    @lazy_property
    def length(self):
        return mgh.distance(self.nodes[0], self.nodes[1])

    @lazy_property
    def rads(self):
        return math.atan((self.nodes[0].y - self.nodes[1].y) /
                         (self.nodes[0].x - self.nodes[1].x))

    def __repr__(self):
        return "MyEdge with nodes {} {}".format(self.nodes[0], self.nodes[1])

    def __eq__(self, other):
        return ((self.nodes[0] == other.nodes[0]
                 and self.nodes[1] == other.nodes[1])
                or (self.nodes[0] == other.nodes[1]
                    and self.nodes[1] == other.nodes[0]))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.nodes)


class MyFace(object):
    """class defines a face (with name and list of edges & nodes)
       from a list of edges in the face"""

    def __init__(self, list_of_edges):
        # make a list of all the nodes in the face

        isMyEdge = False
        if len(list_of_edges) > 0:
            isMyEdge = type(list_of_edges[0]) != tuple

        if isMyEdge:
            node_set = set(n for edge in list_of_edges for n in edge.nodes)
        else:
            node_set = set(n for edge in list_of_edges for n in edge)

        self.nodes = sorted(list(node_set))
        alpha_nodes = map(str, self.nodes)
        self.name = ".".join(alpha_nodes)
        self.paths = None
        self.on_road = False
        self.even_nodes = {}
        self.odd_node = {}

        # the position of the face is the centroid of the nodes that
        # compose the face

        if isMyEdge:
            self.edges = set(list_of_edges)
            self.ordered_edges = list_of_edges
        else:
            self.edges = set(MyEdge(e) for e in list_of_edges)
            self.ordered_edges = [MyEdge(e) for e in list_of_edges]

    @lazy_property
    def area(self):
        return 0.5 * abs(
            sum(e.nodes[0].x * e.nodes[1].y - e.nodes[1].x * e.nodes[0].y
                for e in self.ordered_edges))

    @lazy_property
    def centroid(self):
        """finds the centroid of a MyFace, based on the shoelace method
        e.g. http://en.wikipedia.org/wiki/Shoelace_formula and
        http://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon
        The method relies on properly ordered edges. """

        a = 0.5 * (sum(
            e.nodes[0].x * e.nodes[1].y - e.nodes[1].x * e.nodes[0].y
            for e in self.ordered_edges))
        if abs(a) < 0.01:
            cx = np.mean([n.x for n in self.nodes])
            cy = np.mean([n.y for n in self.nodes])
        else:
            cx = (1 / (6 * a)) * sum(
                [(e.nodes[0].x + e.nodes[1].x) *
                 (e.nodes[0].x * e.nodes[1].y - e.nodes[1].x * e.nodes[0].y)
                 for e in self.ordered_edges])
            cy = (1 / (6 * a)) * sum(
                [(e.nodes[0].y + e.nodes[1].y) *
                 (e.nodes[0].x * e.nodes[1].y - e.nodes[1].x * e.nodes[0].y)
                 for e in self.ordered_edges])

        return MyNode((cx, cy))

    def __len__(self):
        return len(self.edges)

    def __repr__(self):
        return "Face with centroid at (%.2f,%.2f)" % (self.centroid.x,
                                                      self.centroid.y)


class MyGraph(object):

    def __init__(self, G=None, name="S0"):
        self.name = name
        self.roads_update = True                    # Pending
        self.rezero_vector = np.array([0, 0])
        self.rescale_vector = np.array([1, 1])
        self.td_dict = {}

        ##########################
        # Summarize added features
        self.node_list: list[MyNode] = [] 
        self.edge_list: list[MyEdge] = []
        self.stage2edges: list[MyEdge] = []
        self.full_connected_road_num: int = 0

        self.f2f_data: list[float] = []
        self.cost_data: list[float] = []
        self.parcels_data: list[int] = []


        self.POI_A_data: list[float] = []
        self.POI_B_data: list[float] = []
        self.POI_C_data: list[float] = []
        self.POI_Mean_data: list[float] = []
        self.save_step_dataCount: int  = 0

        self.road_nodes: list[MyNode] = [] 
        self.road_nodes_idx: list[int] = []
        self.road_edges: list[MyEdge] = [] 

        self.max_road_cost: float = 0
        self.total_road_cost: float = 0

        self.roads_update: bool = True

        self.build_road_num: int = 0
        self.max_road_num: int = 0

        ##########################
        # Summarize added features
        # self.inner_facelist: list[MyFace] = [] # lazy property
        # self.outerface: MyFace = None

        self.interior_parcels: list[MyFace] = [] 
        self.interior_nodes: list[MyNode] = [] 

        #self.max_interior_parcels: int = 0
        #self.max_del_interior_parcels: int = 0

        ##########################
        # Summarize added features
        # RoadG
        self.td_dict: dict[int, dict[int, float]] = {}
        self.td_dict_face: dict[MyFace, dict[MyFace, float]] = {}
        self.f2f_avg:float = 0 # this is ave for the graph

        # MinG
        self.td_dict_min: dict[MyNode, float] = {}   # this is ave
        self.td_dict_face_min: dict[MyFace, dict[MyFace, float]] = {}
        self.td_face_min: dict[MyFace, float] = {}   # this is ave
        self.f2f_avg_min = 0 # this is ave for the graph

        ##########################
        # Summarize added features
        self.POIInfo: dict[MyEdge,list[MyNode]] = {}
        self.POIEdges: list[MyEdge] = []
        self.POINodes: list[MyNode] = []
        self.inner_facelist_True: list[MyFace] = [] 

        ##########################
        # Summarize added features
        self.td_dict_nodeToPOInode: dict[MyNode, dict[MyNode, float]] = {}
        self.td_dict_nodeToPOIEdge: dict[MyNode, dict[MyEdge, float]] = {}
        self.td_dict_faceToPOIEdge: dict[MyFace, dict[MyEdge, float]] = {}
        self.td_dict_ave_faceToPOIEdge: dict[MyFace, float] = {}
        self.td_dict_ave_faceToPOIEdge_min: dict[MyFace, float] = {}

        self.td_dict_nodeToPOInode: dict[MyNode, dict[MyNode, float]] = {}
        self.td_dict_nodeToPOIEdge: dict[MyNode, dict[MyEdge, float]] = {}
        self.td_dict_faceToPOIEdge: dict[MyFace, dict[MyEdge, float]] = {}
        self.td_dict_ave_faceToPOIEdge: dict[MyFace, float] = {}
        self.td_dict_ave_faceToPOIEdge_min: dict[MyFace, float] = {}

        ##########################
        self.ringRoads = []
        ##########################
        # Summarize added features
        self.f2POI_avg = 10000
        self.f2POI_avg_min = 10000
        self.f2POI_avg_EachCat_mean = 10000    
        if G is None:
            self.G = nx.Graph()

        else:
            self.G = G

        ##########################
        # Record the total data
        self.total_angle_cost: float = 0    
        self.total_hit_boundary_reward: float = 0   
        self.total_explcit_connection_reward: float = 0   
        self.total_implcit_connection_reward: float = 0   
        self.total_all_connection_reward: float = 0  
        self.total_L_T_reward: float = 0
        self.total_RingRoad_reward: float = 0   
        self.accumulated_reward: float = 0   


    def __repr__(self):
        return "Graph (%s) with %d nodes" % (self.name,
                                             self.G.number_of_nodes())

    def add_node(self, n):
        self.G.add_node(n)

    def add_edge(self, e, weight=None):               
        assert isinstance(e, MyEdge)
        if weight is None:
            w = e.length
        else:
            w = weight
        self.G.add_edge(e.nodes[0], e.nodes[1], myedge=e, weight=w)

    def _fake_edge(self, centroid, mynode):
        newedge = MyEdge((centroid, mynode))
        newedge.length = 0
        self.add_edge(newedge)

    def location_dict(self):
        return dict((n, n.loc) for n in self.G.nodes())

    def connected_components(self):
        return [
            MyGraph(g, self.name) for i, g in enumerate([
                self.G.subgraph(c).copy()
                for c in nx.connected_components(self.G)
            ])
        ]

    def myedges(self):
        return [self.G[e[0]][e[1]]["myedge"] for e in self.G.edges()]

    def myweight(self):
        return [self.G[e[0]][e[1]]["weight"] for e in self.G.edges()]

    # Be careful, only the graph itself is a DeepCopy
    def copy(self,recalculateTag=True):
        """  Relies fundamentally on nx.copy function.  This creates a copy of
        the nx graph, where the nodes and edges retain their properties.
        MyGraph properties have to be recalculated, because copy needs to make
        entirely new faces and face attributes.
        """

        nx_copy = self.G.copy()
        copy = MyGraph(nx_copy)
 
        copy.name = self.name
        copy.rezero_vector = self.rezero_vector
        copy.rescale_vector = self.rescale_vector
        copy.td_dict = self.td_dict    # this is not deepcopy

        copy.total_road_cost = self.total_road_cost 
        copy.f2f_data = [value for value in self.f2f_data] 
        copy.full_connected_road_num = self.full_connected_road_num
        copy.build_road_num  = self.build_road_num
        copy.stage2edges = [e for e in copy.myedges() if e in self.stage2edges]
        # outerface is a side effect of the creation of inner_facelist
        # so we operate on that in order to not CALL inner_facelist for every
        # copy.

        if hasattr(self, 'outerface') and recalculateTag==True:
            copy.inner_facelist
        else:
            pass
        


        # order matters.  road nodes before interior parcels
        if hasattr(self, 'road_nodes'):
            copy.road_nodes = [n for n in copy.G.nodes() if n.road]

        if hasattr(self, 'road_edges'):
            copy.road_edges = [e for e in copy.myedges() if e.road]

        if hasattr(self, 'edge_list'):
            copy.edge_list = [e for e in copy.myedges() if e in self.edge_list]

        if hasattr(self, 'node_list'):
            copy.node_list = [n for n in copy.G.nodes() if n in self.node_list]

        if recalculateTag==False:
            # Manually duplicate the face
            copy.inner_facelist = []
            for originalFace in self.inner_facelist:
                originalEdgeList = originalFace.ordered_edges
                newEdgeList = []
                for edge in originalEdgeList:
                    edgeIndex = self.edge_list.index(edge)
                    newEdgeList.append(copy.edge_list[edgeIndex])
                copy.inner_facelist.append(MyFace(newEdgeList))

            copy.outerface = None
            for edge in self.outerface.ordered_edges:
                edgeIndex = self.edge_list.index(edge)
                newEdgeList.append(copy.edge_list[edgeIndex])
            copy.outerface = MyFace(newEdgeList)

        if hasattr(self, 'interior_parcels'):
            copy.define_interior_parcels()



        ###################################
        ######## Manual Duplicate #########
        ###################################
        copy.node_degree_road = {}
        for node in self.node_list:
            nodeIndex = self.node_list.index(node)
            copy.node_degree_road[copy.node_list[nodeIndex]] = self.node_degree_road[node]


        copy.node_isroad = {}
        for node in self.node_isroad:
            nodeIndex = self.node_list.index(node)
            copy.node_isroad[copy.node_list[nodeIndex]] = self.node_isroad[node]

        copy.inner_nodelist_True = []
        for node in self.inner_nodelist_True:
            nodeIndex = self.node_list.index(node)
            copy.inner_nodelist_True.append(copy.node_list[nodeIndex])


        copy.inner_facelist_True = []
        for face in self.inner_facelist_True:
            faceIndex = self.inner_facelist.index(face)
            copy.inner_facelist_True.append(copy.inner_facelist[faceIndex])


        copy.POIEdges = []
        for edge in self.POIEdges:
            edgeIndex = self.edge_list.index(edge)
            copy.POIEdges.append(copy.edge_list[edgeIndex])

        copy.shortcutEdges = []
        for edge in self.shortcutEdges:
            edgeIndex = self.edge_list.index(edge)
            copy.shortcutEdges.append(copy.edge_list[edgeIndex])

        copy.POINodes = []
        for node in self.POINodes:
            nodeIndex = self.node_list.index(node)
            copy.POINodes.append(copy.node_list[nodeIndex])

        copy.road_edges = []
        for edge in self.road_edges:
            edgeIndex = self.edge_list.index(edge)
            copy.road_edges.append(copy.edge_list[edgeIndex])

        try:
            copy.td_dict_POI_Related_init()
        except:
            copy.td_dict_POI_Related_init_New()
    
        return copy
    
    @lazy_property
    def inner_facelist(self):
        inner_facelist = self.__trace_faces()
        # print "inner_facelist called for graph {}".format(self)

        self.totalArea = sum([f.area for f in inner_facelist])

        return inner_facelist


############################
# FEATURE FUNCTIONS
############################

    def _cal_node_degree_and_isroad(self):
        self.node_degree_total = {}
        self.node_degree_road = {}
        self.node_isroad = {}

        for n in self.node_list:
            self.node_degree_total[n] = len(list(self.G.neighbors(n)))
            self.node_degree_road[n] = 0
            self.node_isroad[n] = 0

        for e in self.road_edges:
            for n in e.nodes:
                self.node_degree_road[n] += 1
                self.node_isroad[n] = 1

    def _cal_graph_centrality(self):
        self.degree_cen = nx.degree_centrality(self.G)
        self.betweenness_cen = nx.betweenness_centrality(self.G,
                                                         weight='weight')
        self.eigenvector_cen = nx.eigenvector_centrality_numpy(self.G,
                                                               weight='weight')
        self.closeness_cen = nx.closeness_centrality(self.G, distance='weight')

    def _cal_edge_index_and_length(self):
        self.edge_index = []
        self.edge_length = []

        for e in self.edge_list:
            idx1 = self.node_list.index(e.nodes[0])
            idx2 = self.node_list.index(e.nodes[1])
            self.edge_index.append([idx1, idx2])
            self.edge_length.append(self.G[e.nodes[0]][e.nodes[1]]['weight'])

    #### Adapted for shorted path edge in the parcel
    def _cal_edge_face_index(self):
        self.edge_face_index = []
        for e in self.edge_list:                                                    ##### if e is the shortcut egde, this will still be calculated
            pair = []
            for f in self.inner_facelist:
                if len(set(e.nodes).intersection(set(f.nodes))) == 2:
                    pair.append(f)
                if len(pair) == 2:
                    break
                        
            if pair == []:

                for f in self.inner_facelist:
                
                # meaning this is the shortcut edge
                    polygon_points = []
                    for edge in f.ordered_edges:
                        polygon_points.append(Point(edge.nodes[0].x, edge.nodes[0].y))
                    polygon_points.append(polygon_points[0])
                    polygon = Polygon(polygon_points)
                    point0 = Point(e.nodes[0].x, e.nodes[0].y)
                    point1 = Point(e.nodes[1].x, e.nodes[1].y)
                    if polygon.contains(point0) or polygon.contains(point1):
                        pair.append(f)
                        break
          

            self.edge_face_index.append(pair)


    def _cal_graph_node_feature(self,withPOITag = False):
        self.graph_node_feature = {}

        if withPOITag == False:
            for n in self.node_list:
                self.graph_node_feature[n] = self._get_node_loc(
                    n) + self._get_node_centrality(n)

        elif withPOITag == True:
            for n in self.node_list:
                self.graph_node_feature[n] = self._get_node_loc(
                    n) + self._get_node_centrality(n) + self._get_node_isPOI(n)

    def feature_init(self):
        self._cal_graph_centrality()                     # Create the dictionaries of graph info
        self._cal_graph_node_feature()                   # graph_node_feature : [x,y,degree_cen,betweenness_cen,eigenvector_cen,closeness_cen] , they are all number
        self._cal_edge_index_and_length()                # Create the list pair for edge info  
        self._cal_node_degree_and_isroad()               # Create the node degree info, these are pure number, differ from the above ones
        self._cal_edge_face_index()                      # Pair the faces that share with one edge


    # For now this is working like this:
    # If there is no culdesac, it will this will select the edge at the mid of two connected road edges or edges connected with cul-de-sac
    # Else, it will select the edge that is connected with the new road edge from the cul-de-sac side
    # There is an insurance to make sure it wont select the existing road edge
    
    ########################################################################################################################################## 
    ########################### this version is for strict selection rule - forcing selecting straightly connect edges
    def get_obs_strict(self):
        numerical = self._get_numerical()
        node_feature = np.concatenate(
            [[self._get_node_feature(n) for n in self.node_list]], axis=1)          # [[1 x (numNodeFeature x numNode)]]
        # node_feature = np.zeros_like(node_feature)

        edge_part_feature = self._get_edge_part_feature()
        # edge_part_feature = np.zeros_like(edge_part_feature)
        edge_index = self.edge_index

        ############################
        ####### new Adaption #######
        edge_mask = self.Combinded_Selection_Rule_First()
            #print ("edge_mask",edge_mask)


        edge_mask = np.array(edge_mask)

        
        ############################
        return numerical, node_feature, edge_part_feature, edge_index, edge_mask

    # For this, the task is to quickly get it connected. So exclude the edge of mid connection
    def get_obs_stage2_culdesac_strict(self):
        
        numerical = self._get_numerical()
        node_feature = np.concatenate(
            [[self._get_node_feature(n) for n in self.node_list]], axis=1)          # [[1 x (numNodeFeature x numNode)]]
        # node_feature = np.zeros_like(node_feature)

        edge_part_feature = self._get_edge_part_feature()
        # edge_part_feature = np.zeros_like(edge_part_feature)
        edge_index = self.edge_index
        edge_mask = self.Combinded_Selection_Rule_Second()
        # print ("get_obs_stage2_culdesac",edge_mask)

        return numerical, node_feature, edge_part_feature, edge_index, edge_mask
    ############################################################################################################################################

    ###########################################################################################################################################
    ########################### this version is for relatively flexible selection rule - forcing selecting straightly connect edges
    def get_obs(self):
        numerical = self._get_numerical()
        node_feature = np.concatenate(
            [[self._get_node_feature(n) for n in self.node_list]], axis=1)          # [[1 x (numNodeFeature x numNode)]]
        # node_feature = np.zeros_like(node_feature)

        edge_part_feature = self._get_edge_part_feature()
        # edge_part_feature = np.zeros_like(edge_part_feature)
        edge_index = self.edge_index

        ############################
        ####### new Adaption #######

        # print ("culdesacNum!!",self.culdesacNum)
        newAddedRoad = [e for e in self.road_edges if e not in self.originalRoadEdges]
        if len(newAddedRoad) == 0:
        #if self.culdesacNum == 0:          
            # print ("_get_edge_mask()!!!!!!!!!!!!!!")                                                              
            edge_mask = self._get_edge_mask()
        else:
            edge_mask = self._get_edge_mask_from_new_road(False)
            # print ("_get_edge_mask_from_new_road()!!!!!!!!!!!!!!") 
            #print ("edge_mask",edge_mask)


        edge_mask = np.array(edge_mask)

        
        ############################
        return numerical, node_feature, edge_part_feature, edge_index, edge_mask

    # For this, the task is to quickly get it connected. So exclude the edge of mid connection
    def get_obs_stage2_culdesac(self):
        
        numerical = self._get_numerical()
        node_feature = np.concatenate(
            [[self._get_node_feature(n) for n in self.node_list]], axis=1)          # [[1 x (numNodeFeature x numNode)]]
        # node_feature = np.zeros_like(node_feature)

        edge_part_feature = self._get_edge_part_feature()
        # edge_part_feature = np.zeros_like(edge_part_feature)
        edge_index = self.edge_index
        edge_mask = self._get_edge_mask_from_new_road(True)
        # print ("get_obs_stage2_culdesac",edge_mask)

        return numerical, node_feature, edge_part_feature, edge_index, edge_mask
    ###########################################################################################################################################


    # ok
    def _get_edge_part_feature(self,withPOITag = False):
        if withPOITag == False:
            edge_isroad = np.array(self._get_edge_isroad()).reshape(-1, 1)
            edge_length = np.array(self.edge_length).reshape(-1, 1)

            edge_face_interior = np.array(self._get_edge_face_interior()).reshape(-1, 1)
            # edge_face_interior = np.zeros_like(edge_face_interior)
            edge_avg_dis = np.array(self._get_edge_avg_dis()).reshape(-1, 1)
            
            # edge_avg_dis = np.zeros_like(edge_avg_dis)
            # edge_outerface_dis = np.array(self._get_edge_outerface_dis()).reshape(-1, 1)
            edge_ration_dis = np.array(self._get_edge_ration_dis()).reshape(-1, 1)
            # edge_ration_dis = np.zeros_like(edge_ration_dis)

            edge_part_feature = np.concatenate(
                [edge_isroad, edge_length, edge_face_interior ,edge_avg_dis, edge_ration_dis], axis=1)          # [[numEdge x (numEdgeFeature)]]
        
        elif withPOITag == True:
            edge_isroad = np.array(self._get_edge_isroad()).reshape(-1, 1)
            edge_length = np.array(self.edge_length).reshape(-1, 1)

            edge_face_interior = np.array(self._get_edge_face_interior()).reshape(-1, 1)
            # edge_face_interior = np.zeros_like(edge_face_interior)
            edge_avg_dis = np.array(self._get_edge_avg_dis()).reshape(-1, 1)
            # edge_avg_dis = np.zeros_like(edge_avg_dis)
            # edge_outerface_dis = np.array(self._get_edge_outerface_dis()).reshape(-1, 1)
            edge_ration_dis = np.array(self._get_edge_ration_dis()).reshape(-1, 1)
            # edge_ration_dis = np.zeros_like(edge_ration_dis)

            edge_isPOI = np.array(self._get_edge_isPOI()).reshape(-1, 1)
            edge_part_feature = np.concatenate(
                [edge_isroad, edge_isPOI, edge_length, edge_face_interior ,edge_avg_dis, edge_ration_dis], axis=1)
        

        return edge_part_feature



    ####################
    # edge in road network: is 0
    # edge not in road network: it could be 2  (intersected) or 1 (not intersected),  is there is 2, prioritize 2
    # in code running, this will select the edge at the mid of two connected road edges or edges connected with cul-de-sac
    ####################
    def _get_edge_mask(self):    
        #print ("normal _get_edge_mask!!!!!!!!!!!!!!")               
        edge_mask = []
        interior_del_able = False
        for e in self.edge_list:
            if (e not in self.road_edges) and set(e.nodes).intersection(set(self.road_nodes)):
                if len(self.interior_parcels):
                    if set(e.nodes).intersection(set(self.interior_nodes)):
                        edge_mask.append(2)
                        interior_del_able = True
                    else:
                        edge_mask.append(1)
                else:
                    edge_mask.append(1)
            else:
                edge_mask.append(0)

        if interior_del_able:
            #print ("normal _get_edge_mask!!!!!!!!!!!!!!  if interior_del_able")    
            index_equ2 = np.argwhere(np.array(edge_mask) == 2)
            edge_mask = np.zeros(len(edge_mask))
            edge_mask[index_equ2] = 1

        return edge_mask
   
 
    def _get_edge_mask_from_new_road(self,excludeMidConnection):
        #print ("!!!!!!! _get_edge_mask_from_new_road!!!!!!!!!")
        edge_mask = [0 for edge in self.edge_list]
        #originalRoadEdges = [e for e in self.edge_list if e.isRoad]
        newAddedRoad = [e for e in self.road_edges if e not in self.originalRoadEdges]

        new_roadNodeCollection = []
        for edge in newAddedRoad:
            new_roadNodeCollection.append(edge.nodes[0])
            new_roadNodeCollection.append(edge.nodes[1])    # This new road node collection excludes the nodes on boundary


        old_roadNodeCollection = []
        for edge in self.originalRoadEdges:
            old_roadNodeCollection.append(edge.nodes[0])
            old_roadNodeCollection.append(edge.nodes[1])

        all_roadNodeCollection = new_roadNodeCollection + old_roadNodeCollection
        
        for i in range(len(self.edge_list)):
            e = self.edge_list[i]
            if e not in self.road_edges:
                if e.nodes[0] in new_roadNodeCollection or e.nodes[1] in new_roadNodeCollection: # case 1, select from mid of cul-de-sac or end of cul-de-sac
                    edge_mask[i] = 1
                # elif e.nodes[0] in new_roadNodeCollection and e.nodes[1] in old_roadNodeCollection: # case 2, select a edge in between the new road edge and the old road edge
                #     edge_mask[i] = 1
                # elif e.nodes[1] in new_roadNodeCollection and e.nodes[0] in old_roadNodeCollection: # case 2, select a edge in between the new road edge and the old road edge
                #     edge_mask[i] = 1   

                # exclude wasted connection
                # if all_roadNodeCollection.count(e.nodes[1]) >=2 and all_roadNodeCollection.count(e.nodes[0]) >=2:   # comment to see if it increases T junction performance
                #     edge_mask[i] = 0  
                    

                if excludeMidConnection == True:
                    # if self.edge_list.index(e) == 219:
                    #     print ("asd1111",e.nodes[1] in new_roadNodeCollection,e.nodes[0] in new_roadNodeCollection,edge_mask[i])
                    
                    if all_roadNodeCollection.count(e.nodes[1]) >=2 and all_roadNodeCollection.count(e.nodes[0]) !=1:
                        edge_mask[i] = 0
                    elif all_roadNodeCollection.count(e.nodes[0]) >=2 and all_roadNodeCollection.count(e.nodes[1]) !=1:
                        edge_mask[i] = 0

                    # if self.edge_list.index(e) == 219:
                    #     print ("asd2222",e.nodes[1] in new_roadNodeCollection,e.nodes[0] in new_roadNodeCollection,edge_mask[i])
                    #     print ("asd2222",new_roadNodeCollection.count(e.nodes[0]),new_roadNodeCollection.count(e.nodes[1]))
                    #     print (e.nodes[0],e.nodes[1])

                    # if i == 63:
                    #     print (new_roadNodeCollection.count(e.nodes[1])),new_roadNodeCollection.count(e.nodes[0])
        #Avoid Selecting shortcut edge for both case
        for i in range(len(self.edge_list)):
            e = self.edge_list[i]
            if e.isShortCut:
                edge_mask[i] = 0

        #To make sure it wont select the existing road edge     
        if 1 not in edge_mask:
            print ("if 1 not in edge_mask!!!!!!!!!")
            # print ("culdesac num",self.culdesacNum)
            # print ("Debug")
            # roadG = MyGraph()
            # for idx in range(len(self.road_edges)):
            #     e = self.road_edges[idx]
            #     roadG.add_edge(self.road_edges[idx],weight=self.G[e.nodes[0]][e.nodes[1]]['weight'])

            # single_neighbor_nodes = [node for node in roadG.G.nodes() if len(list(roadG.G.neighbors(node))) == 1]
            #print (single_neighbor_nodes,single_neighbor_nodes)



            edge_mask = self._get_edge_mask()   #original
            # info = []
            # for edge in self.edge_list:
            #     for node in edge.nodes:
            #         info.append(node.x)
            #         info.append(node.y)

            # print ("info",info)
            # roadEdgeIndices = []
            # for roadEdge in self.road_edges:
            #     roadEdgeIndices.append(self.edge_list.index(roadEdge))
            # print ("roadEdgeIndices",roadEdgeIndices)
            # print ("edge_mask",edge_mask.tolist())

        # 

        # print ("info",info)
        # roadEdgeIndices = []
        # for roadEdge in self.road_edges:
        #     roadEdgeIndices.append(self.edge_list.index(roadEdge))
        # print ("roadEdgeIndices",roadEdgeIndices)
        #print ("edge_mask",edge_mask.tolist())

        # if excludeMidConnection == True:
        #     info = []
        #     # for edge in self.edge_list:
        #     #     for node in edge.nodes:
        #     #         info.append(node.x)
        #     #         info.append(node.y)

        #     #print ("info",info)
        #     roadEdgeIndices = []
        #     for roadEdge in self.road_edges:
        #         roadEdgeIndices.append(self.edge_list.index(roadEdge))
        #     print ("roadEdgeIndices",roadEdgeIndices)
        #     print ("edge_mask",edge_mask)
            
        #print ("edge_mask219",edge_mask[219])
        return edge_mask


    # def _get_edge_mask_stage2_culdesac(self):                   # so the edge selection can start for the cul-de-sac
    #     edge_mask = [0 for edge in self.edge_list]

    #     roadNodeCollection = []
    #     for edge in self.road_edges:
    #         roadNodeCollection.append(edge.nodes[0])
    #         roadNodeCollection.append(edge.nodes[1])

    #     for i in range(len(self.edge_list)):
    #         e = self.edge_list[i]
    #         repeatCount = 0
    #         case = None
    #         if e not in self.road_edges:
       
    #             repeatCount = 0
    #             case = None
    #             if e.nodes[0].road == True and e.nodes[1].road != True:
    #                 repeatCount = roadNodeCollection.count(e.nodes[0])
    #                 #case = "case1"
    #             elif e.nodes[0].road != True and e.nodes[1].road == True:
    #                 repeatCount = roadNodeCollection.count(e.nodes[1])
    #                 #case = "case2"
    #             elif e.nodes[0].road == True and e.nodes[1].road == True:
    #                 if roadNodeCollection.count(e.nodes[1]) == 1 or roadNodeCollection.count(e.nodes[0]) == 1:
    #                     repeatCount = 1
    #                     #case = "case3"
             
    #             if repeatCount == 1:          # this is to avoid picking the L - path branch to get more cul-de-sac
    #                 edge_mask[i] = 1


    #     #Avoid Selecting shortcut edge for both case
    #     for i in range(len(self.edge_list)):
    #         e = self.edge_list[i]
    #         if e.isShortCut:
    #             edge_mask[i] = 0

    #     #To make sure it wont select the existing road edge     
    #     if 1 not in edge_mask:
    #         edge_mask = self._get_edge_mask()   #original

    #     edge_mask = np.array(edge_mask)

     
    #     return edge_mask
    
    ##########
    # Strict rule:  Staright first
    ##########

    def Collect_T_Junction_Edge(self):    # For now exclude the edges that are connected a "variance == 5" vertex
        T_junction_edges = []

        roadNodeCollection = []
        for edge in self.road_edges:
            roadNodeCollection.append(edge.nodes[0])
            roadNodeCollection.append(edge.nodes[1])
        
        nonRoadEdges = []
        for edge in self.edge_list:
            if edge not in self.road_edges:
                nonRoadEdges.append(edge)

        for edge in nonRoadEdges:
            if roadNodeCollection.count(edge.nodes[0]) >=3:  # 
                if self.G.degree(edge.nodes[0]) == 4:
                    T_junction_edges.append(edge)

            elif roadNodeCollection.count(edge.nodes[1]) >=3:
                if self.G.degree(edge.nodes[1]) == 4:
                    T_junction_edges.append(edge)

        self.curr_T_junction_edges = T_junction_edges



    def Combinded_Selection_Rule_First(self):
        edge_mask = [0 for edge in self.edge_list]
        # # Check_T_Junction_Edge
        # T_junction_edges = self.Collect_T_Junction_Edge()
        # T_junction_edges =[] # !!!!!!!!!!!!!!!!!!dont consider it first
        # if len(T_junction_edges) != 0:
        #     # print ("Case1: prioritize T-junctions")
        #     for edge in T_junction_edges:
        #         edge_mask[self.edge_list.index(edge)] = 1
        # # Check_Culdesac_Edge
        # else:
        culdesacEdges = self.CollectCuldesacEdges()
        if len(culdesacEdges) != 0:
            # check if there is any edge that the connection angle < 45 degree
            culdesacEdge_SmoothConnections = []
            for edge in culdesacEdges:
                minAngle, minAngleEdge, angles,connectedEdges  = self.AngleOfConnections(edge)
                if minAngle < 45:
                    culdesacEdge_SmoothConnections.append(minAngleEdge)
                pass
            #print ("connectedEdges",[self.edge_list.index(edge) for edge in connectedEdges])        
            if len(culdesacEdge_SmoothConnections) != 0:
                # print ("Case2: there are connection to cul-de-sac egdes and the angle is less than 45 degree")
                for edge in culdesacEdge_SmoothConnections:
                    edge_mask[self.edge_list.index(edge)] = 1
                #print (edge_mask)
            else:  # go for creating T-junctions in new roads
                newAddedRoad = [e for e in self.road_edges if e not in self.originalRoadEdges]
                new_roadNodeCollection = []
                for edge in newAddedRoad:
                    new_roadNodeCollection.append(edge.nodes[0])
                    new_roadNodeCollection.append(edge.nodes[1])    # This new road node collection excludes the nodes on boundary

                new_T_junction_edges = []
                for edge in self.edge_list:
                    if edge not in self.road_edges:
                        if new_roadNodeCollection.count(edge.nodes[0]) ==2 or new_roadNodeCollection.count(edge.nodes[1]) ==2:
                            new_T_junction_edges.append(edge)

        
                currentCandidates = connectedEdges + new_T_junction_edges
                if len(currentCandidates) != 0:
                    currentCandidates_TouchInternalParcel = []
                    internalNodeCollection = []
                    for parcel in self.interior_parcels:
                        for node in parcel.nodes:
                            internalNodeCollection.append(node)

                    for edge in currentCandidates:
                        internalNodeCount = internalNodeCollection.count(edge.nodes[0]) + internalNodeCollection.count(edge.nodes[1])
                        if internalNodeCount!=0:    
                            currentCandidates_TouchInternalParcel.append(edge)
                    
                    if len(currentCandidates_TouchInternalParcel) != 0:
                        finalSelection = currentCandidates_TouchInternalParcel
                    else:
                        finalSelection = currentCandidates
                    # print ("Case3: creating new T-junctions between new roads")
                    for edge in finalSelection:
                        edge_mask[self.edge_list.index(edge)] = 1
                
                else:
                    # print ("Case4: going back to the original rule, finding edges that are connected to boundary")
                    edge_mask = self._get_edge_mask()   #original
        #Avoid Selecting shortcut edge for both case
        # WIP
        #       
        elif len(culdesacEdges) == 0:
            edge_mask = self._get_edge_mask()   #original
            # print ("Case5: going back to the original rule, finding edges that are connected to boundary")
        # print ("-----")   
        return edge_mask

    def Combinded_Selection_Rule_Second(self):
        edge_mask = [0 for edge in self.edge_list]
        culdesacEdges = self.CollectCuldesacEdges()

        if len(culdesacEdges) != 0:
            # check if there is any edge that the connection angle < 45 degree
            culdesacEdge_SmoothConnections = []
            for edge in culdesacEdges:
                minAngle, minAngleEdge, angles, connectedEdges= self.AngleOfConnections(edge)
                culdesacEdge_SmoothConnections.append(minAngleEdge)

            for edge in culdesacEdge_SmoothConnections:
                edge_mask[self.edge_list.index(edge)] = 1
        
        else:
            edge_mask = self._get_edge_mask()   #original
        return edge_mask
    

    def Collect_L_Junction_Edge(self,ratio = 0):
        roadG = MyGraph()
        for idx in range(len(self.road_edges)):
            e = self.road_edges[idx]
            roadG.add_edge(self.road_edges[idx],weight=self.G[e.nodes[0]][e.nodes[1]]['weight'])

        L_junction_edges = []

        roadNodeCollection = []
        for edge in self.road_edges:
            roadNodeCollection.append(edge.nodes[0])
            roadNodeCollection.append(edge.nodes[1])
        
        for eachRoadNode in roadNodeCollection:
            if roadNodeCollection.count(eachRoadNode) == 2:
                # Get these two edges
                twoConnectedEdges = []
                for edge in self.road_edges:
                    if edge.nodes[0] == eachRoadNode or edge.nodes[1] == eachRoadNode:
                        twoConnectedEdges.append(edge)
          
                angle =  roadG.GetAngle(twoConnectedEdges[0],twoConnectedEdges[1])

                relativeLength_A = twoConnectedEdges[0].length / self.max_road_cost
                relativeLength_B = twoConnectedEdges[1].length  / self.max_road_cost

                
                if angle > 45:   # meaning this is a L-junction
                    # print ("twoConnectedEdges",[self.edge_list.index(e) for e in twoConnectedEdges])
                    # print ("twoConnectedEdges",eachRoadNode,angle)
                    if relativeLength_A > ratio and relativeLength_B > ratio:
                        
                        allEdgesIncludeRoadEdges = []
                        for edge in self.edge_list:
                            if edge.nodes[0] == eachRoadNode or edge.nodes[1] == eachRoadNode:
                                allEdgesIncludeRoadEdges.append(edge)

                        for edge in allEdgesIncludeRoadEdges:
                            if edge not in self.road_edges and edge not in L_junction_edges:
                                L_junction_edges.append(edge)
        
        self.curr_L_junction_edges = L_junction_edges


    # ok
    def _get_edge_face_interior(self):
        edge_face_interior=[]
        #pairLength = []
        for pair in self.edge_face_index:          ##### if e is the shortcut egde, this will still be calculated
            if len(pair) == 1:
                edge_face_interior.append(0)
            elif len(pair) == 2:
                f1 = pair[0]
                f2 = pair[1]
                inter=0
                if f1 in self.interior_parcels:
                    inter += 1
                if f2 in self.interior_parcels:
                    inter += 1
                
                edge_face_interior.append(inter/2)
            
            #pairLength.append(len(pair) )
            # elif len(pair) == 0:  # new case
            #     edge_face_interior.append(0)
        # return np.zeros_like(edge_face_interior)
        # print (pairLength)
        # print ("edge_face_interior",len(edge_face_interior))
        return edge_face_interior
    
    # ok
    def _get_edge_ration_dis(self):
        edge_dis_ration=[]
        for idx in range(len(self.edge_list)):
            idx1 = self.edge_index[idx][0]
            idx2 = self.edge_index[idx][1]
            if self.td_dict[idx1][idx2] == 10000:
                ration = 0.8
            else:
                ration = self.edge_length[idx]/self.td_dict[idx1][idx2] 
            edge_dis_ration.append(ration) 
        # return np.zeros_like(edge_dis_ration)
        return edge_dis_ration

    # ok
    def _get_edge_avg_dis(self):
        edge_dis = []
        face_mean_dis = {}
        for f1 in self.inner_facelist:
            dis=0
            count=0
            for f2 in self.inner_facelist:
                dis += self.td_dict_face[f1][f2]
                count += 1

            face_mean_dis[f1] = self.td_face_min[f1]/(dis/(count-1))

        #pairLength = []
        for pair in self.edge_face_index:                                       ##### if e is the shortcut egde, this will still be calculated
            if len(pair) == 1:
                f = pair[0]
                mean_dis = face_mean_dis[f]
            elif len(pair) == 2:
                f1 = pair[0]
                f2 = pair[1]
                mean_dis = (face_mean_dis[f1] + face_mean_dis[f2]) / 2
            # elif len(pair) == 0:
            #     # then this should be consider the same as len(pair) == 1
            #     mean_dis = 10000
            #pairLength.append(len(pair) )
            edge_dis.append(mean_dis)

        # print (pairLength)
        # print ("edge_dis",len(edge_dis))
        return edge_dis

    # ok
    def _get_edge_isroad(self):
        edge_isroad = []
        for e in self.edge_list:
            if e in self.road_edges:
                edge_isroad.append(1)
            else:
                edge_isroad.append(0)
        # return np.zeros_like(edge_isroad)
        return edge_isroad

    def _get_edge_isPOI(self):           #### New Added
        edge_isPOI= []
        for e in self.edge_list:
            if e.isPOI == True:
                edge_isPOI.append(1)
            else:
                edge_isPOI.append(0)
   
        return edge_isPOI
    
    # ok - double check
    def get_numerical_feature_size(self):
        return 5   # original is 4, change it to 5 becasue of cul-de-sac num

    # what is 0.5 used for
    def _get_numerical(self):
        
        if self.full_connected_road_num == 0:
            stage1_num = self.build_road_num
            stage2_num = 0
        else:
            stage1_num = self.full_connected_road_num
            stage2_num = self.build_road_num - self.full_connected_road_num
        stage1_ration = stage1_num / self.max_road_num
        stage2_ration = stage2_num / self.max_road_num

        if self.max_interior_parcels!= 0:
            interior_ration = len(
                self.interior_parcels) / self.max_interior_parcels
        else:
            interior_ration = 0          # newly added for internal path cases

        # include culdesacNum as one data
        culdesacNum = self.culdesacNum

        # print(stage1_ration, stage2_ration, interior_ration)
        return [0.5, stage1_ration, stage2_ration, interior_ration,culdesacNum]

    # ok
    def _get_full_connected_road_num(self):
        return self.full_connected_road_num
    # ok
    def _get_node_feature(self, node: MyNode):
        return self.graph_node_feature[node] + self._get_node_degree_ration(
            node) + self._get_node_isroad(node)  + self._get_node_interior(node) + self._get_node_dis(node)
    # ok
    def _get_node_loc(self, node: MyNode):
        return [node.x, node.y]
    # ok
    def _get_node_degree_ration(self, node: MyNode):
        return [self.node_degree_road[node]/self.node_degree_total[node]]
    # ok
    def _get_node_isroad(self, node: MyNode):
        return [self.node_isroad[node]]
    # ok
    def _get_node_interior(self,node:MyNode):
        # return [0]
        if node in self.interior_nodes:
            return [1]
        else: 
            return [0]
    # ok
    def _get_node_dis(self, node: MyNode):
        idx = self.node_list.index(node)
        # return [0]
        return [self.td_dict_min[self.node_list[idx]]/np.mean(self.td_dict[idx])]
    # ok
    def _get_node_centrality(self, node: MyNode):
        return [
            self.degree_cen[node], self.betweenness_cen[node],
            self.eigenvector_cen[node], self.closeness_cen[node]
        ]
        
    def _get_node_isPOI(self, node: MyNode):    # new added
        return [node.isPOI]

############################
# GEOMETRY CLEAN UP FUNCTIONS
############################

##########################################
#    WEAK DUAL CALCULATION FUNCTIONS
########################################

    def get_embedding(self):
        emb = {}
        for i in self.G.nodes():
            neighbors = self.G.neighbors(i)

            def angle(b):
                dx = b.x - i.x
                dy = b.y - i.y
                return np.arctan2(dx, dy)

            reorder_neighbors = sorted(neighbors, key=angle)
            emb[i] = reorder_neighbors
        return emb

    def __trace_faces(self):
        """Algorithm from SAGE"""
        if len(self.G.nodes()) < 2:
            inner_facelist = []
            return []

        # grab the embedding
        comb_emb = self.get_embedding()

        # Establish set of possible edges
        edgeset = set()
        for edge in self.G.edges():
            edgeset = edgeset.union(
                set([(edge[0], edge[1]), (edge[1], edge[0])]))

        # Storage for face paths
        faces = []

        # Trace faces
        face = [edgeset.pop()]
        while (len(edgeset) > 0):
            neighbors = comb_emb[face[-1][-1]]
            next_node = neighbors[(neighbors.index(face[-1][-2]) + 1) %
                                  (len(neighbors))]
            edge_tup = (face[-1][-1], next_node)
            if edge_tup == face[0]:
                faces.append(face)
                face = [edgeset.pop()]
            else:
                face.append(edge_tup)
                edgeset.remove(edge_tup)

        if len(face) > 0:
            faces.append(face)

        # remove the outer "sphere" face
        facelist = sorted(faces, key=len)
        self.outerface = MyFace(facelist[-1])
        self.outerface.edges = [
            self.G[e[1]][e[0]]["myedge"] for e in facelist[-1]
        ]
        inner_facelist = []
        for face in facelist[:-1]:
            iface = MyFace(face)
            iface.edges = [self.G[e[1]][e[0]]["myedge"] for e in face]
            inner_facelist.append(iface)
            iface.down1_node = iface.centroid

        return inner_facelist

    def weak_dual(self):
        """This function will create a networkx graph of the weak dual
        of a planar graph G with locations for each node.Each node in
        the dual graph corresponds to a face in G. The position of each
        node in the dual is caluclated as the mean of the nodes composing
        the corresponding face in G."""
        #print (self.G)
        try:
            assert len(list(nx.connected_components(self.G))) <= 1
        except AssertionError:
            raise RuntimeError("weak_dual() can only be called on" +
                               " graphs which are fully connected.")

        # name the dual
        if len(self.name) == 0:
            dual_name = ""
        else:
            lname = list(self.name)
            nums = []
            while True:
                try:
                    nums.append(int(lname[-1]))
                except ValueError:
                    break
                else:
                    lname.pop()

            if len(nums) > 0:
                my_num = int(''.join(map(str, nums)))
            else:
                my_num = -1
            my_str = ''.join(lname)
            dual_name = my_str + str(my_num + 1)

        # check for empty graph
        if self.G.number_of_nodes() < 2:
            return MyGraph(name=dual_name)

        # get a list of all faces
        # self.trace_faces()

        # make a new graph, with faces from G as nodes and edges
        # if the faces share an edge
        dual = MyGraph(name=dual_name)
        if len(self.inner_facelist) == 1:
            face = self.inner_facelist[0]
            dual.add_node(face.centroid)
        else:
            combos = list(itertools.combinations(self.inner_facelist, 2))
            for c in combos:
                c0 = [e for e in c[0].edges if not e.road]
                c1 = [e for e in c[1].edges if not e.road]
                if len(set(c0).intersection(c1)) > 0:
                    dual.add_edge(MyEdge((c[0].centroid, c[1].centroid)))
        return dual

    def S1_nodes(self):
        """Gets the odd_node dict started for depth 1 (all parcels have a
        centroid) """
        for f in self.inner_facelist:
            f.odd_node[1] = f.centroid

    def formClass(self, duals, depth, result):
        """ function finds the groups of parcels that are represented in the
        dual graph with depth "depth+1".  The depth value provided must be even
        and less than the max depth of duals for the graph.

        need to figure out why I can return a result with depth d+1 with an
        empty list.

        """

        dm1 = depth - 1

        is_odd = bool(depth % 2)

        try:
            assert not is_odd
        except AssertionError:
            raise RuntimeError("depth ({}) should be even".format(depth))

        # flist is the list of parcels in self which are represented in the
        # dual of depth depth-1 (dm1)
        flist = [
            f for f in self.inner_facelist
            if (dm1 in f.odd_node and f.odd_node[dm1])
        ]

        dual1 = duals[dm1]
        dual2 = duals[depth]

        # flat list of faces in duals 1 and 2 for potentially many disconnected
        # dual graphs.
        dual1_faces = [f for G in dual1 for f in G.inner_facelist]
        dual2_faces = [f for G in dual2 for f in G.inner_facelist]

        # creates an association between the faces in self and the centroids
        # of faces in dual1, for faces in dual1 that overlap a face (face0) in
        # self.
        for face0 in flist:
            down2_nodes = [
                f.centroid for f in dual1_faces
                if face0.odd_node[depth - 1] in f.nodes
            ]
            face0.even_nodes[depth] = set(down2_nodes)
#            down2_nodes = []
#            for face1 in dual1_faces:
#                if face0.odd_node[depth-1] in face1.nodes:
#                    down2_nodes.append(face1.centroid)
#                    face0.even_nodes[depth] = set(down2_nodes)

# if the down2 faces for face0 make up a face in the dual2 graph, then
# the centroid of that face in the dual2 graph represents face0 in the
# dual graph with depth depth+1
        for face0 in flist:
            if depth in face0.even_nodes:
                for face2 in dual2_faces:
                    if set(face0.even_nodes[depth]) == set(face2.nodes):
                        face0.odd_node[depth + 1] = face2.centroid

        # return the results as a dict for depth depth+1, also stored as a
        # a property of each face.
        result[depth + 1] = [
            f for f in self.inner_facelist
            if depth + 1 in f.odd_node and f.odd_node[depth + 1]
        ]

        depth = depth + 2
        return duals, depth, result

    def stacked_duals(self, maxdepth=15):
        """to protect myself from an infinite loop, max depth defaults to 15"""

        def level_up(Slist):
            Sns = [g.weak_dual().connected_components() for g in Slist]
            Sn = [cc for duals in Sns for cc in duals]
            return Sn

        stacks = []
        stacks.append([self])
        while len(stacks) < maxdepth:
            slist = level_up(stacks[-1])
            if len(slist) == 0:
                break
            stacks.append(slist)

        for G in stacks:
            for g in G:
                try:
                    g.inner_facelist
                except AttributeError:
                    g.__trace_faces()
                    print("tracing faces needed")

        return stacks

#############################################
#  DEFINING ROADS AND INTERIOR PARCELS
#############################################
    
    def define_roads(self) : 
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
            return []

        # self.trace_faces()
        self.inner_facelist
        print ("745!!!",self.outerface)
        of = self.outerface

        for e in of.edges:
            e.road = True
            road_edges.append(e)
        for n in of.nodes:
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

        # print "define roads called"

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
            self.max_del_interior_parcels = self.max_del_interior_parcels - 2     # -2 is a designed adjustment， reduce the existing accessible parcel

        self.td_dict_init()
        # print "define interior parcels called"

    def update_node_properties(self): 
        for n in self.G.nodes():
            mgh.is_roadnode(n, self)
            mgh.is_interiornode(n, self)

    # ok 
    def find_interior_edges(self):
        """ finds and returns the pairs of nodes (not the myEdge) for all edges that
        are not on roads."""

        interior_etup = []

        for etup in self.G.edges():
            if not self.G[etup[0]][etup[1]]["myedge"].road:
                interior_etup.append(etup)

        return interior_etup

    # ok 
    def build_road_from_action(self, action: List,POIVersionTag = False,fixedOrder = None):
        if fixedOrder == None:
            e = self.edge_list[int(action)]
    
        else:
            roadEdgeIndices = [self.edge_list.index(e) for e in self.road_edges]
            for index in fixedOrder:
                if index not in roadEdgeIndices:
                    e = self.edge_list[index]
                    break

        self.add_road_segment(e,POIVersionTag)

    # ok 
    def road_update(self, edge):
        self.G[edge.nodes[0]][edge.nodes[1]]['road'] = self.G[edge.nodes[0]][
            edge.nodes[1]]['weight']
    
    # ok 
    def add_road_segment(self, edge: MyEdge,POIVersionTag = False):
        # print ("add_road_segment",edge)
        """ Updates properties of graph to make edge a road. """
        
        edge = self.G[edge.nodes[0]][edge.nodes[1]]['myedge']

        # self.myw = self.G[edge.nodes[0]][edge.nodes[1]]['weight']
        
        if POIVersionTag == False:
            self.td_dict_update(edge)
        elif POIVersionTag == True:
            self.td_dict_update_ForPOI(edge)



        edge.road = True
        #print ("len(self.road_edges)",len(self.road_edges),edge)
        if edge in self.road_edges:
            print (edge)

            info = []
            for edge in self.edge_list:
                for node in edge.nodes:
                    info.append(node.x)
                    info.append(node.y)

            print (info)     
            print ("------")
            print ([self.edge_list.index(e) for e in self.road_edges])

            raise ValueError("[!]Already in ")
        # if len(set(edge.nodes).intersection(set(self.road_nodes))) == 0:
        #     raise ValueError("Invalid edge")

        if hasattr(self, 'road_edges'):
            self.road_edges.append(edge)
        else:
            self.road_edges = [edge]
        
        if self.full_connected_road_num:
            self.stage2edges.append(edge)

        if hasattr(self, 'road_nodes'):
            rn = self.road_nodes
            rn_idx = self.road_nodes_idx
        else:
            rn = []
            rn_idx = []

        for n in edge.nodes:
            n.road = True
            self.node_degree_road[n] += 1
            self.node_isroad[n] = 1
            idx = self.node_list.index(n)
            if idx not in rn_idx:
                rn_idx.append(idx)
                rn.append(n)

        self.roads_update = False
        self.road_nodes = rn
        self.road_nodes_idx = rn_idx
        self.build_road_num += 1

        ### This is a new adaption so that it allows to add the seperated road and count the adjacent parcels accessible
        for parcel in self.interior_parcels:
            if edge.nodes[0] in parcel.nodes and edge.nodes[1] in parcel.nodes:  # and  or  make differneces
                parcel.parcelEdgeAccessible = True
            else:
                parcel.parcelEdgeAccessible = False
        
        self.interior_parcels_update()
        
        

    # ok 
    def add_all_road(self):
        for e in self.myedges():
            if not e.road:
                self.add_road_segment(e)


#############################################
#  DEFINING ROADS AND INTERIOR PARCELS
#############################################
    def define_roads_FirstTime(self):
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

        for e in self.myedges():
            if e.isRoad == True:
                e.road = True
                road_edges.append(e)
                for n in e.nodes:
                    n.road = True
                    road_nodes.append(n)
                    road_nodes_idx.append(self.node_list.index(n))

        try:
            self.max_road_cost = max([
                self.G[e.nodes[0]][e.nodes[1]]['weight'] for e in self.myedges()
                if not e.road
            ])
        except:
            self.max_road_cost = 0
            
        self.total_road_cost = 0

        self.roads_update = True
        self.road_nodes = road_nodes
        self.road_nodes_idx = road_nodes_idx
        self.road_edges = road_edges

        self.build_road_num = 0
        self.max_road_num = len(self.edge_list) - len(self.road_edges)





#############################################
#   GEOMETRY AROUND BUILDING A GIVEN ROAD SEGMENT - c/(sh?)ould be deleted.
#############################################

############################
# REWARD FUNCTIONS
############################
    def save_step_data(self):    # For now it only save the last one
        pass
        # if "road_planning" in cwd:  # for ssh remote terminal
        #     path = os.path.join(cwd, 'data', "data.csv")
        # else:
        #     path = os.path.join(cwd, "road_planning", 'data', "data.csv")
        # # if "road_planning" in cwd:  # for ssh remote terminal
        # #     path = os.path.join(cwd, 'data', f"data_{self.save_step_dataCount}.csv")
        # # else:
        # #     path = os.path.join(cwd, "road_planning", 'data', f"data_{self.save_step_dataCount}.csv")

        # #data=pd.DataFrame(data=[self.parcels_data,self.f2f_data,self.cost_data])  original
        # data=pd.DataFrame(data=[self.POI_A_data,self.POI_B_data,self.POI_C_data,self.POI_Mean_data])
        # data.to_csv(path,encoding='gbk')
        # self.save_step_dataCount +=1
        # print (self.save_step_dataCount)

    def road_cost(self):
        self.cost_data.append(self.total_road_cost)
        return self.current_road_cost / self.max_road_cost

    # ok  
    def total_cost(self):
        return self.total_road_cost

    # ok  
    def travel_distance(self) -> float:
        # if self._reward_count == 0:
        #     return 0
        # return 10*self._td_reward / self._reward_count

        
        if len(self.interior_parcels) or self.del_parcel_num:  # So it is not in use when the road is not fully connected
            before = self.face2face_avg()
            return 0
        else:
            before = self.f2f_avg   
            now = self.face2face_avg()    # execute to get the current
            return  (before-now)/(before-self.f2f_avg_min)

    # ok  
    def face2face_avg(self):
        sum = 0
        for i in self.inner_facelist:
                for j in self.inner_facelist:
                    sum += self.td_dict_face[i][j]
        self.f2f_avg = sum / (len(self.inner_facelist)*(len(self.inner_facelist)-1))

        sum=0
        count=0
        for i in self.inner_facelist:
                for j in self.inner_facelist:
                    if self.td_dict_face[i][j] != 10000:
                        sum += self.td_dict_face[i][j]
                        count+=1
                count -= 1 
        tmp = sum / (count)
        self.f2f_data.append(tmp)

        return self.f2f_avg

    # ok  
    def connected_ration(self):
        #print ("using this connected_ration")
     
        
        if self.del_parcel_num == 0 and len(self.interior_parcels) != 0:
            #print ("case1","self.del_parcel_num",self.del_parcel_num,"self.max_del_interior_parcels",self.max_del_interior_parcels )
            return -1/self.max_del_interior_parcels
        else:
            # print ("case2","self.del_parcel_num",self.del_parcel_num,"self.max_del_interior_parcels",self.max_del_interior_parcels )
            return self.del_parcel_num / self.max_del_interior_parcels

    def td_dict_init(self): 
        roadG = MyGraph()
        for idx in range(len(self.road_edges)):
            e = self.road_edges[idx]
            roadG.add_edge(self.road_edges[idx],weight=self.G[e.nodes[0]][e.nodes[1]]['weight'])
        td_dict = dict(nx.shortest_path_length(roadG.G, weight="weight"))


### init_td_dict
        node_length = len(self.node_list)
        # print('node:', node_length, 'edge:', len(self.edge_list))
        self.td_dict = [[a for a in range(node_length)]
                        for _ in range(node_length)]
        for n in range(node_length):
            for nn in range(node_length):
                if n == nn:
                    self.td_dict[n][nn] = 0
                else:
                    self.td_dict[n][nn] = 10000
        for i in td_dict:
            idx1 = self.node_list.index(i)
            for j in td_dict:
                idx2 = self.node_list.index(j)
                self.td_dict[idx1][idx2] = td_dict[i][j]
### init_face_dis
        self.td_dict_face = {}
        for f1 in self.inner_facelist:
            self.td_dict_face[f1] = {}
            for f2 in self.inner_facelist:
                if f1.centroid == f2.centroid:
                    self.td_dict_face[f1][f2] = 0
                else:
                    self.td_dict_face[f1][f2] = 10000
        for f1 in set(self.inner_facelist).difference(
                set(self.interior_parcels)):
            for f2 in set(self.inner_facelist).difference(
                    set(self.interior_parcels)):
                if self.td_dict_face[f1][f2] == 0:
                    continue
                else:
                    for n1 in f1.nodes:
                        idx1 = self.node_list.index(n1)
                        for n2 in f2.nodes:
                            idx2 = self.node_list.index(n2)
                            if idx1 != idx2 or (idx1 in self.road_nodes_idx):
                                self.td_dict_face[f1][f2] = self.td_dict_face[
                                    f2][f1] = min(self.td_dict_face[f1][f2],
                                                  self.td_dict[idx1][idx2])
### init_outface_dis
        self.td_dict_face[self.outerface]={}
        for f in self.inner_facelist:
            self.td_dict_face[self.outerface][f] = 0
        for f in self.interior_parcels:
            self.td_dict_face[self.outerface][f] = 10000
### cal_face_dis_min
        td_dict_min = dict(nx.shortest_path_length(self.G, weight="weight"))
        self.td_dict_min={}
        for n1 in self.node_list:
            dis=0
            count=0
            for n2 in self.node_list:
                dis += td_dict_min[n1][n2]
                count += 1
            self.td_dict_min[n1] = dis/(count-1)
                
        self.td_dict_face_min = {}
        self.td_face_min={}
        for f1 in self.inner_facelist:
            self.td_dict_face_min[f1] = {}
            for f2 in self.inner_facelist:
                self.td_dict_face_min[f1][f2] = 10000
                for n1 in f1.nodes:
                    for n2 in f2.nodes:
                        self.td_dict_face_min[f1][f2] = min(
                            self.td_dict_face_min[f1][f2], td_dict_min[n1][n2])
                        if self.td_dict_face_min[f1][f2] == 0:
                            break
                            
        for f1 in self.inner_facelist:
            dis=0
            count=0
            for f2 in self.inner_facelist:
                dis += self.td_dict_face_min[f1][f2]
                count += 1
            self.td_face_min[f1] = dis/(count-1)

        self.face2face_avg()
        sum = 0
        for i in self.td_dict_face_min:
            for j in self.td_dict_face_min[i]:
                sum += self.td_dict_face_min[i][j]
        self.f2f_avg_min = sum / (len(self.inner_facelist)*(len(self.inner_facelist)-1))

        #print('td_init')

    # update node2node & face2face distace
    # update node2node & face2face distace
    def td_dict_update(self, edge):
        n1 = edge.nodes[0]
        n2 = edge.nodes[1]
        idx1 = self.node_list.index(n1)
        idx2 = self.node_list.index(n2)
        self.current_road_cost = self.G[n1][n2]['weight']
        self.total_road_cost += self.current_road_cost
        change_two = False
        change_node = []

        ### update node2node shortest distance
        if n1 not in self.road_nodes:
            if n2 not in self.road_nodes:
                change_node.append([idx1, idx2])
                self.td_dict[idx1][idx2] = self.td_dict[idx1][idx2] = self.G[n1][n2]['weight']
            else:
                change_node.append([idx1, idx1])
                for i in self.road_nodes_idx:
                    self.td_dict[i][idx1] = self.td_dict[idx1][
                        i] = self.td_dict[i][idx2] + self.G[n1][n2]['weight']
                    change_node.append([idx1, i])

        elif n2 not in self.road_nodes:
            if n1 not in self.road_nodes:
                change_node.append([idx1, idx2])
                self.td_dict[idx1][idx2] = self.td_dict[idx1][idx2] = self.G[n1][n2]['weight']
            else:
                change_node.append([idx2, idx2])
                for i in self.road_nodes_idx:
                    self.td_dict[i][idx2] = self.td_dict[idx2][
                        i] = self.td_dict[i][idx1] + self.G[n1][n2]['weight']
                    change_node.append([idx2, i])

        else:
            change_two = True
            for i in self.road_nodes_idx:
                for j in self.road_nodes_idx:
                    before = self.td_dict[i][j]
                    self.td_dict[i][j] = self.td_dict[j][i] = min(
                        self.td_dict[i][j],
                        (min(self.td_dict[i][idx1] + self.td_dict[idx2][j],
                             self.td_dict[i][idx2] + self.td_dict[idx1][j]) +
                         self.G[n1][n2]['weight']))
                    if self.td_dict[i][j] < before:
                        change_node.append([i, j])

        # update face2face distance
        if change_two:
            for pair in change_node:
                idx1 = pair[0]
                idx2 = pair[1]
                n1 = self.node_list[idx1]
                n2 = self.node_list[idx2]
                for f1 in [f for f in self.inner_facelist if n1 in f.nodes]:
                    for f2 in [
                            f for f in self.inner_facelist if n2 in f.nodes
                    ]:
                        before = self.td_dict_face[f1][f2]
                        if before == self.td_dict_face_min[f1][f2]:
                            continue
                        else:
                            self.td_dict_face[f1][f2] = self.td_dict_face[f2][
                                f1] = min(self.td_dict_face[f1][f2],
                                          self.td_dict[idx1][idx2])

        else:
            idx1 = change_node[0][0]
            n1 = self.node_list[idx1]
            for f1 in [f for f in self.inner_facelist if n1 in f.nodes]:
                for pair in change_node:
                    idx2 = pair[1]
                    n2 = self.node_list[idx2]
                    for f2 in [
                            f for f in self.inner_facelist if n2 in f.nodes
                    ]:
                        before = self.td_dict_face[f1][f2]
                        if before == self.td_dict_face_min[f1][f2]:
                            continue
                        else:
                            self.td_dict_face[f1][f2] = self.td_dict_face[f2][
                                f1] = min(self.td_dict_face[f1][f2],
                                          self.td_dict[idx1][idx2])

        for pair in change_node:
            idx1 = pair[0]
            idx2 = pair[1]
            n1 = self.node_list[idx1]
            n2 = self.node_list[idx2]
            if n1 in self.outerface.nodes:
                for f2 in [f for f in self.inner_facelist if n2 in f.nodes]:
                    self.td_dict_face[self.outerface][f2] = min(self.td_dict_face[self.outerface][f2], self.td_dict[idx1][idx2])
            elif n2 in self.outerface.nodes:
                for f1 in [f for f in self.inner_facelist if n1 in f.nodes]:
                    self.td_dict_face[self.outerface][f1] = min(self.td_dict_face[self.outerface][f1], self.td_dict[idx1][idx2])
                
    def td_dict_update_ForPOI(self,edge):
        #print ("td_dict_update_ForPOI")
        n1 = edge.nodes[0]
        n2 = edge.nodes[1]
        idx1 = self.node_list.index(n1)
        idx2 = self.node_list.index(n2)
        self.current_road_cost = self.G[n1][n2]['weight']
        self.total_road_cost += self.current_road_cost
        change_two = False
        change_node = []



        ### update node2POInode shortest distance
        if n1 not in self.road_nodes:
            if n2 not in self.road_nodes:
                change_node.append([idx1, idx2])
                self.td_dict[idx1][idx2] = self.td_dict[idx1][idx2] = self.G[n1][n2]['weight']
                # For POI: In this case, no need to update node to POI

            else:
                change_node.append([idx1, idx1])
                for i in self.road_nodes_idx:
                    self.td_dict[i][idx1] = self.td_dict[idx1][
                        i] = self.td_dict[i][idx2] + self.G[n1][n2]['weight']
                    change_node.append([idx1, i])

                # For POI: In this case, n1 is the new node added in the network
                for POINode in self.POINodes: 
                    self.td_dict_nodeToPOInode[n1][POINode] = self.td_dict[idx1][
                        self.node_list.index(POINode)] = self.td_dict[i][idx2] + self.G[n1][n2]['weight']

                for POIEdge in self.POIEdges: 
                    if self.td_dict_nodeToPOInode[n1][POIEdge.nodes[0]] <  self.td_dict_nodeToPOInode[n1][POIEdge.nodes[1]]:
                        self.td_dict_nodeToPOIEdge[n1][POIEdge]["POINode": POIEdge.nodes[0]]
                        self.td_dict_nodeToPOIEdge[n1][POIEdge]["Dist": self.td_dict_nodeToPOInode[n1][POIEdge.nodes[0]]]
                    elif self.td_dict_nodeToPOInode[n1][POIEdge.nodes[0]] >  self.td_dict_nodeToPOInode[n1][POIEdge.nodes[1]]:
                        self.td_dict_nodeToPOIEdge[n1][POIEdge]["POINode": POIEdge.nodes[1]]
                        self.td_dict_nodeToPOIEdge[n1][POIEdge]["Dist": self.td_dict_nodeToPOInode[n1][POIEdge.nodes[1]]]

        elif n2 not in self.road_nodes:
            if n1 not in self.road_nodes:
                change_node.append([idx1, idx2])
                self.td_dict[idx1][idx2] = self.td_dict[idx1][idx2] = self.G[n1][n2]['weight']
                # In this case, no need to update node to POI

            else:
                change_node.append([idx2, idx2])
                for i in self.road_nodes_idx:
                    self.td_dict[i][idx2] = self.td_dict[idx2][
                        i] = self.td_dict[i][idx1] + self.G[n1][n2]['weight']
                    change_node.append([idx2, i])

                # For POI: In this case, n2 is the new node added in the network
                for POINode in self.POINodes: 
                    self.td_dict_nodeToPOInode[n2][POINode] = self.td_dict[idx2][
                        self.node_list.index(POINode)] = self.td_dict[i][idx1] + self.G[n1][n2]['weight']
                    
                for POIEdge in self.POIEdges: 
                    if self.td_dict_nodeToPOInode[n2][POIEdge.nodes[0]] <  self.td_dict_nodeToPOInode[n2][POIEdge.nodes[1]]:
                        self.td_dict_nodeToPOIEdge[n2][POIEdge]["POINode": POIEdge.nodes[0]]
                        self.td_dict_nodeToPOIEdge[n2][POIEdge]["Dist": self.td_dict_nodeToPOInode[n2][POIEdge.nodes[0]]]
                    elif self.td_dict_nodeToPOInode[n2][POIEdge.nodes[0]] >  self.td_dict_nodeToPOInode[n2][POIEdge.nodes[1]]:
                        self.td_dict_nodeToPOIEdge[n2][POIEdge]["POINode": POIEdge.nodes[1]]
                        self.td_dict_nodeToPOIEdge[n2][POIEdge]["Dist": self.td_dict_nodeToPOInode[n2][POIEdge.nodes[1]]]
                    
    

        else:
            change_two = True   # to indicate 2 nodes now are connected, need to update td
            for i in self.road_nodes_idx:
                for j in self.road_nodes_idx:
                    before = self.td_dict[i][j]
                    self.td_dict[i][j] = self.td_dict[j][i] = min(
                        self.td_dict[i][j],
                        (min(self.td_dict[i][idx1] + self.td_dict[idx2][j],
                             self.td_dict[i][idx2] + self.td_dict[idx1][j]) +
                         self.G[n1][n2]['weight']))
                    if self.td_dict[i][j] < before:
                        change_node.append([i, j])

            # For POI: Need to update all
            for node in self.inner_nodelist_True:
                for POINode in self.POINodes: 
                    self.td_dict_nodeToPOInode[node][POINode] = self.td_dict[self.node_list.index(node)][self.node_list.index(POINode)]
            
            for node in self.inner_nodelist_True:
                for POIEdge in self.POIEdges: 
                    if self.td_dict_nodeToPOInode[node][POIEdge.nodes[0]] <  self.td_dict_nodeToPOInode[node][POIEdge.nodes[1]]:
                        self.td_dict_nodeToPOIEdge[node][POIEdge]["POINode"] = POIEdge.nodes[0]
                        self.td_dict_nodeToPOIEdge[node][POIEdge]["Dist"] =  self.td_dict_nodeToPOInode[node][POIEdge.nodes[0]]
                    elif self.td_dict_nodeToPOInode[node][POIEdge.nodes[0]] >  self.td_dict_nodeToPOInode[node][POIEdge.nodes[1]]:
                        self.td_dict_nodeToPOIEdge[node][POIEdge]["POINode"] = POIEdge.nodes[1]
                        self.td_dict_nodeToPOIEdge[node][POIEdge]["Dist"] = self.td_dict_nodeToPOInode[node][POIEdge.nodes[1]]

        # update face2face distance
        if change_two:
            for pair in change_node:
                idx1 = pair[0]
                idx2 = pair[1]
                n1 = self.node_list[idx1]
                n2 = self.node_list[idx2]
                for f1 in [f for f in self.inner_facelist if n1 in f.nodes]:
                    for f2 in [
                            f for f in self.inner_facelist if n2 in f.nodes
                    ]:
                        before = self.td_dict_face[f1][f2]
                        if before == self.td_dict_face_min[f1][f2]:
                            continue
                        else:
                            self.td_dict_face[f1][f2] = self.td_dict_face[f2][
                                f1] = min(self.td_dict_face[f1][f2],
                                          self.td_dict[idx1][idx2])
                
                

                # For POI: 
                for changeNode in [n1, n2]:
                    for face in [f for f in self.inner_facelist_True if changeNode in f.nodes]:
                        for POIEdge in self.POIEdges:
                            allRecord = [self.td_dict_nodeToPOIEdge[node][POIEdge] for node in face.nodes]
                            allDist = [record["Dist"] for record in allRecord]
                            minDist = min(allDist)
                            selectedIndex = allDist.index(minDist)
                            self.td_dict_faceToPOIEdge[face][POIEdge]["Dist"] = minDist
                            self.td_dict_faceToPOIEdge[face][POIEdge]["POINode"] = allRecord[selectedIndex]["POINode"]
                            self.td_dict_faceToPOIEdge_TheNodePair[face][POIEdge]["POINode"] = allRecord[selectedIndex]["POINode"]
                            self.td_dict_faceToPOIEdge_TheNodePair[face][POIEdge]["node"] = face.nodes[selectedIndex]



  

        else:
            idx1 = change_node[0][0]        # This is the new added one
            n1 = self.node_list[idx1]
            for f1 in [f for f in self.inner_facelist if n1 in f.nodes]:
                for pair in change_node:
                    idx2 = pair[1]
                    n2 = self.node_list[idx2]
                    for f2 in [
                            f for f in self.inner_facelist if n2 in f.nodes
                    ]:
                        before = self.td_dict_face[f1][f2]
                        if before == self.td_dict_face_min[f1][f2]:
                            continue
                        else:
                            self.td_dict_face[f1][f2] = self.td_dict_face[f2][
                                f1] = min(self.td_dict_face[f1][f2],
                                          self.td_dict[idx1][idx2])

            # For POI: 
            for face in [f for f in self.inner_facelist_True if n1 in f.nodes]:
                for POIEdge in self.POIEdges:
                    allRecord = [self.td_dict_nodeToPOIEdge[node][POIEdge]for node in face.nodes]
                    allDist = [record["Dist"] for record in allRecord]
                    minDist = min(allDist)
                    selectedIndex = allDist.index(minDist)
                    self.td_dict_faceToPOIEdge[face][POIEdge]["Dist"] = minDist
                    self.td_dict_faceToPOIEdge[face][POIEdge]["POINode"] = allRecord[selectedIndex]["POINode"]
                    self.td_dict_faceToPOIEdge_TheNodePair[face][POIEdge]["POINode"] = allRecord[selectedIndex]["POINode"]
                    self.td_dict_faceToPOIEdge_TheNodePair[face][POIEdge]["node"] = face.nodes[selectedIndex]

        for pair in change_node:
            idx1 = pair[0]
            idx2 = pair[1]
            n1 = self.node_list[idx1]
            n2 = self.node_list[idx2]
            if n1 in self.outerface.nodes:
                for f2 in [f for f in self.inner_facelist if n2 in f.nodes]:
                    self.td_dict_face[self.outerface][f2] = min(self.td_dict_face[self.outerface][f2], self.td_dict[idx1][idx2])
            elif n2 in self.outerface.nodes:
                for f1 in [f for f in self.inner_facelist if n1 in f.nodes]:
                    self.td_dict_face[self.outerface][f1] = min(self.td_dict_face[self.outerface][f1], self.td_dict[idx1][idx2])
                


    # ok             
    def interior_parcels_update_original(self):
        #print ("interior_parcels_update")
        parcels = len(self.interior_parcels)
        # print ("parcels",parcels)
        self.interior_parcels=[]
        for f in self.inner_facelist:
            if self.td_dict_face[self.outerface][f] == 10000:
            
                    self.interior_parcels.append(f)
                    # print ("add F", f,self.td_dict_face[self.outerface][f])
                    

        for e in self.myedges():
            e.interior = False
        for p in self.interior_parcels:
            for e in p.edges:
                e.interior = True
        for n in self.G.nodes():
            mgh.is_interiornode(n, self)
        self.interior_nodes = [n for n in self.G.nodes() if n.interior]

        self.del_parcel_num = parcels - len(self.interior_parcels)
        # print ("self.interior_parcels",len(self.interior_parcels))
        # print ("del_parcel_num",self.del_parcel_num)

        # debug_interior_parcels = []
        # for f in self.inner_facelist:  #   interior_parcels  replaced by czb
        #     if len(set(f.nodes).intersection(set(self.road_nodes))) == 0:
        #         f.on_road = False
        #         debug_interior_parcels.append(f)
        #     else:
        #         f.on_road = True
        #         for n in f.nodes:
        #             n.interior = False
        # print ("debug_interior_parcels",len(debug_interior_parcels))

        # for parcel in self.interior_parcels:
        #     if parcel not in debug_interior_parcels:
        #         print (parcel)

        # if self.del_parcel_num == 0:
        #     self.plot(self.del_parcel_num)  # for debug
       
        if len(self.interior_parcels) == 0 and self.del_parcel_num != 0:
            self.full_connected_road_num = self.build_road_num
    
    
    def interior_parcels_update(self):
        # print ("interior_parcels_update")
        parcels = len(self.interior_parcels)
        # print ("parcels",parcels)
        old_interior_parcels = self.interior_parcels
        self.interior_parcels=[]
        for f in old_interior_parcels:
            if f.parcelEdgeAccessible == False:
                self.interior_parcels.append(f)
            # print ("add F", f,self.td_dict_face[self.outerface][f])
                    

        for e in self.myedges():
            e.interior = False
        for p in self.interior_parcels:
            for e in p.edges:
                e.interior = True
        for n in self.G.nodes():
            mgh.is_interiornode(n, self)
        self.interior_nodes = [n for n in self.G.nodes() if n.interior]

        self.del_parcel_num = parcels - len(self.interior_parcels)
        # print ("self.interior_parcels",len(self.interior_parcels))
        # print ("del_parcel_num",self.del_parcel_num)

        # debug_interior_parcels = []
        # for f in self.inner_facelist:  #   interior_parcels  replaced by czb
        #     if len(set(f.nodes).intersection(set(self.road_nodes))) == 0:
        #         f.on_road = False
        #         debug_interior_parcels.append(f)
        #     else:
        #         f.on_road = True
        #         for n in f.nodes:
        #             n.interior = False
        # print ("debug_interior_parcels",len(debug_interior_parcels))

        # for parcel in self.interior_parcels:
        #     if parcel not in debug_interior_parcels:
        #         print (parcel)

        # if self.del_parcel_num == 0:
        #     self.plot(self.del_parcel_num)  # for debug
       
        if len(self.interior_parcels) == 0 and self.del_parcel_num != 0:
            self.full_connected_road_num = self.build_road_num


############################
# REWARD FUNCTIONS _ NEW FOR POI
############################
###
# The first collection: using road graph
###
# The dict records each node to POI node; This is the base, all other caculations depend on it

    def td_dict_nodeToPOInode_init(self,infiniteDist = 10000):
        # Create a current road graph
        roadG = MyGraph()
        for idx in range(len(self.road_edges)):
            e = self.road_edges[idx]
            roadG.add_edge(self.road_edges[idx],weight=self.G[e.nodes[0]][e.nodes[1]]['weight'])


        ### init_td_dict_nodeToPOI
        td_dict_nodeToPOInode = {}
        for node in self.inner_nodelist_True:
            td_dict_nodeToPOInode[node] = {}
            for nodePOI in self.POINodes:   
                if node in roadG.G:            # For POI edge, use node,  # For POInode, use POI node
                    length = nx.shortest_path_length(roadG.G,source=node, target=nodePOI, weight="weight")
                else:
                    length = infiniteDist
                td_dict_nodeToPOInode[node][nodePOI] = length

        self.td_dict_nodeToPOInode = td_dict_nodeToPOInode

    # The dict records each node's ave distance to all POI nodes - [it seems useless]

    def td_dict_ave_nodeToPOInode_init(self):
        self.td_dict_ave_nodeToPOI = {}
        for n1 in self.inner_nodelist_True:
            dis=0
            count=0
            for n2 in self.POINodes:
                if n1 == n2:                # Dont need to minus (check again)
                    continue
                else:
                    dis += self.td_dict_nodeToPOInode[n1][n2]
                    count += 1
            self.td_dict_ave_nodeToPOI[n1] = dis/count     

    # The dict records each node to POI egde
    def td_dict_nodeToPOIEdge_init(self):
        ### init_td_dict_nodeToPOIEdge
        self.td_dict_nodeToPOIEdge = {}
        for node in self.inner_nodelist_True:
            self.td_dict_nodeToPOIEdge[node] = {}
            for POIEdge in self.POIEdges:
                POINode_A = POIEdge.nodes[0]
                POINode_B = POIEdge.nodes[1]
                
                if self.td_dict_nodeToPOInode[node][POINode_A] <= self.td_dict_nodeToPOInode[node][POINode_B]:
                    self.td_dict_nodeToPOIEdge[node][POIEdge] = {"POINode": POINode_A, "Dist":self.td_dict_nodeToPOInode[node][POINode_A]}
                else:
                    self.td_dict_nodeToPOIEdge[node][POIEdge] = {"POINode": POINode_B, "Dist":self.td_dict_nodeToPOInode[node][POINode_B]}


    # The dict records each face's distance to all POI Edges
    def td_dict_faceToPOIEdge_init(self,infiniteDist = 10000):
   
        self.td_dict_faceToPOIEdge = {}
        self.td_dict_faceToPOIEdge_TheNodePair = {} #[node on face, node on POIEdge]
        for f1 in self.inner_facelist_True:
            self.td_dict_faceToPOIEdge[f1] = {} 
            self.td_dict_faceToPOIEdge_TheNodePair[f1] = {}
            for POIEdge in self.POIEdges:
                self.td_dict_faceToPOIEdge[f1][POIEdge] = {"POINode": None, "Dist":infiniteDist}
                self.td_dict_faceToPOIEdge_TheNodePair[f1][POIEdge] = {"POINode":None,"node":None}
                thisFaceInfo = []
                for edge in f1.ordered_edges:
                    for node in edge.nodes:
                        thisFaceInfo.append(node.x)
                        thisFaceInfo.append(node.y)
                for node in f1.nodes:
                    if self.td_dict_faceToPOIEdge[f1][POIEdge]["Dist"] > self.td_dict_nodeToPOIEdge[node][POIEdge]["Dist"]:  
                        self.td_dict_faceToPOIEdge[f1][POIEdge] = {"POINode":self.td_dict_nodeToPOIEdge[node][POIEdge]["POINode"], "Dist":self.td_dict_nodeToPOIEdge[node][POIEdge]["Dist"] }  
                        self.td_dict_faceToPOIEdge_TheNodePair[f1][POIEdge] = {"POINode": self.td_dict_nodeToPOIEdge[node][POIEdge]["POINode"],"node":node}

    # The dict records each face's ave distance to all POI Edges
    def td_dict_ave_faceToPOIEdge_init(self):
        self.td_dict_ave_faceToPOIEdge = {}
        for f1 in self.inner_facelist_True:
            sumDist = 0
            for POIEdge in self.POIEdges:
                sumDist += self.td_dict_faceToPOIEdge[f1][POIEdge]["Dist"]
            ave = sumDist/len(self.POIEdges)    
            self.td_dict_ave_faceToPOIEdge[f1] = ave
        



    ###
    # The second collection: assume all edge it is possible 
    ###
    def td_dict_nodeToPOInode_min_init(self,infiniteDist = 10000):
        ### init_td_dict_nodeToPOI
        td_dict_nodeToPOInode_min = {}
        for node in self.inner_nodelist_True:
            td_dict_nodeToPOInode_min[node] = {}
            for nodePOI in self.POINodes:   
                if node in self.G:
                    length = nx.shortest_path_length(self.G,source=node, target=nodePOI, weight="weight")
                else:
                    length = infiniteDist
                td_dict_nodeToPOInode_min[node][nodePOI] = length

        self.td_dict_nodeToPOInode_min = td_dict_nodeToPOInode_min


    def td_dict_ave_nodeToPOInode_min_init(self):
        self.td_dict_ave_nodeToPOI_min = {}
        for n1 in self.inner_nodelist_True:
            dis=0
            count=0
            for n2 in self.POINodes:
                if n1 == n2:                # Dont need to minus (check again)
                    continue
                else:
                    dis += self.td_dict_nodeToPOInode_min[n1][n2]
                    count += 1
            self.td_dict_ave_nodeToPOI_min[n1] = dis/count     


    def td_dict_nodeToPOIEdge_min_init(self):
        ### init_td_dict_nodeToPOIEdge
        self.td_dict_nodeToPOIEdge_min = {}
        for node in self.inner_nodelist_True:
            self.td_dict_nodeToPOIEdge_min[node] = {}
            for POIEdge in self.POIEdges:
                POINode_A = POIEdge.nodes[0]
                POINode_B = POIEdge.nodes[1]
                
                if self.td_dict_nodeToPOInode_min[node][POINode_A] <= self.td_dict_nodeToPOInode_min[node][POINode_B]:
                    self.td_dict_nodeToPOIEdge_min[node][POIEdge] = {"POINode": POINode_A, "Dist":self.td_dict_nodeToPOInode_min[node][POINode_A]}
                else:
                    self.td_dict_nodeToPOIEdge_min[node][POIEdge] = {"POINode": POINode_B, "Dist":self.td_dict_nodeToPOInode_min[node][POINode_B]}

    def td_dict_faceToPOIEdge_min_init(self,infiniteDist = 10000):
        self.td_dict_faceToPOIEdge_min = {}
        self.td_dict_faceToPOIEdge_TheNodePair_min = {} #[node on face, node on POIEdge]
        for f1 in self.inner_facelist_True:
            self.td_dict_faceToPOIEdge_min[f1] = {} 
            self.td_dict_faceToPOIEdge_TheNodePair_min[f1] = {}
            for POIEdge in self.POIEdges:
                self.td_dict_faceToPOIEdge_min[f1][POIEdge] = {"POINode": None, "Dist":infiniteDist}
                self.td_dict_faceToPOIEdge_TheNodePair_min[f1][POIEdge] = {"POINode":None,"node":None}
                for node in f1.nodes:
                    if self.td_dict_faceToPOIEdge_min[f1][POIEdge]["Dist"] > self.td_dict_nodeToPOIEdge_min[node][POIEdge]["Dist"]:  
                        self.td_dict_faceToPOIEdge_min[f1][POIEdge] = {"POINode":self.td_dict_nodeToPOIEdge_min[node][POIEdge]["POINode"], "Dist":self.td_dict_nodeToPOIEdge_min[node][POIEdge]["Dist"] }  
                        self.td_dict_faceToPOIEdge_TheNodePair_min[f1][POIEdge] = {"POINode": self.td_dict_nodeToPOIEdge_min[node][POIEdge]["POINode"],"node":node}
        
 
    def td_dict_ave_faceToPOIEdge_min_init(self):
        self.td_dict_ave_faceToPOIEdge_min = {}
        for f1 in self.inner_facelist_True:
            sumDist = 0
            for POIEdge in self.POIEdges:
                sumDist += self.td_dict_faceToPOIEdge_min[f1][POIEdge]["Dist"]
            ave = sumDist/len(self.POIEdges)    
            self.td_dict_ave_faceToPOIEdge_min[f1] = ave


    # The overall function
    def td_dict_POI_Related_init(self):
        self.td_dict_nodeToPOInode_init()
        # td_dict_ave_nodeToPOInode_init(self)
        self.td_dict_nodeToPOIEdge_init()
        self.td_dict_faceToPOIEdge_init()
        self.td_dict_ave_faceToPOIEdge_init()
        self.face2POI_avg()
        self.CheckCuldesacNum()

        
        ####
        self.td_dict_nodeToPOInode_min_init()
        #self.td_dict_ave_nodeToPOInode_min_init()
        self.td_dict_nodeToPOIEdge_min_init()
        self.td_dict_faceToPOIEdge_min_init()
        self.td_dict_ave_faceToPOIEdge_min_init()
        self.face2POI_avg_min()


    def td_dict_POI_Related_init_New(self):

        #### use road network to calculate the distance
        self.td_dict_nodeToPOInode_MultiCat_init()
        self.td_dict_faceToPOInode_MultiCat_init()
        self.td_dict_faceToPOInode_EachCat_init()   #### Closest one, becasue one parcel only need to go to the closest POI for each category
        self.face2POI_EachCat_avg()
        self.face2POI_EachCat_avg_mean()
        self.CheckCuldesacNum()

        #### use all network to calculate the distance
        self.td_dict_nodeToPOInode_MultiCat_min_init()
        self.td_dict_faceToPOInode_MultiCat_min_init()
        self.td_dict_faceToPOInode_EachCat_min_init()
        self.face2POI_EachCat_avg_min()
        self.face2POI_EachCat_avg_min_mean()

        

    def face2POI_avg(self):
        ave = sum(self.td_dict_ave_faceToPOIEdge[f1] for f1 in self.inner_facelist_True) / len(self.inner_facelist_True)
        self.f2POI_avg = ave
        # self.f2POI_data.append(ave)  Pending to decide what to record
 
        return ave

    def face2POI_avg_min(self):
        ave = sum(self.td_dict_ave_faceToPOIEdge_min[f1] for f1 in self.inner_facelist_True) / len(self.inner_facelist_True)
        self.f2POI_avg_min = ave
        return ave


    def travel_distance_forPOI(self) -> float:
        if len(self.interior_parcels) or self.del_parcel_num:
            before = self.face2POI_avg()
            return 0
        else:
      
            before = self.f2POI_avg
            now = self.face2POI_avg()
            # print ("self.f2POI_avg_min",self.f2POI_avg_min)
            return  (before-now)/(before-self.f2POI_avg_min)

    def travel_distance_forPOI_New(self) -> float:
        if len(self.interior_parcels) or self.del_parcel_num:
            before = self.face2POI_EachCat_avg_mean()
            return 0
        else:
            before = self.f2POI_avg_EachCat_mean
            now = self.face2POI_EachCat_avg_mean()
     
            return  (before-now)/(before-self.f2POI_avg_EachCat_min_mean)
        
    def CheckNewRoadEdgeCase(self):
        caseTag = None
        newRoadEdge = self.road_edges[-1]

        roadNodeCollection = []
        for edge in self.road_edges:
            roadNodeCollection.append(edge.nodes[0])
            roadNodeCollection.append(edge.nodes[1])

        if roadNodeCollection.count(newRoadEdge.nodes[0]) >= 2 and roadNodeCollection.count(newRoadEdge.nodes[1]) >= 2:
            culdesacRoadNode = None
            notCuldesacRoadNode = None
            caseTag = "case3"
        else:
            if roadNodeCollection.count(newRoadEdge.nodes[0]) >= 2 and roadNodeCollection.count(newRoadEdge.nodes[1]) == 1:
                notCuldesacRoadNode = newRoadEdge.nodes[0]
                culdesacRoadNode = newRoadEdge.nodes[1]
                if roadNodeCollection.count(newRoadEdge.nodes[0]) == 2:
                    caseTag = "case1"
                elif roadNodeCollection.count(newRoadEdge.nodes[0]) > 2:
                    caseTag = "case2"

            elif roadNodeCollection.count(newRoadEdge.nodes[0]) == 1 and roadNodeCollection.count(newRoadEdge.nodes[1]) >= 2:
                notCuldesacRoadNode = newRoadEdge.nodes[1]
                culdesacRoadNode = newRoadEdge.nodes[0]
                if roadNodeCollection.count(newRoadEdge.nodes[1]) == 2:
                    caseTag = "case1"
                elif roadNodeCollection.count(newRoadEdge.nodes[1]) > 2:
                    caseTag = "case2"

        return caseTag

    def CollectCuldesacEdges(self):
        roadNodeCollection = []
        for edge in self.road_edges:
            roadNodeCollection.append(edge.nodes[0])
            roadNodeCollection.append(edge.nodes[1])
        culdesacEdges = [edge for edge in self.road_edges if roadNodeCollection.count(edge.nodes[0]) == 1 or roadNodeCollection.count(edge.nodes[1]) == 1]
        return culdesacEdges


    def CheckCuldesacNum(self):
        roadG = MyGraph()
        for idx in range(len(self.road_edges)):
            e = self.road_edges[idx]
            roadG.add_edge(self.road_edges[idx],weight=self.G[e.nodes[0]][e.nodes[1]]['weight'])

        single_neighbor_nodes = [node for node in roadG.G.nodes() if len(list(roadG.G.neighbors(node))) == 1]
        num_single_neighbor_nodes = len(single_neighbor_nodes)

        #print ("single_neighbor_nodes",single_neighbor_nodes)
        info = []
        for node in single_neighbor_nodes:
            info.append(node.x)
            info.append(node.y)
        
        #print (info)
        
        self.culdesacNum = num_single_neighbor_nodes
        return num_single_neighbor_nodes

    def CheckCuldesacNum_NotAssign(self):

        roadG = MyGraph()
        for idx in range(len(self.road_edges)):
            e = self.road_edges[idx]
            roadG.add_edge(self.road_edges[idx],weight=self.G[e.nodes[0]][e.nodes[1]]['weight'])

        single_neighbor_nodes = [node for node in roadG.G.nodes() if len(list(roadG.G.neighbors(node))) == 1]
        num_single_neighbor_nodes = len(single_neighbor_nodes)

        return num_single_neighbor_nodes

    def CuldesacReward(self) -> float:
        before = self.culdesacNum 
        now = self.CheckCuldesacNum()
        # print ("in CuldesacRewardbefore",before)
        # print ("in CuldesacRewardnow",now)

        if before == 0 or before == now:
            culdesacReward = 0
        elif now > before:
            culdesacReward = -1
        elif now < before:
            culdesacReward = 1
        return  culdesacReward  

    def ImplicitCuldesacReward_Negative(self,threhold=2) -> float:
        newEdge = self.road_edges[-1]         # it is added in the road network     

        repeatCount = 0
        roadNodeCollection = []
        for edge in self.road_edges:
            roadNodeCollection.append(edge.nodes[0])
            roadNodeCollection.append(edge.nodes[1])

        if roadNodeCollection.count(newEdge.nodes[0]) >= 2 and roadNodeCollection.count(newEdge.nodes[1]) >= 2:
            ImplicitCuldesacReward = 0
      
        else:
            if roadNodeCollection.count(newEdge.nodes[0]) >= 2 and roadNodeCollection.count(newEdge.nodes[1]) == 1:
                notCuldesacRoadNode = newEdge.nodes[0]
                culdesacRoadNode = newEdge.nodes[1]
            elif roadNodeCollection.count(newEdge.nodes[0]) == 1 and roadNodeCollection.count(newEdge.nodes[1]) >= 2:
                notCuldesacRoadNode = newEdge.nodes[1]
                culdesacRoadNode = newEdge.nodes[0]
        
            min_nodes = float('inf')
            min_Path = None
            min_target = None
       

            copyG = self.G.copy()
     

            copyG.remove_edge(culdesacRoadNode,notCuldesacRoadNode)
   
            allRoadNodes = self.road_nodes
            nodes_within_3_hops = [node for node, length in nx.single_source_shortest_path_length(copyG, culdesacRoadNode).items() if length <= 3]
            allRoadNodes = [item for item in allRoadNodes if item in nodes_within_3_hops]

            # 遍历nodelist，找到到每个节点的最短路径
            for target in allRoadNodes:
                if target == culdesacRoadNode or target == notCuldesacRoadNode: 
                    continue
                # 使用 NetworkX 的 shortest_path_length 找到最短路径长度
                try:
                    path = nx.shortest_path(copyG, source=culdesacRoadNode, target=target)
                except:
                    roadIndices = []
                    for index in range(len(self.edge_list)):
                        if self.edge_list[index] in self.road_edges:
                            roadIndices.append(index)
                    
                    print ("roadIndices",roadIndices)
                    print ("culdesacRoadNode",culdesacRoadNode)
                    print ("target",target)   

                else:
                    path = nx.shortest_path(copyG, source=culdesacRoadNode, target=target)
                # 比较并更新经过节点最少的路径
                if len(path) < min_nodes:
                    min_nodes = len(path)
                    min_Path = path
                    min_target = target

                
            # roadIndices = []
            # for index in range(len(self.edge_list)):
            #     if self.edge_list[index] in self.road_edges:
            #         roadIndices.append(index)
            
            #print ("roadIndices",roadIndices)
    
            min_nodes = min_nodes-1

            if min_nodes >=threhold:   
                ImplicitCuldesacReward = -1
            else:
                ImplicitCuldesacReward = - min_nodes/threhold

        return ImplicitCuldesacReward

    def ImplicitCuldesacReward(self,culdesacReward,threhold=2) -> float:
        
        if culdesacReward >0:
            ImplicitCuldesacReward = 0
        elif culdesacReward < 0:
            ImplicitCuldesacReward = 0
        elif culdesacReward == 0:
            newEdge = self.road_edges[-1]         # it is added in the road network     

            repeatCount = 0
            roadNodeCollection = []
            for edge in self.road_edges:
                roadNodeCollection.append(edge.nodes[0])
                roadNodeCollection.append(edge.nodes[1])

            if roadNodeCollection.count(newEdge.nodes[0]) >= 2 and roadNodeCollection.count(newEdge.nodes[1]) >= 2:
                ImplicitCuldesacReward = 0
        
            else:
                if roadNodeCollection.count(newEdge.nodes[0]) >= 2 and roadNodeCollection.count(newEdge.nodes[1]) == 1:
                    notCuldesacRoadNode = newEdge.nodes[0]
                    culdesacRoadNode = newEdge.nodes[1]
                elif roadNodeCollection.count(newEdge.nodes[0]) == 1 and roadNodeCollection.count(newEdge.nodes[1]) >= 2:
                    notCuldesacRoadNode = newEdge.nodes[1]
                    culdesacRoadNode = newEdge.nodes[0]
            
                min_nodes = float('inf')
                min_Path = None
                min_target = None
        

                copyG = self.G.copy()
        

                copyG.remove_edge(culdesacRoadNode,notCuldesacRoadNode)
    
                allRoadNodes = self.road_nodes
                nodes_within_3_hops = [node for node, length in nx.single_source_shortest_path_length(copyG, culdesacRoadNode).items() if length <= 3]
                
                
                allRoadNodes = [item for item in allRoadNodes if item in nodes_within_3_hops]

                # 遍历nodelist，找到到每个节点的最短路径
                for target in allRoadNodes:
                    if target == culdesacRoadNode or target == notCuldesacRoadNode: 
                        continue
                    # 使用 NetworkX 的 shortest_path_length 找到最短路径长度
                    try:
                        path = nx.shortest_path(copyG, source=culdesacRoadNode, target=target)
                    except:
                        roadIndices = []
                        for index in range(len(self.edge_list)):
                            if self.edge_list[index] in self.road_edges:
                                roadIndices.append(index)
                        
                        print ("roadIndices",roadIndices)
                        print ("culdesacRoadNode",culdesacRoadNode)
                        print ("target",target)   

                    else:
                        path = nx.shortest_path(copyG, source=culdesacRoadNode, target=target)
                    # 比较并更新经过节点最少的路径
                    if len(path) < min_nodes:
                        min_nodes = len(path)
                        min_Path = path
                        min_target = target

                    
                # roadIndices = []
                # for index in range(len(self.edge_list)):
                #     if self.edge_list[index] in self.road_edges:
                #         roadIndices.append(index)
                
                #print ("roadIndices",roadIndices)

                if min_nodes != float('inf'):
                    min_nodes = min_nodes-1
                    if min_nodes >=threhold:   
                        ImplicitCuldesacReward = 0
                    else:
                        ImplicitCuldesacReward =  1/min_nodes
                else:
                    ImplicitCuldesacReward = 0

        return ImplicitCuldesacReward

    def SingleParcleRingRoadPunishment_affected_byRingRoad_NotInUse(self,angleCost_Original):
        print ("SingleParcleRingRoadPunishment",angleCost_Original)
        singleParcleRingRoadPunishment = 0
        singleRingRoadTag = False

        # Select the last edge and second last edge
        newlyAddedEdge = self.road_edges[-1]
        secondLastNewlyAddedEdge = self.road_edges[-2]
        # Related parcels
        edgeIndex = self.edge_list.index(newlyAddedEdge)
        relatedParcels = self.edge_face_index[edgeIndex]

        print ("relatedParcels",relatedParcels)
        for parcel in relatedParcels:
            singleRingRoadTag = all((edge in self.road_edges) == True for edge in parcel.edges)

        print ("singleRingRoadTag",singleRingRoadTag)
        if singleRingRoadTag == False:
            return singleParcleRingRoadPunishment
        
        else:
            if angleCost_Original < 45:
                return singleParcleRingRoadPunishment
            
            else:
                # find those possible edges that are connected the second last edge
                possibleEdges = [] 
                for edge in self.edge_list:
                    if edge not in self.road_edges:
                        if edge.nodes[0] == secondLastNewlyAddedEdge.nodes[0] or edge.nodes[1] == secondLastNewlyAddedEdge.nodes[0] or edge.nodes[0] == secondLastNewlyAddedEdge.nodes[1] or edge.nodes[1] == secondLastNewlyAddedEdge.nodes[1]:
                            possibleEdges.append(edge)
                possibleEdges.append(newlyAddedEdge)

                possibleAngles = self.GetComparisionAngle(secondLastNewlyAddedEdge,possibleEdges)
                lessThan45Tag = any(angle < 45 for angle in possibleAngles)
                print ("lessThan45Tag",lessThan45Tag)
                if lessThan45Tag == True:
                    singleParcleRingRoadPunishment = -1
                    print ("nagetive singleParcleRingRoadPunishment")
                else:
                    singleParcleRingRoadPunishment = 0
        
            return singleParcleRingRoadPunishment

    def SingleParcleRingRoadPunishment(self):

        singleParcleRingRoadPunishment = 0
        singleRingRoadTag = False

        # Select the last edge and second last edge
        newlyAddedEdge = self.road_edges[-1]
        secondLastNewlyAddedEdge = self.road_edges[-2]
        # Related parcels
        edgeIndex = self.edge_list.index(newlyAddedEdge)
        relatedParcels = self.edge_face_index[edgeIndex]

        for parcel in relatedParcels:
            singleRingRoadTag = all((edge in self.road_edges) == True for edge in parcel.edges)
            if singleRingRoadTag == True:
                break
        # print ("newlyAddedEdge",newlyAddedEdge)
        # print ("relatedParcels",relatedParcels)
        # for parcel in relatedParcels:
        #     temp = [edge in self.road_edges for edge in parcel.edges]
        #     print ("temp",temp)
        #     print (all((edge in self.road_edges) == True for edge in parcel.edges))
        # print ("SingleParcleRingRoadPunishment,singleRingRoadTag",singleRingRoadTag)
        if singleRingRoadTag == False:
            return singleParcleRingRoadPunishment
        
        else:
            singleParcleRingRoadPunishment = -1
            # print ("nagetive singleParcleRingRoadPunishment")
            return singleParcleRingRoadPunishment


    def GetRingRoads(myG):
        roadG = MyGraph()
        for idx in range(len(myG.road_edges)):
            e = myG.road_edges[idx]
            roadG.add_edge(myG.road_edges[idx],weight=myG.G[e.nodes[0]][e.nodes[1]]['weight'])

        cycles = nx.minimum_cycle_basis(roadG.G)

        return cycles


    def PolygonArea(self,cycle):
        n = len(cycle)
        area = 0
        for i in range(n):
            x1, y1 = [cycle[i].x,cycle[i].y]  # Current vertex
            x2, y2 = [cycle[(i + 1) % n].x,cycle[(i + 1) % n].y]  # Next vertex (cycle wraps around)
            area += x1 * y2 - y1 * x2
        return abs(area) / 2

    def RingRoadAreaPunishment(self,percentage = 0.02):
        # record ring road
        newAddedRingRoads = []
        ringRoads = self.GetRingRoads()
        for ringRoad in ringRoads:
            if ringRoad not in self.ringRoads:
                self.ringRoads.append(ringRoad)
                newAddedRingRoads.append(ringRoad)
        
        reward = 0
        for ringRoad in newAddedRingRoads:
            if self.PolygonArea(ringRoad) < self.totalArea * percentage:
                reward = -1
        
        return reward

    def GetAngle(self,oldEdge,thisEdge):
        if oldEdge.nodes[0] == thisEdge.nodes[0]:
            mutualPt = thisEdge.nodes[0]
            nonMutualPt_New = thisEdge.nodes[1]
            nonMutualPt_Old = oldEdge.nodes[1]

        elif oldEdge.nodes[1] == thisEdge.nodes[0]:
            mutualPt = thisEdge.nodes[0]
            nonMutualPt_New = thisEdge.nodes[1]
            nonMutualPt_Old = oldEdge.nodes[0]

        elif oldEdge.nodes[0] == thisEdge.nodes[1]:
            mutualPt = thisEdge.nodes[1]
            nonMutualPt_New = thisEdge.nodes[0]
            nonMutualPt_Old = oldEdge.nodes[1]

        elif oldEdge.nodes[1] == thisEdge.nodes[1]:
            mutualPt = thisEdge.nodes[1]
            nonMutualPt_New = thisEdge.nodes[0]
            nonMutualPt_Old = oldEdge.nodes[0]  

        vec0 = [nonMutualPt_New.x -mutualPt.x, nonMutualPt_New.y - mutualPt.y]
        vec1 = [mutualPt.x - nonMutualPt_Old.x, mutualPt.y - nonMutualPt_Old.y]

        # 向量的点积
        dot_product = vec0[0] * vec1[0] + vec0[1] * vec1[1]

        length_vec0 = math.sqrt(vec0[0]**2 + vec0[1]**2)
        length_vec1 = math.sqrt(vec1[0]**2 + vec1[1]**2)


        angle_radians = math.acos(dot_product / (length_vec0 * length_vec1))
        angle_degrees = math.degrees(angle_radians)

        return angle_degrees

    def GetComparisionAngle(self, thisEdge, connectedEdges):
        angles = []
        for oldEdge in connectedEdges:
            if oldEdge == thisEdge:
                continue
           
            if oldEdge.nodes[0] == thisEdge.nodes[0]:
                mutualPt = thisEdge.nodes[0]
                nonMutualPt_New = thisEdge.nodes[1]
                nonMutualPt_Old = oldEdge.nodes[1]

            elif oldEdge.nodes[1] == thisEdge.nodes[0]:
                mutualPt = thisEdge.nodes[0]
                nonMutualPt_New = thisEdge.nodes[1]
                nonMutualPt_Old = oldEdge.nodes[0]

            elif oldEdge.nodes[0] == thisEdge.nodes[1]:
                mutualPt = thisEdge.nodes[1]
                nonMutualPt_New = thisEdge.nodes[0]
                nonMutualPt_Old = oldEdge.nodes[1]

            elif oldEdge.nodes[1] == thisEdge.nodes[1]:
                mutualPt = thisEdge.nodes[1]
                nonMutualPt_New = thisEdge.nodes[0]
                nonMutualPt_Old = oldEdge.nodes[0]  

            vec0 = [nonMutualPt_New.x -mutualPt.x, nonMutualPt_New.y - mutualPt.y]
            vec1 = [mutualPt.x - nonMutualPt_Old.x, mutualPt.y - nonMutualPt_Old.y]

            # 向量的点积
            dot_product = vec0[0] * vec1[0] + vec0[1] * vec1[1  ]

            length_vec0 = math.sqrt(vec0[0]**2 + vec0[1]**2)
            length_vec1 = math.sqrt(vec1[0]**2 + vec1[1]**2)


            angle_radians = math.acos(dot_product / (length_vec0 * length_vec1))
            angle_degrees = math.degrees(angle_radians)

            angles.append(angle_degrees)
        return angles

    def AngleReward(self):
        newEdge = self.road_edges[-1]
        connectedEdges = [edge for edge in self.road_edges if edge.nodes[0] == newEdge.nodes[0] or edge.nodes[1] == newEdge.nodes[0] or edge.nodes[0] == newEdge.nodes[1] or edge.nodes[1] == newEdge.nodes[1]]
        minAngle = None
        # print ("newEdge",newEdge)
        # print ("connectedEdges",connectedEdges)
        angles = self.GetComparisionAngle(newEdge,connectedEdges)
        
        if len(angles) == 0:
            minAngle = 0
            minAngle_remap = 0
        else:

            minAngle = min(angles)
            minAngle_remap = (minAngle - 0) / (180 - 0) * (1 - 0) + 0

            minAngle = -minAngle
            minAngle_remap = -minAngle_remap        
            # print ("minAngle",minAngle)
        return minAngle,minAngle_remap


    def AngleReward_OriginalAngle(self):
        newEdge = self.road_edges[-1]
        connectedEdges = [edge for edge in self.road_edges if edge.nodes[0] == newEdge.nodes[0] or edge.nodes[1] == newEdge.nodes[0] or edge.nodes[0] == newEdge.nodes[1] or edge.nodes[1] == newEdge.nodes[1]]
        minAngle = None
        angles = self.GetComparisionAngle(newEdge,connectedEdges)
        
        if len(angles) == 0:
            minAngle = 0
        else:
            minAngle = min(angles)
        
        return minAngle

    def AngleOfConnections(self,thisEdge):
        connectedEdges = []

        roadNodeCollection = []
        for edge in self.road_edges:
            roadNodeCollection.append(edge.nodes[0])
            roadNodeCollection.append(edge.nodes[1])       


        if roadNodeCollection.count(thisEdge.nodes[0]) == 1:
            culdesacPoint = thisEdge.nodes[0]
        
        elif roadNodeCollection.count(thisEdge.nodes[1]) == 1:
            culdesacPoint = thisEdge.nodes[1]


        for edge in self.edge_list:
            if edge not in self.road_edges:
                if edge != thisEdge:
                    if edge.nodes[0] == culdesacPoint or edge.nodes[1] == culdesacPoint:
                        connectedEdges.append(edge)
    
        minAngle = None
        angles = self.GetComparisionAngle(thisEdge,connectedEdges)

        if len(connectedEdges) == 0:
            print ("thisEdge",thisEdge)
            info = []
            for edge in self.edge_list:
                for node in edge.nodes:
                    info.append(node.x)
                    info.append(node.y)

            print (info)     
            print ("------")
            print ([self.edge_list.index(e) for e in self.road_edges])

        if angles!= []:
            minAngle = min(angles)
            minIndex = angles.index(minAngle)
            minAngleEdge = connectedEdges[minIndex]
            return minAngle, minAngleEdge, angles,connectedEdges
        else:
            return None, None, [],[]


    def ImplicitConnectReward_Negative(self,threhold=2):
        newEdge = self.road_edges[-1]    # it is added in the road network   

        roadNodeCollection = []
        for edge in self.road_edges:
            roadNodeCollection.append(edge.nodes[0])
            roadNodeCollection.append(edge.nodes[1])       

        if roadNodeCollection.count(newEdge.nodes[0]) >= 2 and roadNodeCollection.count(newEdge.nodes[1]) >= 2:      # 2 is using this edge as connection, more than 2 might be a T connection
            implicitConnectReward = -1                  # more than nodes far away     it  does not help, but this can be offset by the culdesac reward

        else:
            internalNodeCollection = []
            for parcel in self.interior_parcels:
                for node in parcel.nodes:
                    internalNodeCollection.append(node)

            if roadNodeCollection.count(newEdge.nodes[0]) >= 2 and roadNodeCollection.count(newEdge.nodes[1]) == 1:
                notCuldesacRoadNode = newEdge.nodes[0]
                culdesacRoadNode = newEdge.nodes[1]
            elif roadNodeCollection.count(newEdge.nodes[0]) == 1 and roadNodeCollection.count(newEdge.nodes[1]) >= 2:
                notCuldesacRoadNode = newEdge.nodes[1]
                culdesacRoadNode = newEdge.nodes[0]

            copyG = self.G.copy()
            copyG.remove_edge(culdesacRoadNode,notCuldesacRoadNode)

            nodes_within_1_hops = [node for node, length in nx.single_source_shortest_path_length(copyG, culdesacRoadNode).items() if length <= 1]
            nodes_within_2_hops = [node for node, length in nx.single_source_shortest_path_length(copyG, culdesacRoadNode).items() if length <= 2]
            # nodes_within_3_hops = [node for node, length in nx.single_source_shortest_path_length(copyG, culdesacRoadNode).items() if length <= 3]

            internalNodeCount = internalNodeCollection.count(culdesacRoadNode)  # it should be in 0,1 or 2
            
            if internalNodeCount == 2 :                                         # the notCuldesacRoadNode is shared by 2 interior parcels
                implicitConnectReward = - 1/5
            elif internalNodeCount == 1:                                        # the notCuldesacRoadNode is shared by 1 interior parcel
                implicitConnectReward = - 2/5
            elif internalNodeCount == 0:                                        # the notCuldesacRoadNode is not shared by any interior parcel
                if len(list(set(nodes_within_1_hops) & set(internalNodeCollection)))  > 0:    # 1 node far away
                    implicitConnectReward = - 3/5
                elif len(list(set(nodes_within_2_hops) & set(internalNodeCollection)))  > 0:   # 2 node far away
                    implicitConnectReward = -4/5
                else:
                    implicitConnectReward = -1                                  # more than nodes far away

        return implicitConnectReward

    
    def ImplicitConnectReward_save(self,connectionReward,threhold=2):

        if connectionReward < 0:    
            implicitConnectReward = 0
        else:
            newEdge = self.road_edges[-1]    # it is added in the road network   

            roadNodeCollection = []
            for edge in self.road_edges:
                roadNodeCollection.append(edge.nodes[0])
                roadNodeCollection.append(edge.nodes[1])       

            if roadNodeCollection.count(newEdge.nodes[0]) >= 2 and roadNodeCollection.count(newEdge.nodes[1]) >= 2:      # 2 is using this edge as connection, more than 2 might be a T connection
                implicitConnectReward = 0                  # the positive reward has been given in ConnectReward 

            else:
                internalNodeCollection = []
                for parcel in self.interior_parcels:
                    for node in parcel.nodes:
                        internalNodeCollection.append(node)

                if roadNodeCollection.count(newEdge.nodes[0]) >= 2 and roadNodeCollection.count(newEdge.nodes[1]) == 1:
                    notCuldesacRoadNode = newEdge.nodes[0]
                    culdesacRoadNode = newEdge.nodes[1]
                elif roadNodeCollection.count(newEdge.nodes[0]) == 1 and roadNodeCollection.count(newEdge.nodes[1]) >= 2:
                    notCuldesacRoadNode = newEdge.nodes[1]
                    culdesacRoadNode = newEdge.nodes[0]

                copyG = self.G.copy()
                copyG.remove_edge(culdesacRoadNode,notCuldesacRoadNode)

                nodes_within_1_hops = [node for node, length in nx.single_source_shortest_path_length(copyG, culdesacRoadNode).items() if length <= 1]
                # nodes_within_2_hops = [node for node, length in nx.single_source_shortest_path_length(copyG, culdesacRoadNode).items() if length <= 2]
                # nodes_within_3_hops = [node for node, length in nx.single_source_shortest_path_length(copyG, culdesacRoadNode).items() if length <= 3]

                internalNodeCount = internalNodeCollection.count(culdesacRoadNode)  # it should be in 0,1 or 2
                
                if internalNodeCount >= 2 :                                         # the notCuldesacRoadNode is shared by 2 interior parcels
                    implicitConnectReward = 1                                       # Becuase in connection reward it is -1/2, so in this case in total it becomes 1/2
                elif internalNodeCount == 1:                                        # the notCuldesacRoadNode is shared by 1 interior parcel
                    implicitConnectReward = 3/4                                     # Becuase in connection reward it is -1/2, so in this case in total it becomes 1/4
                elif internalNodeCount == 0:                                        # the notCuldesacRoadNode is not shared by any interior parcel
                    if len(list(set(nodes_within_1_hops) & set(internalNodeCollection)))  > 0:    # 1 node far away
                        implicitConnectReward = 5/8
                    # elif len(list(set(nodes_within_2_hops) & set(internalNodeCollection)))  > 0:   # 2 node far away
                    #     implicitConnectReward = 1/5
                    else:
                        implicitConnectReward = 0                                  # more than nodes far away

   
        return implicitConnectReward

    def ImplicitConnectReward(self,connectionReward,threhold=2):
        
        if connectionReward < 0:    

            newEdge = self.road_edges[-1]    # it is added in the road network   

            roadNodeCollection = []
            for edge in self.road_edges:
                roadNodeCollection.append(edge.nodes[0])
                roadNodeCollection.append(edge.nodes[1])       

            if roadNodeCollection.count(newEdge.nodes[0]) >= 2 and roadNodeCollection.count(newEdge.nodes[1]) >= 2:      # 2 is using this edge as connection, more than 2 might be a T connection
                implicitConnectReward = 0                  # the positive reward has been given in ConnectReward 
                
            else:
   
                internalNodeCollection = []
                for parcel in self.interior_parcels:
                    for node in parcel.nodes:
                        internalNodeCollection.append(node)

                if roadNodeCollection.count(newEdge.nodes[0]) >= 2 and roadNodeCollection.count(newEdge.nodes[1]) == 1:
                    notCuldesacRoadNode = newEdge.nodes[0]
                    culdesacRoadNode = newEdge.nodes[1]
                elif roadNodeCollection.count(newEdge.nodes[0]) == 1 and roadNodeCollection.count(newEdge.nodes[1]) >= 2:
                    notCuldesacRoadNode = newEdge.nodes[1]
                    culdesacRoadNode = newEdge.nodes[0]

                copyG = self.G.copy()
                copyG.remove_edge(culdesacRoadNode,notCuldesacRoadNode)

                nodes_within_1_hops = [node for node, length in nx.single_source_shortest_path_length(copyG, culdesacRoadNode).items() if length <= 1]
                # nodes_within_2_hops = [node for node, length in nx.single_source_shortest_path_length(copyG, culdesacRoadNode).items() if length <= 2]
                # nodes_within_3_hops = [node for node, length in nx.single_source_shortest_path_length(copyG, culdesacRoadNode).items() if length <= 3]

                internalNodeCount = internalNodeCollection.count(culdesacRoadNode)  # it should be in 0,1 or 2

                if internalNodeCount >= 2 :                                         # the notCuldesacRoadNode is shared by 2 interior parcels
                    implicitConnectReward = 1                                       # Becuase in connection reward it is -1/2, so in this case in total it becomes 1/2
                elif internalNodeCount == 1:                                        # the notCuldesacRoadNode is shared by 1 interior parcel
                    implicitConnectReward = 3/4                                     # Becuase in connection reward it is -1/2, so in this case in total it becomes 1/4
                elif internalNodeCount == 0:                                        # the notCuldesacRoadNode is not shared by any interior parcel
                    if len(list(set(nodes_within_1_hops) & set(internalNodeCollection)))  > 0:    # 1 node far away
                        implicitConnectReward = 5/8
                    # elif len(list(set(nodes_within_2_hops) & set(internalNodeCollection)))  > 0:   # 2 node far away
                    #     implicitConnectReward = 1/5
                    else:
                        implicitConnectReward = 0                                  # more than nodes far away

        else:
            implicitConnectReward = 0
   
        return implicitConnectReward


############################
# REWARD FUNCTIONS _ NEW FOR POI _ MultiCat
############################
    #####################
    # Didnt consider one parcel only need to access one POI for each category
    ####################
    def td_dict_nodeToPOInode_MultiCat_init(self,infiniteDist = 10000):
        # Create a current road graph
        roadG = MyGraph()
        for idx in range(len(self.road_edges)):
            e = self.road_edges[idx]
            roadG.add_edge(self.road_edges[idx],weight=self.G[e.nodes[0]][e.nodes[1]]['weight'])


        ### init_td_dict_nodeToPOI
        td_dict_nodeToPOInode_MultiCat = {}
        td_dict_nodeToPOInode_MultiCat["A"] = {}
        td_dict_nodeToPOInode_MultiCat["B"] = {}
        td_dict_nodeToPOInode_MultiCat["C"] = {}

        for node in self.inner_nodelist_True:
            td_dict_nodeToPOInode_MultiCat["A"][node] = {}
            td_dict_nodeToPOInode_MultiCat["B"][node] = {}
            td_dict_nodeToPOInode_MultiCat["C"][node] = {}
            for nodePOI in self.POINodes:   
                if nodePOI in roadG.G:
                    length = nx.shortest_path_length(roadG.G,source=node, target=nodePOI, weight="weight")
                else:
                    length = infiniteDist
                
                td_dict_nodeToPOInode_MultiCat[nodePOI.POI_Cat][node][nodePOI] = length

        self.td_dict_nodeToPOInode_MultiCat = td_dict_nodeToPOInode_MultiCat

    def td_dict_faceToPOInode_MultiCat_init(self,infiniteDist = 10000):
        self.td_dict_faceToPOInode_MultiCat = {}
        self.td_dict_faceToPOInode_MultiCat["A"] = {}
        self.td_dict_faceToPOInode_MultiCat["B"] = {}
        self.td_dict_faceToPOInode_MultiCat["C"] = {}        

        for f1 in self.inner_facelist_True:
            self.td_dict_faceToPOInode_MultiCat["A"][f1] = {}
            self.td_dict_faceToPOInode_MultiCat["B"][f1] = {}
            self.td_dict_faceToPOInode_MultiCat["C"][f1] = {}
            for nodePOI in self.POINodes:
                self.td_dict_faceToPOInode_MultiCat["A"][f1][nodePOI] = {"Node": None, "Dist":infiniteDist}
                self.td_dict_faceToPOInode_MultiCat["B"][f1][nodePOI] = {"Node": None, "Dist":infiniteDist}
                self.td_dict_faceToPOInode_MultiCat["C"][f1][nodePOI] = {"Node": None, "Dist":infiniteDist}
                for node in f1.nodes:
                    if nodePOI.POI_Cat == "A":
                        
                        if self.td_dict_faceToPOInode_MultiCat["A"][f1][nodePOI]["Dist"] > self.td_dict_nodeToPOInode_MultiCat["A"][node][nodePOI]:  
                            self.td_dict_faceToPOInode_MultiCat["A"][f1][nodePOI] = {"Node":node, "Dist":self.td_dict_nodeToPOInode_MultiCat["A"][node][nodePOI]}    # record the node in the face
             
                    if nodePOI.POI_Cat == "B":
                      
                        if self.td_dict_faceToPOInode_MultiCat["B"][f1][nodePOI]["Dist"] > self.td_dict_nodeToPOInode_MultiCat["B"][node][nodePOI]:  
                            self.td_dict_faceToPOInode_MultiCat["B"][f1][nodePOI] = {"Node":node, "Dist":self.td_dict_nodeToPOInode_MultiCat["B"][node][nodePOI]}    # record the node in the face

                    if nodePOI.POI_Cat == "C":
                       
                        if self.td_dict_faceToPOInode_MultiCat["C"][f1][nodePOI]["Dist"] > self.td_dict_nodeToPOInode_MultiCat["C"][node][nodePOI]:  
                            self.td_dict_faceToPOInode_MultiCat["C"][f1][nodePOI] = {"Node":node, "Dist":self.td_dict_nodeToPOInode_MultiCat["C"][node][nodePOI]}    # record the node in the face
    
    # Not in use, becasue consider one parcel only need to access one POI for each category
    def td_dict_ave_faceToPOInode_MultiCat_init(self):
        self.td_dict_ave_faceToPOInode_MultiCat = {}
        self.td_dict_ave_faceToPOInode_MultiCat["A"] = {}
        self.td_dict_ave_faceToPOInode_MultiCat["B"] = {}
        self.td_dict_ave_faceToPOInode_MultiCat["C"] = {}

        for f1 in self.inner_facelist_True:

            sumDist_A = 0
            sumDist_B = 0
            sumDist_C = 0

            sumPOI_A = 0
            sumPOI_B = 0
            sumPOI_C = 0

        
            for nodePOI in self.POINodes:
                if nodePOI.POI_Cat == "A":
                    sumDist_A += self.td_dict_faceToPOInode_MultiCat["A"][f1][nodePOI]["Dist"]
                    sumPOI_A += 1

                if nodePOI.POI_Cat == "B":
                    sumDist_B += self.td_dict_faceToPOInode_MultiCat["B"][f1][nodePOI]["Dist"]
                    sumPOI_B += 1

                if nodePOI.POI_Cat == "C":
                    sumDist_C += self.td_dict_faceToPOInode_MultiCat["C"][f1][nodePOI]["Dist"]
                    sumPOI_C += 1

         
            ave_A = sumDist_A/sumPOI_A   
            self.td_dict_ave_faceToPOInode_MultiCat["A"][f1] = ave_A
            ave_B = sumDist_B/sumPOI_B   
            self.td_dict_ave_faceToPOInode_MultiCat["B"][f1] = ave_B
            ave_C = sumDist_C/sumPOI_C   
            self.td_dict_ave_faceToPOInode_MultiCat["C"][f1] = ave_C
    # Not in use, becasue consider one parcel only need to access one POI for each category
    def face2POI_MultiCat_avg(self):

        ave_A = sum(self.td_dict_ave_faceToPOInode_MultiCat["A"][f1] for f1 in self.inner_facelist_True) / len(self.inner_facelist_True)
        self.f2POI_avg_MultiCat_A = ave_A
        ave_B = sum(self.td_dict_ave_faceToPOInode_MultiCat["B"][f1] for f1 in self.inner_facelist_True) / len(self.inner_facelist_True)
        self.f2POI_avg_MultiCat_B = ave_B
        ave_C = sum(self.td_dict_ave_faceToPOInode_MultiCat["C"][f1] for f1 in self.inner_facelist_True) / len(self.inner_facelist_True)
        self.f2POI_avg_MultiCat_C = ave_C
        
        return ave_A,ave_B,ave_C

    def td_dict_nodeToPOInode_MultiCat_min_init(self,infiniteDist = 10000):
        ### init_td_dict_nodeToPOI
        td_dict_nodeToPOInode_MultiCat_min = {}
        td_dict_nodeToPOInode_MultiCat_min["A"] = {}
        td_dict_nodeToPOInode_MultiCat_min["B"] = {}
        td_dict_nodeToPOInode_MultiCat_min["C"] = {}

        for node in self.inner_nodelist_True:
            td_dict_nodeToPOInode_MultiCat_min["A"][node] = {}
            td_dict_nodeToPOInode_MultiCat_min["B"][node] = {}
            td_dict_nodeToPOInode_MultiCat_min["C"][node] = {}
            for nodePOI in self.POINodes:   
                if node in self.G:
                    length = nx.shortest_path_length(self.G,source=node, target=nodePOI, weight="weight")
                else:
                    length = infiniteDist
                
                td_dict_nodeToPOInode_MultiCat_min[nodePOI.POI_Cat][node][nodePOI] = length

        self.td_dict_nodeToPOInode_MultiCat_min = td_dict_nodeToPOInode_MultiCat_min

    def td_dict_faceToPOInode_MultiCat_min_init(self,infiniteDist = 10000):
        self.td_dict_faceToPOInode_MultiCat_min = {}
        self.td_dict_faceToPOInode_MultiCat_min["A"] = {}
        self.td_dict_faceToPOInode_MultiCat_min["B"] = {}
        self.td_dict_faceToPOInode_MultiCat_min["C"] = {}        

        for f1 in self.inner_facelist_True:
            self.td_dict_faceToPOInode_MultiCat_min["A"][f1] = {}
            self.td_dict_faceToPOInode_MultiCat_min["B"][f1] = {}
            self.td_dict_faceToPOInode_MultiCat_min["C"][f1] = {}
            for nodePOI in self.POINodes:
                self.td_dict_faceToPOInode_MultiCat_min["A"][f1][nodePOI] = {"Node": None, "Dist":infiniteDist}
                self.td_dict_faceToPOInode_MultiCat_min["B"][f1][nodePOI] = {"Node": None, "Dist":infiniteDist}
                self.td_dict_faceToPOInode_MultiCat_min["C"][f1][nodePOI] = {"Node": None, "Dist":infiniteDist}
                for node in f1.nodes:
                    if nodePOI.POI_Cat == "A":
                        
                        if self.td_dict_faceToPOInode_MultiCat_min["A"][f1][nodePOI]["Dist"] > self.td_dict_nodeToPOInode_MultiCat_min["A"][node][nodePOI]:  
                            self.td_dict_faceToPOInode_MultiCat_min["A"][f1][nodePOI] = {"Node":node, "Dist":self.td_dict_nodeToPOInode_MultiCat_min["A"][node][nodePOI]}    # record the node in the face
             
                    if nodePOI.POI_Cat == "B":
                
                        if self.td_dict_faceToPOInode_MultiCat_min["B"][f1][nodePOI]["Dist"] > self.td_dict_nodeToPOInode_MultiCat_min["B"][node][nodePOI]:  
                            self.td_dict_faceToPOInode_MultiCat_min["B"][f1][nodePOI] = {"Node":node, "Dist":self.td_dict_nodeToPOInode_MultiCat_min["B"][node][nodePOI]}    # record the node in the face

                    if nodePOI.POI_Cat == "C":
                     
                        if self.td_dict_faceToPOInode_MultiCat_min["C"][f1][nodePOI]["Dist"] > self.td_dict_nodeToPOInode_MultiCat_min["C"][node][nodePOI]:  
                            self.td_dict_faceToPOInode_MultiCat_min["C"][f1][nodePOI] = {"Node":node, "Dist":self.td_dict_nodeToPOInode_MultiCat_min["C"][node][nodePOI]}    # record the node in the face
    
    def td_dict_ave_faceToPOInode_MultiCat_min_init(self):
        self.td_dict_ave_faceToPOInode_MultiCat_min = {}
        self.td_dict_ave_faceToPOInode_MultiCat_min["A"] = {}
        self.td_dict_ave_faceToPOInode_MultiCat_min["B"] = {}
        self.td_dict_ave_faceToPOInode_MultiCat_min["C"] = {}

        for f1 in self.inner_facelist_True:

            sumDist_A = 0
            sumDist_B = 0
            sumDist_C = 0

            sumPOI_A = 0
            sumPOI_B = 0
            sumPOI_C = 0

            for nodePOI in self.POINodes:
                if nodePOI.POI_Cat == "A":
                    sumDist_A += self.td_dict_faceToPOInode_MultiCat_min["A"][f1][nodePOI]["Dist"]
                    sumPOI_A += 1

                if nodePOI.POI_Cat == "B":
                    sumDist_B += self.td_dict_faceToPOInode_MultiCat_min["B"][f1][nodePOI]["Dist"]
                    sumPOI_B += 1

                if nodePOI.POI_Cat == "C":
                    sumDist_C += self.td_dict_faceToPOInode_MultiCat_min["C"][f1][nodePOI]["Dist"]
                    sumPOI_C += 1

         
            ave_A = sumDist_A/sumPOI_A   
            self.td_dict_ave_faceToPOInode_MultiCat_min["A"][f1] = ave_A
            ave_B = sumDist_B/sumPOI_B   
            self.td_dict_ave_faceToPOInode_MultiCat_min["B"][f1] = ave_B
            ave_C = sumDist_C/sumPOI_C   
            self.td_dict_ave_faceToPOInode_MultiCat_min["C"][f1] = ave_C
    
    def face2POI_MultiCat_min_avg(self):

        ave_A_min = sum(self.td_dict_ave_faceToPOInode_MultiCat_min["A"][f1] for f1 in self.inner_facelist_True) / len(self.inner_facelist_True)
        self.f2POI_avg_MultiCat_A_min = ave_A_min
        ave_B_min = sum(self.td_dict_ave_faceToPOInode_MultiCat_min["B"][f1] for f1 in self.inner_facelist_True) / len(self.inner_facelist_True)
        self.f2POI_avg_MultiCat_B_min = ave_B_min
        ave_C_min = sum(self.td_dict_ave_faceToPOInode_MultiCat_min["C"][f1] for f1 in self.inner_facelist_True) / len(self.inner_facelist_True)
        self.f2POI_avg_MultiCat_C_min = ave_C_min
        
        return ave_A_min,ave_B_min,ave_C_min


    #####################
    # Consider one parcel only need to access one POI for each category
    ####################
    def td_dict_faceToPOInode_EachCat_init(self,infiniteDist = 10000):
        print ("td_dict_faceToPOInode_EachCat_init")
        self.td_dict_faceToPOInode_EachCat = {}
        self.td_dict_faceToPOInode_EachCat["A"] = {}
        self.td_dict_faceToPOInode_EachCat["B"] = {}
        self.td_dict_faceToPOInode_EachCat["C"] = {}    

        for f1 in self.inner_facelist_True:
            self.td_dict_faceToPOInode_EachCat["A"][f1] = {"POINode": None, "Node": None, "Dist": infiniteDist}
            self.td_dict_faceToPOInode_EachCat["B"][f1] = {"POINode": None, "Node": None, "Dist": infiniteDist}
            self.td_dict_faceToPOInode_EachCat["C"][f1] = {"POINode": None, "Node": None, "Dist": infiniteDist}

            for nodePOI in self.td_dict_faceToPOInode_MultiCat["A"][f1].keys():
                if self.td_dict_faceToPOInode_EachCat["A"][f1]["Dist"] > self.td_dict_faceToPOInode_MultiCat["A"][f1][nodePOI]["Dist"]:
                    self.td_dict_faceToPOInode_EachCat["A"][f1]["Dist"] = self.td_dict_faceToPOInode_MultiCat["A"][f1][nodePOI]["Dist"]
                    self.td_dict_faceToPOInode_EachCat["A"][f1]["POINode"] = nodePOI
                    self.td_dict_faceToPOInode_EachCat["A"][f1]["Node"]  = self.td_dict_faceToPOInode_MultiCat["A"][f1][nodePOI]["Node"]

            for nodePOI in self.td_dict_faceToPOInode_MultiCat["B"][f1].keys():
                if self.td_dict_faceToPOInode_EachCat["B"][f1]["Dist"] > self.td_dict_faceToPOInode_MultiCat["B"][f1][nodePOI]["Dist"]:
                    self.td_dict_faceToPOInode_EachCat["B"][f1]["Dist"] = self.td_dict_faceToPOInode_MultiCat["B"][f1][nodePOI]["Dist"]
                    self.td_dict_faceToPOInode_EachCat["B"][f1]["POINode"] = nodePOI
                    self.td_dict_faceToPOInode_EachCat["B"][f1]["Node"]  = self.td_dict_faceToPOInode_MultiCat["B"][f1][nodePOI]["Node"]
            
            for nodePOI in self.td_dict_faceToPOInode_MultiCat["C"][f1].keys():
                if self.td_dict_faceToPOInode_EachCat["C"][f1]["Dist"] > self.td_dict_faceToPOInode_MultiCat["C"][f1][nodePOI]["Dist"]:
                    self.td_dict_faceToPOInode_EachCat["C"][f1]["Dist"] = self.td_dict_faceToPOInode_MultiCat["C"][f1][nodePOI]["Dist"]
                    self.td_dict_faceToPOInode_EachCat["C"][f1]["POINode"] = nodePOI
                    self.td_dict_faceToPOInode_EachCat["C"][f1]["Node"]  = self.td_dict_faceToPOInode_MultiCat["C"][f1][nodePOI]["Node"]

    # Consider one parcel only need to access one POI for each category
    def face2POI_EachCat_avg(self):
        ave_A = sum(self.td_dict_faceToPOInode_EachCat["A"][f1]["Dist"] for f1 in self.inner_facelist_True) / len(self.inner_facelist_True)
        self.f2POI_avg_EachCat_A = ave_A
        ave_B = sum(self.td_dict_faceToPOInode_EachCat["B"][f1]["Dist"] for f1 in self.inner_facelist_True) / len(self.inner_facelist_True)
        self.f2POI_avg_EachCat_B = ave_B
        ave_C = sum(self.td_dict_faceToPOInode_EachCat["C"][f1]["Dist"] for f1 in self.inner_facelist_True) / len(self.inner_facelist_True)
        self.f2POI_avg_EachCat_C = ave_C

        self.POI_A_data.append(ave_A)
        self.POI_B_data.append(ave_B)
        self.POI_C_data.append(ave_C)
        
    def face2POI_EachCat_avg_mean(self,wA=1,wB=1,wC=1):
        self.f2POI_avg_EachCat_mean = wA * self.f2POI_avg_EachCat_A + wB * self.f2POI_avg_EachCat_B + wC * self.f2POI_avg_EachCat_C
        self.POI_Mean_data.append(self.f2POI_avg_EachCat_mean)
        return self.f2POI_avg_EachCat_mean

    def td_dict_faceToPOInode_EachCat_min_init(self,infiniteDist = 10000):
        self.td_dict_faceToPOInode_EachCat_min = {}
        self.td_dict_faceToPOInode_EachCat_min["A"] = {}
        self.td_dict_faceToPOInode_EachCat_min["B"] = {}
        self.td_dict_faceToPOInode_EachCat_min["C"] = {}        

        for f1 in self.inner_facelist_True:
            self.td_dict_faceToPOInode_EachCat_min["A"][f1] = {"POINode": None, "Node": None, "Dist": infiniteDist}
            self.td_dict_faceToPOInode_EachCat_min["B"][f1] = {"POINode": None, "Node": None, "Dist": infiniteDist}
            self.td_dict_faceToPOInode_EachCat_min["C"][f1] = {"POINode": None, "Node": None, "Dist": infiniteDist}

            for nodePOI in self.td_dict_faceToPOInode_MultiCat_min["A"][f1].keys():
                if self.td_dict_faceToPOInode_EachCat_min["A"][f1]["Dist"] > self.td_dict_faceToPOInode_MultiCat_min["A"][f1][nodePOI]["Dist"]:
                    self.td_dict_faceToPOInode_EachCat_min["A"][f1]["Dist"] = self.td_dict_faceToPOInode_MultiCat_min["A"][f1][nodePOI]["Dist"]
                    self.td_dict_faceToPOInode_EachCat_min["A"][f1]["POINode"] = nodePOI
                    self.td_dict_faceToPOInode_EachCat_min["A"][f1]["Node"]  = self.td_dict_faceToPOInode_MultiCat_min["A"][f1][nodePOI]["Node"]

            for nodePOI in self.td_dict_faceToPOInode_MultiCat_min["B"][f1].keys():
                if self.td_dict_faceToPOInode_EachCat_min["B"][f1]["Dist"] > self.td_dict_faceToPOInode_MultiCat_min["B"][f1][nodePOI]["Dist"]:
                    self.td_dict_faceToPOInode_EachCat_min["B"][f1]["Dist"] = self.td_dict_faceToPOInode_MultiCat_min["B"][f1][nodePOI]["Dist"]
                    self.td_dict_faceToPOInode_EachCat_min["B"][f1]["POINode"] = nodePOI
                    self.td_dict_faceToPOInode_EachCat_min["B"][f1]["Node"]  = self.td_dict_faceToPOInode_MultiCat_min["B"][f1][nodePOI]["Node"]
            
            for nodePOI in self.td_dict_faceToPOInode_MultiCat_min["C"][f1].keys():
                if self.td_dict_faceToPOInode_EachCat_min["C"][f1]["Dist"] > self.td_dict_faceToPOInode_MultiCat_min["C"][f1][nodePOI]["Dist"]:
                    self.td_dict_faceToPOInode_EachCat_min["C"][f1]["Dist"] = self.td_dict_faceToPOInode_MultiCat_min["C"][f1][nodePOI]["Dist"]
                    self.td_dict_faceToPOInode_EachCat_min["C"][f1]["POINode"] = nodePOI
                    self.td_dict_faceToPOInode_EachCat_min["C"][f1]["Node"]  = self.td_dict_faceToPOInode_MultiCat_min["C"][f1][nodePOI]["Node"]

    def face2POI_EachCat_avg_min(self):
        ave_A = sum(self.td_dict_faceToPOInode_EachCat_min["A"][f1]["Dist"] for f1 in self.inner_facelist_True) / len(self.inner_facelist_True)
        self.f2POI_avg_EachCat_A_min= ave_A
        ave_B = sum(self.td_dict_faceToPOInode_EachCat_min["B"][f1]["Dist"] for f1 in self.inner_facelist_True) / len(self.inner_facelist_True)
        self.f2POI_avg_EachCat_B_min = ave_B
        ave_C = sum(self.td_dict_faceToPOInode_EachCat_min["C"][f1]["Dist"] for f1 in self.inner_facelist_True) / len(self.inner_facelist_True)
        self.f2POI_avg_EachCat_C_min = ave_C



    def face2POI_EachCat_avg_min_mean(self,wA=1,wB=1,wC=1):
        self.f2POI_avg_EachCat_min_mean = wA * self.f2POI_avg_EachCat_A_min + wB * self.f2POI_avg_EachCat_B_min + wC * self.f2POI_avg_EachCat_C_min 
        
        return self.f2POI_avg_EachCat_min_mean

###################################
#      PLOTTING FUNCTIONS
###################################

    def plot(self, **kwargs): # New to replace

        plt.axes().set_aspect(aspect=1)
        plt.axis('off')
        edge_kwargs = kwargs.copy()
        nlocs = self.location_dict()
        edge_kwargs['label'] = "_nolegend"
        edge_kwargs['pos'] = nlocs

        pos = nx.spring_layout(self.G)
        nx.draw_networkx_edges(self.G, pos,edge_color='g',width=3)
        node_kwargs = kwargs.copy()
        node_kwargs['label'] = self.name
        node_kwargs['pos'] = nlocs
        nodes = nx.draw_networkx_nodes(self.G, pos,node_color='g')
        nodes.set_edgecolor('None')

    def plot_roads(self,
                    master=None,
                    update=False,
                    parcel_labels=False,
                    title="",
                    new_plot=True,
                    new_road_color="blue",
                    new_road_width=1.5,
                    old_node_size=25,
                    old_road_width=6,
                    barriers=True,
                    base_width=1,
                    stage=0):
            
            plt.figure(figsize=(10, 10))  
            plt.axes().set_aspect(aspect=1)
            plt.axis('off')

            nlocs = self.location_dict()

            if update:
                # self.define_roads()
                # self.define_interior_parcels()
                pass

            # if new_plot:
            #     plt.figure()

            edge_colors = [
                'blue' if e in self.road_edges and e not in self.stage2edges 
                else 'purple' if e in self.stage2edges     # it was orange
                else 'red' if e.interior 
                else 'green'
                for e in self.myedges()
            ]

            edge_width = [
                1.5 * new_road_width if e.road else 1.5 *
                new_road_width if e.barrier else 1.5 *
                new_road_width if e.interior else 1 for e in self.myedges()
            ]

            node_colors = [
                'blue' if n.road else
                'green' if n.barrier else 'red' if n.interior else 'black'
                for n in self.G.nodes()
            ]

            node_sizes = [
                1.4
                for n in self.G.nodes()
            ]

            # plot current graph
            nx.draw(self.G,
                            pos=nlocs,
                            with_labels=False,
                            node_size=node_sizes,
                            #  node_color=node_colors,
                            edge_color=edge_colors,
                            width=edge_width)
        
            # plot original roads
            if master:
                copy = master.copy()
                noffroad = [n for n in copy.G.nodes() if not n.road]
                for n in noffroad:
                    copy.G.remove_node(n)
                eoffroad = [e for e in copy.myedges() if not e.road]
                for e in eoffroad:
                    copy.G.remove_edge(e.nodes[0], e.nodes[1])

                # nx.draw_networkx(copy.G, pos=nlocs, with_labels=False,
                #                  node_size=old_node_size, node_color='black',
                #                  edge_color='black', width=old_road_width)
            
            #plt.show()


    def plot_roads_and_POIs(self,
                    master=None,
                    update=False,
                    parcel_labels=False,
                    title="",
                    new_plot=True,
                    new_road_color="blue",
                    new_road_width=1.5,
                    old_node_size=25,
                    old_road_width=6,
                    barriers=True,
                    base_width=1,
                    stage=0):
            
            plt.figure(figsize=(10, 10))  
            plt.axes().set_aspect(aspect=1)
            plt.axis('off')

            nlocs = self.location_dict()

            if update:
                # self.define_roads()
                # self.define_interior_parcels()
                pass

            # if new_plot:
            #     plt.figure()

            edge_colors = [
                'blue' if e in self.road_edges and e not in self.stage2edges 
                else 'purple' if e in self.stage2edges    # it was orange
                else 'red' if e.interior 
                else 'green'
                for e in self.myedges()
            ]

            edge_width = [
                1.5 * new_road_width if e.road else 1.5 *
                new_road_width if e.barrier else 1.5 *
                new_road_width if e.interior else 1 for e in self.myedges()
            ]

            node_colors = []
            for n in self.G.nodes():
        
                if n.POI_Cat=="A":
                    node_colors.append ('blue')
                elif n.POI_Cat=="B":
                    node_colors.append ('purple')
                elif n.POI_Cat=="C":
                    node_colors.append ('magenta')    
                else:
                    node_colors.append ('black')
            
            node_sizes = []
            for n in self.G.nodes():
                if n.isPOI:
                    node_sizes.append(20)
                else:
                    node_sizes.append(1.4)

         

            # plot current graph
            nx.draw(self.G,
                            pos=nlocs,
                            with_labels=False,
                            node_size=node_sizes,
                            node_color=node_colors,
                            edge_color=edge_colors,
                            width=edge_width)
        
            # plot original roads
            if master:
                copy = master.copy()
                noffroad = [n for n in copy.G.nodes() if not n.road]
                for n in noffroad:
                    copy.G.remove_node(n)
                eoffroad = [e for e in copy.myedges() if not e.road]
                for e in eoffroad:
                    copy.G.remove_edge(e.nodes[0], e.nodes[1])

                # nx.draw_networkx(copy.G, pos=nlocs, with_labels=False,
                #                  node_size=old_node_size, node_color='black',
                #                  edge_color='black', width=old_road_width)
            
            #plt.show()


    def plot_all_paths(self, all_paths, update=False):
        """ plots the shortest paths from all interior parcels to the road.
        Optional to update road geometery based on changes in network geometry.
        """

        plt.figure()
        if len(all_paths) == 0:
            self.plot_roads(update=update)
        else:
            Gs = []
            for p in all_paths:
                G = nx.subgraph(self.G, p)
                Gs.append(G)
            Gpaths = nx.compose_all(Gs, name="shortest paths")
            myGpaths = MyGraph(Gpaths)
            self.plot_roads(update=update)
            myGpaths.plot(edge_color='purple', width=6, node_size=1)

    def plot_weak_duals(self,
                        stack=None,
                        colors=None,
                        width=None,
                        node_size=None):
        """Given a list of weak dual graphs, plots them all. Has default colors
        node size, and line widths, but these can be added as lists."""

        if stack is None:
            duals = self.stacked_duals()
        else:
            duals = stack

        if colors is None:
            colors = [
                'grey', 'black', 'blue', 'purple', 'red', 'orange', 'yellow'
            ]
        else:
            colors = colors

        if width is None:
            width = [0.5, 0.75, 1, 1.75, 2.25, 3, 3.5]
        else:
            width = width

        if node_size is None:
            node_size = [0.5, 6, 9, 12, 17, 25, 30]
        else:
            node_size = node_size

        if len(duals) > len(colors):
            warnings.warn("too many dual graphs to draw. simplify fig," +
                          " or add more colors")

        plt.figure()

        for i in range(0, len(duals)):
            for j in duals[i]:
                j.plot(node_size=node_size[i],
                       node_color=colors[i],
                       edge_color=colors[i],
                       width=width[i])
                # print "color = {0}, node_size = {1}, width = {2}".format(
                #       colors[i], node_size[i], width[i])

        plt.axes().set_aspect(aspect=1)
        plt.axis('off')

    def snapshot(self):
            return self.G


###################################
#      PLOTTING FUNCTIONS _ NEW
###################################
    def PlotMyGraph(myG,showInternal = False):
        plt.figure(figsize=(10, 10))
        plt.axes().set_aspect(aspect=1)
        plt.axis('off')

        nlocs = myG.location_dict()

        ### basic layer
        edge_colors = []
        edges_to_draw = []
        for edge in myG.myedges():
            if edge.fake!=True:
                edge_colors.append('grey')
                edges_to_draw.append((edge.nodes[0], edge.nodes[1]))

        nx.draw_networkx_edges(myG.G, nlocs,edgelist=edges_to_draw, edge_color=edge_colors, width=1)
        nx.draw_networkx_nodes(myG.G, nlocs, node_color='black', node_size=1)

        edge_colors = []
        edges_to_draw = []
        for edge in myG.myedges():
            if edge.fake==True:
                edge_colors.append('grey')
                edges_to_draw.append((edge.nodes[0], edge.nodes[1]))

        dash_lines = []
        for edge, color in zip(edges_to_draw, edge_colors):
            line = [(nlocs[edge[0]][0], nlocs[edge[0]][1]), (nlocs[edge[1]][0], nlocs[edge[1]][1])]
            dash_lines.append(line)

        lc = LineCollection(dash_lines, linestyle='dashed', linewidth=1.5, colors=edge_colors)
        plt.gca().add_collection(lc)

        ### inaccessible layer
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

        

        ### info layer
        edge_colors = []
        edges_to_draw = []
        for edge in myG.myedges():
            if edge.isRoad and edge.isPOI!=True and edge.fake!=True:
                edge_colors.append('blue')
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
        plt.show()

    def PlotF2FDist(myG,vmin = 0,vmax = 10):
        n = len(myG.inner_facelist_True)
        matrix = np.zeros((n, n))

        for i, face_i in enumerate(myG.inner_facelist_True):
            for j, face_j in enumerate(myG.inner_facelist_True):
                matrix[i, j] = myG.td_dict_face[face_i][face_j]

        plt.imshow(matrix, cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Distance (m)')
        plt.show()

    def PlotF2POIDist(myG,vmin = 0,vmax = 10):
        m = len(myG.POIEdges)
        n = len(myG.inner_facelist_True)
        matrix = np.zeros((m, n))
        
        for i, POI_i in enumerate(myG.POIEdges):
            for j, face_j in enumerate(myG.inner_facelist_True):
                matrix[i, j] = myG.td_dict_faceToPOIEdge[face_j][POI_i]["Dist"]

        plt.xticks(ticks=np.arange(n), labels=[str(face) for face in myG.inner_facelist_True], rotation=90,fontsize=3)
        plt.yticks(ticks=np.arange(m), labels=[str(poi) for poi in myG.POIEdges],fontsize=3)

        plt.imshow(matrix, cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Distance (m)')
        plt.show()


############################
# Selection for Shortcut
############################
    def AddShortCutInGraph(myG):
        existingNodes = [node for node in myG.G.nodes()]
        
        for edge in myG.shortcutEdges:
            myG.add_edge(edge)

        myG.edge_list = myG.myedges()

        myG.node_list = []

        
        for n in myG.G.nodes():
            myG.node_list.append(n)
            if n not in existingNodes:
                n.shortCutNode = True

        myG.max_road_num = len(myG.edge_list) - len(myG.road_edges)

        myG.max_road_cost = max([
            myG.G[e.nodes[0]][e.nodes[1]]['weight'] for e in myG.myedges()
            if not e.road
        ])


        full_connected_road_num = 0
        for edge in myG.edge_list:
            if edge.internal == True and edge.isRoad == True:
                full_connected_road_num+=1


        myG.full_connected_road_num = full_connected_road_num


###################################
#      PRINT GEO INFO
###################################
    def PrintInnerFaceGeo(myG):
        allEdgeNodes = []
        for face in myG.inner_facelist:
            for edge in face.edges:
                allEdgeNodes.append(edge.nodes)
        print (allEdgeNodes)

##########
## Debug
##########

    def Debug_CheckRepeatedEdges(myG,myEdgeDict):
        edgeIDList = list(myEdgeDict.keys())
        edgeValueList = list(myEdgeDict.values())

        existIDs = []
        for edge in myG.myedges():
            existIDs.append(edgeIDList[edgeValueList.index(edge)])

    

if __name__ == "__main__":
    master = mgh.testGraphLattice(4)
    S0 = master.copy()
    S0.define_roads()
    S0.define_interior_parcels()

    ##### A example of add a road and update interior parcels ##### 
    # road_edge = S0.myedges()[1]
    # S0.add_road_segment(road_edge)
    # S0.define_interior_parcels()

    mgh.test_dual(S0)
    plt.show()
