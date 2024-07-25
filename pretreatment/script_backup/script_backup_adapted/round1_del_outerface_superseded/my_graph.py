import sys
sys.path.append(r"C:\Users\asdbe\OneDrive\Documents\GitHub\road-planning-for-slums")

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
#import plotly.plotly as py
#from plotly.graph_objs import *

class MyNode(object):
    """ rounds float nodes to (2!) decimal places, defines equality """

    def __init__(self, locarray, significant_figs = 2,name=None):
        
        if len(locarray) != 2:
            print("error")
        x = locarray[0]
        y = locarray[1]
        self.x = np.round(float(x), significant_figs)
        self.y = np.round(float(y), significant_figs)
        self.loc = (self.x, self.y)
        self.road = False
        self.interior = False
        self.name = name

    def __repr__(self):
        if self.name:
            return self.name
        else:
            return "(%.2f,%.2f)" % (self.x, self.y)

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

        self.on_road = False

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

        if G is None:
            self.G = nx.Graph()

        else:
            self.G = G
     
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


    def myedges(self):
        return [self.G[e[0]][e[1]]["myedge"] for e in self.G.edges()]

    def myweight(self):
        return [self.G[e[0]][e[1]]["weight"] for e in self.G.edges()]

    def copy(self):
        nx_copy = self.G.copy()
    
        
        copy = MyGraph(nx_copy)
   
        copy.name = self.name
        copy.rezero_vector = self.rezero_vector
        copy.rescale_vector = self.rescale_vector
        copy.td_dict = self.td_dict
    
        # order matters.  road nodes before interior parcels
        if hasattr(self, 'road_nodes'):
            copy.road_nodes = [n for n in copy.G.nodes() if n.road]

        if hasattr(self, 'road_edges'):
            copy.road_edges = [e for e in copy.myedges() if e.road]

        if hasattr(self, 'interior_parcels'):
            copy.define_interior_parcels()
     
        return copy


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

    def _cal_edge_face_index(self):
        self.edge_face_index = []
        for e in self.edge_list:
            pair = []
            for f in self.inner_facelist:
                if len(set(e.nodes).intersection(set(f.nodes))) == 2:
                    pair.append(f)
                if len(pair) == 2:
                    break
            self.edge_face_index.append(pair)

    def _cal_graph_node_feature(self):
        self.graph_node_feature = {}
        for n in self.node_list:
            self.graph_node_feature[n] = self._get_node_loc(
                n) + self._get_node_centrality(n)

    def feature_init(self):
        self._cal_graph_centrality()
        self._cal_graph_node_feature()
        self._cal_edge_index_and_length()
        self._cal_node_degree_and_isroad()
        self._cal_edge_face_index()

    def get_obs(self):
        numerical = self._get_numerical()
        node_feature = np.concatenate(
            [[self._get_node_feature(n) for n in self.node_list]], axis=1)
        # node_feature = np.zeros_like(node_feature)

        edge_part_feature = self._get_edge_part_feature()
        # edge_part_feature = np.zeros_like(edge_part_feature)
        edge_index = self.edge_index
        edge_mask = self._get_edge_mask()

        return numerical, node_feature, edge_part_feature, edge_index, edge_mask

    # ok
    def _get_edge_part_feature(self):
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
            [edge_isroad, edge_length, edge_face_interior ,edge_avg_dis, edge_ration_dis], axis=1)

        return edge_part_feature

    def _get_edge_mask(self):
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
            index_equ2 = np.argwhere(np.array(edge_mask) == 2)
            edge_mask = np.zeros(len(edge_mask))
            edge_mask[index_equ2] = 1

        return edge_mask
    
    # ok
    def _get_edge_face_interior(self):
        edge_face_interior=[]
        for pair in self.edge_face_index:
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

            elif len(pair) == 0:  # new case
                edge_face_interior.append(0)
        # return np.zeros_like(edge_face_interior)
        return edge_face_interior
    
    # ok
    def _get_edge_ration_dis(self):
        edge_dis_ration=[]
        for idx in range(len(self.edge_list)):
            idx1 = self.edge_index[idx][0]
            idx2 = self.edge_index[idx][1]
            if self.td_dict[idx1][idx2] == 1000:
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

        for pair in self.edge_face_index:
            if len(pair) == 1:
                f = pair[0]
                mean_dis = face_mean_dis[f]
            elif len(pair) == 2:
                f1 = pair[0]
                f2 = pair[1]
                mean_dis = (face_mean_dis[f1] + face_mean_dis[f2]) / 2
            elif len(pair) == 0:
                mean_dis = 1000
            edge_dis.append(mean_dis)

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

    # ok - double check
    def get_numerical_feature_size(self):
        return 4

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
        interior_ration = len(
            self.interior_parcels) / self.max_interior_parcels
        # print(stage1_ration, stage2_ration, interior_ration)
        return [0.5, stage1_ration, stage2_ration, interior_ration]

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


############################
# GEOMETRY CLEAN UP FUNCTIONS
############################

##########################################
#    WEAK DUAL CALCULATION FUNCTIONS
########################################

#############################################
#  DEFINING ROADS AND INTERIOR PARCELS
#############################################
    
    def define_roads(self) : # New to replace
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

    def define_interior_parcels(self): # New to replace
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
        # print "define interior parcels called"

    def update_node_properties(self): # New to replace
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
    def build_road_from_action(self, action: List):
        e = self.edge_list[int(action)]
        self.add_road_segment(e)

    # ok 
    def road_update(self, edge):
        self.G[edge.nodes[0]][edge.nodes[1]]['road'] = self.G[edge.nodes[0]][
            edge.nodes[1]]['weight']
    # ok 
    def add_road_segment(self, edge: MyEdge):
        """ Updates properties of graph to make edge a road. """
        edge = self.G[edge.nodes[0]][edge.nodes[1]]['myedge']
        # self.myw = self.G[edge.nodes[0]][edge.nodes[1]]['weight']

        self.td_dict_update(edge)

        edge.road = True
        if edge in self.road_edges:
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
        self.interior_parcels_update()

    # ok 
    def add_all_road(self):
        for e in self.myedges():
            if not e.road:
                self.add_road_segment(e)

    
#############################################
#   GEOMETRY AROUND BUILDING A GIVEN ROAD SEGMENT - c/(sh?)ould be deleted.
#############################################

############################
# REWARD FUNCTIONS
############################
    def save_step_data(self):
        #path = '/data2/suhongyuan/road_planning/data.csv'
        path = r'C:\Users\asdbe\OneDrive\Documents\GitHub\road-planning-for-slums\road_planning\data\data.csv'
        data=pd.DataFrame(data=[self.parcels_data,self.f2f_data,self.cost_data])
        data.to_csv(path,encoding='gbk')

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
        if len(self.interior_parcels) or self.del_parcel_num:
            before = self.face2face_avg()
            return 0
        else:
            before = self.f2f_avg
            now = self.face2face_avg()
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
                    if self.td_dict_face[i][j] != 1000:
                        sum += self.td_dict_face[i][j]
                        count+=1
                count -= 1 
        tmp = sum / (count)
        self.f2f_data.append(tmp)

        return self.f2f_avg

    # ok  
    def connected_ration(self):
        self.parcels_data.append(len(self.interior_parcels))
        if self.del_parcel_num == 0 and len(self.interior_parcels) != 0:
            return -1/self.max_del_interior_parcels
        else:
            return self.del_parcel_num / self.max_del_interior_parcels

    def td_dict_init(self): # comment about outerface
        roadG = MyGraph()
        for idx in range(len(self.road_edges)):
            e = self.road_edges[idx]
            roadG.add_edge(self.road_edges[idx],weight=self.G[e.nodes[0]][e.nodes[1]]['weight'])
        td_dict = dict(nx.shortest_path_length(roadG.G, weight="weight"))
### init_td_dict
        node_length = len(self.node_list)
        print('node:', node_length, 'edge:', len(self.edge_list))
        self.td_dict = [[a for a in range(node_length)]
                        for _ in range(node_length)]
        for n in range(node_length):
            for nn in range(node_length):
                if n == nn:
                    self.td_dict[n][nn] = 0
                else:
                    self.td_dict[n][nn] = 1000
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
                    self.td_dict_face[f1][f2] = 1000
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
# ### init_outface_dis
#         self.td_dict_face[self.outerface]={}
#         for f in self.inner_facelist:
#             self.td_dict_face[self.outerface][f] = 0
#         for f in self.interior_parcels:
#             self.td_dict_face[self.outerface][f] = 1000
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
                self.td_dict_face_min[f1][f2] = 1000
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

        print('td_init')

    # update node2node & face2face distace
    def td_dict_update(self, edge):# comment about outerface
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

        # for pair in change_node:
        #     idx1 = pair[0]
        #     idx2 = pair[1]
        #     n1 = self.node_list[idx1]
        #     n2 = self.node_list[idx2]
        #     if n1 in self.outerface.nodes:
        #         for f2 in [f for f in self.inner_facelist if n2 in f.nodes]:
        #             self.td_dict_face[self.outerface][f2] = min(self.td_dict_face[self.outerface][f2], self.td_dict[idx1][idx2])
        #     elif n2 in self.outerface.nodes:
        #         for f1 in [f for f in self.inner_facelist if n1 in f.nodes]:
        #             self.td_dict_face[self.outerface][f1] = min(self.td_dict_face[self.outerface][f1], self.td_dict[idx1][idx2])

    # ok             
    def interior_parcels_update(self):
        parcels = len(self.interior_parcels)
        self.interior_parcels=[]
        for f in self.inner_facelist:
            if self.td_dict_face[self.outerface][f] == 1000:
                    self.interior_parcels.append(f)

        for e in self.myedges():
            e.interior = False
        for p in self.interior_parcels:
            for e in p.edges:
                e.interior = True
        for n in self.G.nodes():
            mgh.is_interiornode(n, self)
        self.interior_nodes = [n for n in self.G.nodes() if n.interior]

        self.del_parcel_num = parcels - len(self.interior_parcels)
        if len(self.interior_parcels) == 0 and self.del_parcel_num != 0:
            self.full_connected_road_num = self.build_road_num


# ###################################
#      PLOTTING FUNCTIONS
# ##################################

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

            nlocs = self.location_dict()

            if update:
                # self.define_roads()
                # self.define_interior_parcels()
                pass

            if new_plot:
                plt.figure()

            edge_colors = [
                'blue' if e in self.road_edges and e not in self.stage2edges 
                else 'black' if e in self.stage2edges 
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
