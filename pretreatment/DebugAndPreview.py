import sys
import os
cwd = os.getcwd()
sys.path.append(cwd) 

import pickle
import copy
import inspect
import time
import pretreatment.my_graph_helpers as mgh


def SaveGraph_mg(graph,save_dir,filename):
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the instance
    with open(os.path.join(save_dir, '{}.mg'.format(filename)), 'wb') as mgfile:
        pickle.dump(graph, mgfile)


def LoadGraph_mg(file_path):
    # Read the data
    with open(file_path, 'rb') as mgfile:
        graph_loaded = pickle.load(mgfile)

    return graph_loaded


def PrintEdgeCoords(edges):
    info = []
    for edge in edges:
        for node in edge.nodes:
            info.append(node.x)
            info.append(node.y)

    print (info)


def PrintNodeCoords(nodes):
    info = []
    for node in nodes:
        info.append(node.x)
        info.append(node.y)

    print (info)


### Check DeepCopy
class MyGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.some_dict = {1: 'a', 2: 'b'}
        self.some_keys = self.some_dict.keys()  # 可能会导致问题
        self.another_dict = {3: 'c', 4: 'd'}
        self.another_values = self.another_dict.values()  # 也可能会导致问题

    def copy(self):
        return MyGraph()

    def __repr__(self):
        return f"MyGraph(nodes={self.nodes}, edges={self.edges}, some_dict={self.some_dict})"

def get_dict_view_types():
    """返回 dict_keys, dict_values, dict_items 类型"""
    dummy_dict = {}
    return type(dummy_dict.keys()), type(dummy_dict.values()), type(dummy_dict.items())

def detect_dict_keys(obj, seen_objects=None):
    """遍历对象属性，检查是否包含 dict_keys 对象"""
    if seen_objects is None:
        seen_objects = set()
    
    if id(obj) in seen_objects:
        return
    seen_objects.add(id(obj))
    
    dict_view_types = get_dict_view_types()
    for attr_name, attr_value in inspect.getmembers(obj):
        if isinstance(attr_value, dict):
            for key, value in attr_value.items():
                if isinstance(value, dict_view_types):
                    print(f"属性 {attr_name} 包含不可深拷贝的 dict_keys 对象: {value}")
        elif isinstance(attr_value, dict_view_types):
            print(f"属性 {attr_name} 是不可深拷贝的 dict_keys 对象: {attr_value}")
        elif hasattr(attr_value, '__dict__'):
            detect_dict_keys(attr_value, seen_objects)

def CheckDeepCopy():
    # 创建图实例
    myG = MyGraph()
    # 检查 myG 对象是否包含 dict_keys 对象
    detect_dict_keys(myG)

    # 尝试深拷贝
    try:
        S0 = copy.deepcopy(myG)
        print("成功深拷贝")
    except Exception as e:
        print(f"深拷贝失败: {e}")

    # 打印原始对象和深拷贝对象
    print (type(myG.some_keys))
    print (type(myG.another_values))


### Debug
def CompareTwoDicts(myG):
    S0 = copy.deepcopy(myG)
    S1 = copy.deepcopy(myG)

    S0.td_dict_POI_Related_init() 
    S1.td_dict_POI_Related_init()



    print ("--------------------")
 

    # Loop by iteration
    optNum = 2

    #S0.td_dict_POI_Related_init() 
    #S0.PlotF2POIDist()

    time1 = time.time()
    for i in range(optNum): 
        S0.td_dict_POI_Related_init() 
        mgh.bisecting_road_forPOI(S0,True)
        #S0.td_dict_POI_Related_init() 
        print (len(S0.road_edges))
        #S0.plot_roads(parcel_labels=True)
        #S0.PlotF2POIDist()
    time2 = time.time()
    print ("first time: using update", time2-time1)


    for i in range(optNum): 
        S1.td_dict_POI_Related_init() 
        mgh.bisecting_road_forPOI(S1,False)
        S1.td_dict_POI_Related_init() 
        print (len(S1.road_edges))
        #S0.plot_roads(parcel_labels=True)
        #S0.PlotF2POIDist()
    time3 = time.time()
    print ("second time:", time3-time2)


    print ("---------------------")
    print ("S0.td_dict_nodeToPOInode",S0.td_dict_nodeToPOInode)
    print ("---------------------")
    print ("S1.td_dict_nodeToPOInode",S1.td_dict_nodeToPOInode)
    print ("==============")

    print ("S0.td_dict_nodeToPOIEdge",S0.td_dict_nodeToPOIEdge)
    print ("---------------------")
    print ("S1.td_dict_nodeToPOIEdge",S1.td_dict_nodeToPOIEdge)
    print ("==============")

    print ("S0.td_dict_faceToPOIEdge",S0.td_dict_faceToPOIEdge)
    print ("---------------------")
    print ("S1.td_dict_faceToPOIEdge",S1.td_dict_faceToPOIEdge)
    print ("==============")

    print ("S0.td_dict_ave_faceToPOIEdge",S0.td_dict_ave_faceToPOIEdge)   # Pending
    print ("---------------------")
    print ("S1.td_dict_ave_faceToPOIEdge",S1.td_dict_ave_faceToPOIEdge) 
    print ("==============")

def CheckSpecificFace(S0,faceCentroidX,POIEdgeNodeX):
    for face in S0.inner_facelist_True:
        if face.centroid.x == faceCentroidX:
            for POIEdge in S0.POIEdges:
                if POIEdge.nodes[0].x == POIEdgeNodeX:
                    print ("POIEdge",POIEdge)
                    print ("td_dict_faceToPOIEdge:",S0.td_dict_faceToPOIEdge[face][POIEdge])
                    print ("td_dict_faceToPOIEdge_TheNodePair:" ,S0.td_dict_faceToPOIEdge_TheNodePair[face][POIEdge])
