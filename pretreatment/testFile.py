import sys
sys.path.append(r"C:\Users\asdbe\OneDrive\Documents\GitHub\road-planning-for-slums")
import my_graph



# nodeList = []
# node0 = my_graph.MyNode([1.55555555,2],5)
# node1 = my_graph.MyNode([1,2],5)
# node2 = my_graph.MyNode([2,2],5)

# print(f"Hash of node0: {hash(node0)}")
# print(f"Hash of node1: {hash(node1)}")
# print(f"Hash of node2: {hash(node2)}")

# print(f"id of node0: {id(node0)}")
# print(f"id of node1: {id(node1)}")
# print(f"id of node2: {id(node2)}")

# print (node0.x)

# print ("nobug")

# print (set([1,2,3,4,5]) - set([2,5,6]))

# import numpy as np


# neighbors = [0,1],[1,0],[0,-1],[-1,0]


# for b in neighbors:

#     i = [0,0]
#     def angle(b):
#         dx = b[0] - i[0]
#         dy = b[1] - i[1]
#         return np.arctan2(dx, dy)
    
#     reorder_neighbors = sorted(neighbors, key=angle)

# print (reorder_neighbors)

# listA = [1,2,3]
# road_max = None
# print ("hi",listA and road_max) 

# flag = False
# road_num = 5
# interior_parcels = []
# print(not interior_parcels and flag or road_num==10) 



# import numpy as np
# print (np.string_)

# import pickle
# def load_graph():
#     with open(r"C:\Users\asdbe\OneDrive\Documents\GitHub\road-planning-for-slums\data\Epworth_Demo.mg", 'rb') as mgfile:
#         mg = pickle.loads(mgfile.read())
#         mg.define_roads()
#         mg.define_interior_parcels()
#         mg.td_dict_init()
#         mg.feature_init()
#     return mg

# mg = load_graph()

# print (mg.node_list)