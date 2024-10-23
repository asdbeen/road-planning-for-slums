from pymoo.operators.crossover.hux import HUX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems.single.knapsack import create_random_knapsack_problem
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice, Binary

import numpy as np
from pymoo.core.repair import Repair

######################################################################
############################## Add Path ##############################
######################################################################
import sys
import os
cwd = os.getcwd()
if "road_planning" in cwd:      # for ssh remote terminal
    sys.path.append(os.path.join(cwd,'envs'))
    parent_dir = os.path.dirname(cwd)
    sys.path.append(parent_dir) 
    cwd = parent_dir

else:
    sys.path.append(cwd) 
    sys.path.append(os.path.join(cwd,'road_planning'))
    sys.path.append(os.path.join(cwd,'road_planning/envs'))

import pretreatment.my_graph_helpers as mgh


######################################################################
#########################  Load File   ###############################
######################################################################
slum_name = "punggol_1_withShortcut_withConfigAll_2diagonal_woCross_multiPOICat"
jsonPath = cwd + "/JSONInput/" + slum_name + ".json"

myG,myNodeDict,myEdgeDict = mgh.GraphFromJSON_Customized_IgnoreShortCut_MultiPOICat(jsonPath,scaleTag=True,new_min = 0,new_max = 5)   # 5

###### Initialize info
myG.define_roads_FirstTime()                 # Road
myG.define_interior_parcels()                # Interior_Parcels   ， in this case, it is None

myG.AddShortCutInGraph()

myG.td_dict_init()
myG.feature_init()
myG.td_dict_POI_Related_init_New()



def CopyGraph():
    slum_name = "punggol_1_withShortcut_withConfigAll_2diagonal_woCross_multiPOICat"
    jsonPath = cwd + "/JSONInput/" + slum_name + ".json"

    myG,myNodeDict,myEdgeDict = mgh.GraphFromJSON_Customized_IgnoreShortCut(jsonPath,scaleTag=True,new_min = 0,new_max = 5)   # 5
    myG.define_roads_FirstTime()                 # Road
    myG.define_interior_parcels()                # Interior_Parcels   ， in this case, it is None

    myG.AddShortCutInGraph()

    myG.td_dict_init()
    myG.feature_init()
    myG.td_dict_POI_Related_init_New()

    return myG

######################################
### GA
######################################
class MinimizeParcelToPOI(ElementwiseProblem):

    def __init__(self, myG ,numSelection,**kwargs):
        # Define 10 integer variables with bounds (0, 1)
        #vars = {f"y{i}": Integer(bounds=(0, 1)) for i in range(10)}
        #vars = {f"edge{i}": Binary() for i in range(len(myG.shortcutEdges))}

        super().__init__(
                         n_var=len(myG.shortcutEdges),  # 10 variables
                         n_obj=1,
                         n_constr=0,
                         xl=0,  # Lower bound for variables
                         xu=1,  # Upper bound for variables
                         type_var=Integer)  # Type of variables

        self.numShortcutEdges = len(myG.shortcutEdges)
        self.numSelection = numSelection
        self.myG = myG
        self.n_var = self.numShortcutEdges
 

    def _evaluate(self, X,out, *args, **kwargs):

        varList = X.tolist()  # Ensure x is treated as an array
        copyG = self.myG.copy(recalculateTag=False)
       
      
        # edgeMask = self.myG._get_edge_mask()
        
        # edgeMaskShortcut = []
        # for i in range(len(edgeMask)):
        #     thisEdge = self.myG.edge_list[i]
        #     if thisEdge in myG.shortcutEdges:
        #         edgeMaskShortcut.append(edgeMask[i])
   
        info = []
        info_Index = []
        for i in range(self.numShortcutEdges):
            if varList[i] == True:   # and edgeMaskShortcut[i] == 1 
                e = copyG.shortcutEdges[i]
                if not e.isRoad:  # Skip if the edge is already part of the existing road network
                    copyG.add_road_segment(e)
                   
                    for node in e.nodes:
                        info.append(node.x)
                        info.append(node.y)
                            
                    info_Index.append(i)
                    
                    # ### Update Mask
                    # edgeMask = self.myG._get_edge_mask()
                    # edgeMaskShortcut = []
                    # for j in range(len(edgeMask)):
                    #     thisEdge = self.myG.edge_list[j]
                    #     if thisEdge in myG.shortcutEdges:
                    #         edgeMaskShortcut.append([edgeMask[j]])
     
        # print (info)
        # print (info_Index)
 
        # Initial feature and property
        copyG.td_dict_init()
    
        copyG.feature_init()
    
        copyG.td_dict_POI_Related_init_New()
  
        print (copyG.f2POI_avg_EachCat_mean)
        out["F"] = copyG.f2POI_avg_EachCat_mean

        
        
class ConsiderMaximumSelection(Repair):

    def _do(self, problem, Z, **kwargs):

        # maximum capacity for the problem
        Q = problem.numSelection
        
        
        # the corresponding weight of each individual
        # weights = (Z * problem.W).sum(axis=1)
        # print (problem.W)
 
        w = np.ones(problem.n_var)
        weights = (Z * w).sum(axis=1)
        # now repair each indvidiual i
        for i in range(len(Z)):

            # the packing plan for i
            z = Z[i]

            # while the maximum capacity violation holds
            while weights[i] > Q:

                # randomly select an item currently picked
                item_to_remove = np.random.choice(np.where(z)[0])

                # and remove it
                z[item_to_remove] = False

                # adjust the weight
                weights[i] -= 1

        return Z



class MyCallback:
    def __init__(self):
        self.data = []
    
    def __call__(self, algorithm):
        gen = algorithm.n_gen
        opt = algorithm.pop.get("F").min()
        avg = algorithm.pop.get("F").mean()
        self.data.append((gen, opt, avg))


callback = MyCallback()

problem = MinimizeParcelToPOI(myG,10)
algorithm = GA(pop_size=100,
               sampling=BinaryRandomSampling(),
               crossover=HUX(),
               mutation=BitflipMutation(),
               repair=ConsiderMaximumSelection(),
               eliminate_duplicates=True,
               )

res = minimize(problem,
               algorithm,
               termination=('n_gen', 20),
               seed=1,
               verbose=True,
               callback=callback)



