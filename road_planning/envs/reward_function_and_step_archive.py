
def reward_info_function_saveForVersion2(mg: MyGraph, name: Text,
                         travel_distance_weight: float,
                         road_cost_weight: float,
                         travel_distance_POI_weight: float=0.5) -> Tuple[float, Dict]:
    """Returns the RL reward and info.

    Args:
        plc: Plan client object.
        name: Reward name, can be land_use, road, or intermediate.
        road_network_weight:  Weight of road network in the reward function.
        life_circle_weight: Weight of 15-min life circle in the reward function.
        greeness_weight: Weight of greeness in the reward function.
        concept_weight: Weight of planning concept in the reward function.
        calculate_road_style: Whether to calculate the road style.

    Returns:
        The RL reward.
        Info dictionary.
    """



    travel_distance = travel_distance_weight * mg.travel_distance()

    if name == 'connecting':
        travel_distance_POI = 0
        culdesacReward = mg.CuldesacReward()
        culdesacReward = 0   # so it wont affect in stage 1
        
    elif name == 'full_connected':
        
        ####################################### 
        ### Do POI related computation   - Version 1
        #######################################   
        # mg.td_dict_nodeToPOInode_init()
        # mg.td_dict_nodeToPOIEdge_init()
        # mg.td_dict_faceToPOIEdge_init()
        # mg.td_dict_ave_faceToPOIEdge_init()
        # travel_distance_POI = travel_distance_POI_weight * mg.travel_distance_forPOI() 

        ####################################### 
        ### Do POI related computation   - Version 2
        #######################################   
       
        mg.td_dict_nodeToPOInode_MultiCat_init()
        mg.td_dict_faceToPOInode_MultiCat_init()
        mg.td_dict_faceToPOInode_EachCat_init()   #### Closest one, becasue one parcel only need to go to the closest POI for each category
        mg.face2POI_EachCat_avg()
        travel_distance_POI = travel_distance_POI_weight * mg.travel_distance_forPOI_New() 
        # print ("name",name)
        # print ("travel_distance_POI",travel_distance_POI)
        # print ("f2f_avg",mg.f2f_avg)
        # print ("f2POI_avg",mg.f2POI_avg)
        # print ("Culdesac",mg.culdesacNum)    
        # print ("-------------")
   
        ####################################### 
        ### Do culdesac related computation
        ####################################### 
        culdesacReward = mg.CuldesacReward()

    road_cost = road_cost_weight * mg.road_cost()
    connect_reward = mg.connected_ration()

    interior_parcels_num = len(mg.interior_parcels)
    connecting_steps = mg._get_full_connected_road_num()
    
    total_road_cost = mg.total_cost()


   
    # print ("culdesacReward",culdesacReward)
    # print ("travel_distance_POI",travel_distance_POI)  
    # print(connect_reward , travel_distance , road_cost)
    # print(face2face_avg,total_road_cost)

    # print ("travel_distance",travel_distance)
    # print ("travel_distance_POI",travel_distance_POI)
    # print ("road_cost",travel_distance_POI)
    # print ("--------")
    
    #culdesacReward = 0


    #finalReward = connect_reward + travel_distance + travel_distance_POI +  road_cost + culdesacReward
    #finalReward = connect_reward  + travel_distance + travel_distance_POI + road_cost  # + culdesacReward    # for complete roadnetwork  

    finalReward = travel_distance_POI   # Consider POI only

    #print ("name",name, "connect_reward",connect_reward,"culdesacReward",culdesacReward,"culdesacNum",mg.culdesacNum,"finalReward",finalReward)

    return finalReward, {

        'connect_reward': connect_reward,
        'travel_distance_reward': travel_distance,
        #'travel_distance_POI_reward': travel_distance_POI,   # New
        'road_cost_reward': road_cost,

        'interior_parcels_num':interior_parcels_num,
        'connecting_steps':connecting_steps,
        'f2f_dis_avg': mg.f2f_avg,        # original is 0
        'total_road_cost': total_road_cost,

        'travel_distance_weight':travel_distance_weight,
        'road_cost_weight':road_cost_weight,

        'f2POI_dis_avg':mg.f2POI_avg,        # New
        'culdesacReward':culdesacReward,        # New
        'f2POI_avg_EachCat_mean':mg.f2POI_avg_EachCat_mean        # New

    }


### This is the function that is current working to select the vehicular network and make it connected 
def reward_info_function_succeedfortangha2(mg: MyGraph, name: Text,  #s
                         travel_distance_weight: float,
                         road_cost_weight: float,
                         travel_distance_POI_weight: float=0.5,
                         action = None) -> Tuple[float, Dict]:
    """Returns the RL reward and info.

    Args:
        plc: Plan client object.
        name: Reward name, can be land_use, road, or intermediate.
        road_network_weight:  Weight of road network in the reward function.
        life_circle_weight: Weight of 15-min life circle in the reward function.
        greeness_weight: Weight of greeness in the reward function.
        concept_weight: Weight of planning concept in the reward function.
        calculate_road_style: Whether to calculate the road style.

    Returns:
        The RL reward.
        Info dictionary.
    """


    implicitConnectReward = 0
    travel_distance = travel_distance_weight * mg.travel_distance()

    if name == 'connecting':

        travel_distance_POI = 0                                             # it is not the focus in the "connecting" stage
        road_cost = road_cost_weight * mg.road_cost()                       # The default weight is 0.8 now 
        connect_reward = mg.connected_ration()                              # orginal
        culdesacReward = mg.CuldesacReward() *10                            # weight is emphasized
        implicitCuldesacReward =  mg.ImplicitCuldesacReward(culdesacReward)               # implicit culdesac reward    
        angleReward = mg.AngleReward()[1]                                   # angle     

        ####  A strict condition to prioritize the angle reward, if the angle is greater than 45, then the connect reward will be 0
        # originalAngle = mg.AngleReward_OriginalAngle()
        # connect_reward = 0 if angleReward>=45 else connect_reward

        #print ("In reward_info_function:", "name:",name,"connect_reward:",connect_reward,"implicitCuldesacReward",implicitCuldesacReward,"culdesacReward:",culdesacReward,"implicitCuldesacReward:",implicitCuldesacReward,"angleReward:",angleReward)

        
    elif name == 'full_connected':
        connect_reward = 0                                                  # so it wont affect the 'full_connected' stage
        road_cost = road_cost_weight * mg.road_cost()                       # The default weight is 0.8 now 
        culdesacReward = mg.CuldesacReward() *10                            # weight is emphasized      
        implicitCuldesacReward =  mg.ImplicitCuldesacReward(culdesacReward)               # implicit culdesac reward  
        angleReward = mg.AngleReward()[1]                                   # angle 
 


    elif name == 'POI_improvement':
        pass 



    interior_parcels_num = len(mg.interior_parcels)
    connecting_steps = mg._get_full_connected_road_num()
    total_road_cost = mg.total_cost()
    
    
    
   

    #print ("name",name)
    # print ("culdesacReward",culdesacReward)
    # print ("connect_reward",connect_reward)
    # print ("road_cost",road_cost)
    # print ("travel_distance_POI",travel_distance_POI)  
    #print("connect_reward", connect_reward)
    # print(face2face_avg,total_road_cost)

    # print ("travel_distance",travel_distance)
    # print ("travel_distance_POI",travel_distance_POI)
    # print ("road_cost",road_cost)
    # print ("angleReward",angleReward)
    
    
    #culdesacReward = 0


    finalReward = connect_reward + implicitConnectReward + road_cost + culdesacReward + implicitCuldesacReward + angleReward
    #finalReward = connect_reward  + travel_distance + travel_distance_POI + road_cost  # + culdesacReward    # for complete roadnetwork  
    #print ("finalReward",finalReward)
    #print ("total_road_cost",total_road_cost)
    #print ("-----------------------")

    return finalReward, {

        'connect_reward': connect_reward,
        'travel_distance_reward': travel_distance,
        #'travel_distance_POI_reward': travel_distance_POI,   # New
        'road_cost_reward': road_cost,

        'interior_parcels_num':interior_parcels_num,
        'connecting_steps':connecting_steps,
        'f2f_dis_avg': mg.f2f_avg,        # original is 0
        'total_road_cost': total_road_cost,

        'travel_distance_weight':travel_distance_weight,
        'road_cost_weight':road_cost_weight,

        'f2POI_dis_avg':mg.f2POI_avg,        # New
        'culdesacReward':culdesacReward,        # New
        # 'f2POI_avg_EachCat_mean':mg.f2POI_avg_EachCat_mean        # New

    }


### This is the function that is current working to select the vehicular network and make it connected 
def reward_info_function_save_20241115(mg: MyGraph, name: Text,
                         travel_distance_weight: float,
                         road_cost_weight: float,
                         travel_distance_POI_weight: float=0.5,
                         action = None) -> Tuple[float, Dict]:
    """Returns the RL reward and info.

    Args:
        plc: Plan client object.
        name: Reward name, can be land_use, road, or intermediate.
        road_network_weight:  Weight of road network in the reward function.
        life_circle_weight: Weight of 15-min life circle in the reward function.
        greeness_weight: Weight of greeness in the reward function.
        concept_weight: Weight of planning concept in the reward function.
        calculate_road_style: Whether to calculate the road style.

    Returns:
        The RL reward.
        Info dictionary.
    """

    hitBoundaryReward = 0
    singleParcleRingRoadPunishment = 0
    travel_distance = travel_distance_weight * mg.travel_distance()

    if name == 'connecting':
        
        ######################
        # Check the new road edge is "connected to the existing cul-de-sac"(case1) or "grow like a T-junction"(case2) or "in bettween"(case3)

        caseTag = mg.CheckNewRoadEdgeCase()
        ######################
        travel_distance_POI = 0                                             # it is not the focus in the "connecting" stage

        road_cost = road_cost_weight * mg.road_cost()                       # The default weight is 0.8 now 
        angleReward = mg.AngleReward()[1]                                   # angle   

        connect_reward = mg.connected_ration()                              # orginal
        if connect_reward == 0:
            implicitConnectReward = mg.ImplicitConnectReward(connect_reward)                  # implicit connect reward
        else:
            implicitConnectReward = 0

        culdesacReward = mg.CuldesacReward()                               # weight is not emphasized, offset the case in ImplicitConnectReward
        implicitCuldesacReward =  mg.ImplicitCuldesacReward(culdesacReward)              # implicit culdesac reward    

        # culdesacReward = 0                          
        # implicitCuldesacReward = 0    


        ####  A strict condition to prioritize the angle reward, if the angle is greater than 45, then the connect reward will be 0
        originalAngle = mg.AngleReward_OriginalAngle()
        connect_reward = 0 if originalAngle>=45 else connect_reward
        implicitConnectReward = 0 if originalAngle>=45 else implicitConnectReward
        ####  Dont encourage the cul-de-sac fixed by big angle turning 
        if caseTag == "case1":
            
            culdesacReward = 0 if originalAngle>=45 else culdesacReward
            implicitCuldesacReward = 0 if originalAngle>=45 else implicitCuldesacReward      
            # print ("case1",originalAngle,culdesacReward,implicitCuldesacReward)
        if caseTag == "case3":
            # check if one side of edge hit boundary, give extra reward
            newRoadEdge = mg.road_edges[-1]
            if newRoadEdge.nodes[0].onBoundary or newRoadEdge.nodes[1].onBoundary:
                if originalAngle<=45:
                    hitBoundaryReward = 1


        ####  Small Ring Road Punishment ####
        singleParcleRingRoadPunishment = mg.SingleParcleRingRoadPunishment()*10   # New
        #singleParcleRingRoadPunishment = 0
        # print ("In reward_info_function:", "name:",name,"connect_reward:",connect_reward,"implicitConnectReward",implicitConnectReward,"culdesacReward:",culdesacReward,"ImplicitCuldesacReward:",implicitCuldesacReward,"angleReward:",angleReward,"case:",caseTag)
        # print ("In reward_info_function:", "road_cost:",road_cost,"angleReward:",angleReward)

        
    elif name == 'full_connected':
        
        road_cost = road_cost_weight * mg.road_cost()                       # The default weight is 0.8 now 
        angleReward = mg.AngleReward()[1]                                   # angle 

        connect_reward = 0                                                  # so it wont affect the 'full_connected' stage
        implicitConnectReward = 0                           

        culdesacReward = mg.CuldesacReward() *1                            # weight is emphasized      
        implicitCuldesacReward =  mg.ImplicitCuldesacReward(culdesacReward) # implicit culdesac reward  

        ####  Small Ring Road Punishment ####
        originalAngle = mg.AngleReward_OriginalAngle()
        singleParcleRingRoadPunishment = mg.SingleParcleRingRoadPunishment()*10   # New
        #singleParcleRingRoadPunishment = 0

    elif name == 'POI_improvement':
        pass 



    interior_parcels_num = len(mg.interior_parcels)
    connecting_steps = mg._get_full_connected_road_num()
    total_road_cost = mg.total_cost()
    
    
   

    #print ("name",name)
    # print ("culdesacReward",culdesacReward)
    # print ("connect_reward",connect_reward)
    # print ("road_cost",road_cost)
    # print ("travel_distance_POI",travel_distance_POI)  
    #print("connect_reward", connect_reward)
    # print(face2face_avg,total_road_cost)

    # print ("travel_distance",travel_distance)
    # print ("travel_distance_POI",travel_distance_POI)
    # print ("road_cost",road_cost)
    # print ("angleReward",angleReward)
    
    
    #culdesacReward = 0

    #implicitCuldesacReward
    finalReward = connect_reward + implicitConnectReward + road_cost + culdesacReward + implicitCuldesacReward + angleReward + singleParcleRingRoadPunishment + hitBoundaryReward
    #finalReward = connect_reward  + travel_distance + travel_distance_POI + road_cost  # + culdesacReward    # for complete roadnetwork  
    #print ("finalReward",finalReward)
    #print ("total_road_cost",total_road_cost)
    #print ("-----------------------")

    return finalReward, {

        'connect_reward': connect_reward,
        'travel_distance_reward': travel_distance,
        #'travel_distance_POI_reward': travel_distance_POI,   # New
        'road_cost_reward': road_cost,

        'interior_parcels_num':interior_parcels_num,
        'connecting_steps':connecting_steps,
        'f2f_dis_avg': mg.f2f_avg,        # original is 0
        'total_road_cost': total_road_cost,

        'travel_distance_weight':travel_distance_weight,
        'road_cost_weight':road_cost_weight,

        'f2POI_dis_avg':mg.f2POI_avg,        # New
        'culdesacReward':culdesacReward,        # New
        # 'f2POI_avg_EachCat_mean':mg.f2POI_avg_EachCat_mean        # New

    }

def reward_info_function_save_20241118(mg: MyGraph, name: Text,
                         travel_distance_weight: float,
                         road_cost_weight: float,
                         travel_distance_POI_weight: float=0.5,
                         action = None) -> Tuple[float, Dict]:
    """Returns the RL reward and info.

    Args:
        plc: Plan client object.
        name: Reward name, can be land_use, road, or intermediate.
        road_network_weight:  Weight of road network in the reward function.
        life_circle_weight: Weight of 15-min life circle in the reward function.
        greeness_weight: Weight of greeness in the reward function.
        concept_weight: Weight of planning concept in the reward function.
        calculate_road_style: Whether to calculate the road style.

    Returns:
        The RL reward.
        Info dictionary.
    """

    hitBoundaryReward = 0
    singleParcleRingRoadPunishment = 0
    travel_distance = travel_distance_weight * mg.travel_distance()

    if name == 'connecting':
        
        ######################
        # Check the new road edge is "connected to the existing cul-de-sac"(case1) or "grow like a T-junction"(case2) or "in bettween"(case3)

        caseTag = mg.CheckNewRoadEdgeCase()
        ######################
        travel_distance_POI = 0                                             # it is not the focus in the "connecting" stage

        road_cost = road_cost_weight * mg.road_cost()                       # The default weight is 0.8 now 
        angleReward = mg.AngleReward()[1]                                   # angle   

        connect_reward = mg.connected_ration()                              # orginal
        # if connect_reward == 0:
        #     implicitConnectReward = mg.ImplicitConnectReward(connect_reward)                  # implicit connect reward
        # else:
        #     implicitConnectReward = 0

        culdesacReward = mg.CuldesacReward()                               # weight is not emphasized, offset the case in ImplicitConnectReward
        #implicitCuldesacReward =  mg.ImplicitCuldesacReward(culdesacReward)              # implicit culdesac reward    

        # culdesacReward = 0                          
        # implicitCuldesacReward = 0    


        ####  A strict condition to prioritize the angle reward, if the angle is greater than 45, then the connect reward will be 0
        originalAngle = mg.AngleReward_OriginalAngle()
        #connect_reward = 0 if originalAngle>=45 else connect_reward
        #implicitConnectReward = 0 if originalAngle>=45 else implicitConnectReward
        ####  Dont encourage the cul-de-sac fixed by big angle turning 
        if caseTag == "case1":
            # culdesacReward = 0 if originalAngle>=45 else culdesacReward
            #implicitCuldesacReward = 0 if originalAngle>=45 else implicitCuldesacReward  
            if originalAngle>=45:
                angleReward *=5
            # print ("case1",originalAngle,culdesacReward,implicitCuldesacReward)
        
        if caseTag == "case2":
            if originalAngle<=45:
                angleReward = 0

        # if caseTag == "case3":
        #     # check if one side of edge hit boundary, give extra reward
        #     newRoadEdge = mg.road_edges[-1]
        #     if newRoadEdge.nodes[0].onBoundary or newRoadEdge.nodes[1].onBoundary:
        #         if originalAngle<=45:
        #             hitBoundaryReward = 1


        ####  Small Ring Road Punishment ####
        # singleParcleRingRoadPunishment = mg.SingleParcleRingRoadPunishment()*10   # New
        #singleParcleRingRoadPunishment = 0
        # print ("In reward_info_function:", "name:",name,"connect_reward:",connect_reward,"implicitConnectReward",implicitConnectReward,"culdesacReward:",culdesacReward,"ImplicitCuldesacReward:",implicitCuldesacReward,"angleReward:",angleReward,"case:",caseTag)
        # print ("In reward_info_function:", "road_cost:",road_cost,"angleReward:",angleReward)

        
    elif name == 'full_connected':
        
        road_cost = road_cost_weight * mg.road_cost()                       # The default weight is 0.8 now 
        angleReward = mg.AngleReward()[1]                                   # angle 

        connect_reward = 0                                                  # so it wont affect the 'full_connected' stage
        # implicitConnectReward = 0                           

        culdesacReward = mg.CuldesacReward() *1                            # weight is emphasized      
        # implicitCuldesacReward =  mg.ImplicitCuldesacReward(culdesacReward) # implicit culdesac reward  

        ####  Small Ring Road Punishment ####
        originalAngle = mg.AngleReward_OriginalAngle()
        caseTag = mg.CheckNewRoadEdgeCase()
        if caseTag == "case1":
            if originalAngle>=45:
                angleReward *=5
        
        if caseTag == "case2":
            if originalAngle<=45:
                angleReward = 0

        # singleParcleRingRoadPunishment = mg.SingleParcleRingRoadPunishment()*10   # New
        #singleParcleRingRoadPunishment = 0

    elif name == 'POI_improvement':
        pass 



    interior_parcels_num = len(mg.interior_parcels)
    connecting_steps = mg._get_full_connected_road_num()
    total_road_cost = mg.total_cost()
    
    
   

    #print ("name",name)
    # print ("culdesacReward",culdesacReward)
    # print ("connect_reward",connect_reward)
    # print ("road_cost",road_cost)
    # print ("travel_distance_POI",travel_distance_POI)  
    #print("connect_reward", connect_reward)
    # print(face2face_avg,total_road_cost)

    # print ("travel_distance",travel_distance)
    # print ("travel_distance_POI",travel_distance_POI)
    # print ("road_cost",road_cost)
    # print ("angleReward",angleReward)
    
    
    #culdesacReward = 0

    #implicitCuldesacReward
    finalReward = connect_reward  + road_cost + culdesacReward + angleReward + singleParcleRingRoadPunishment + hitBoundaryReward
    #finalReward = connect_reward  + travel_distance + travel_distance_POI + road_cost  # + culdesacReward    # for complete roadnetwork  
    #print ("finalReward",finalReward)
    #print ("total_road_cost",total_road_cost)
    #print ("-----------------------")

    return finalReward, {

        'connect_reward': connect_reward,
        'travel_distance_reward': travel_distance,
        #'travel_distance_POI_reward': travel_distance_POI,   # New
        'road_cost_reward': road_cost,

        'interior_parcels_num':interior_parcels_num,
        'connecting_steps':connecting_steps,
        'f2f_dis_avg': mg.f2f_avg,        # original is 0
        'total_road_cost': total_road_cost,

        'travel_distance_weight':travel_distance_weight,
        'road_cost_weight':road_cost_weight,

        'f2POI_dis_avg':mg.f2POI_avg,        # New
        'culdesacReward':culdesacReward,        # New
        # 'f2POI_avg_EachCat_mean':mg.f2POI_avg_EachCat_mean        # New

    }

def reward_info_function_strict_rule_version(mg: MyGraph, name: Text,
                         travel_distance_weight: float,
                         road_cost_weight: float,
                         travel_distance_POI_weight: float=0.5,
                         action = None) -> Tuple[float, Dict]:
    """Returns the RL reward and info.

    Args:
        plc: Plan client object.
        name: Reward name, can be land_use, road, or intermediate.
        road_network_weight:  Weight of road network in the reward function.
        life_circle_weight: Weight of 15-min life circle in the reward function.
        greeness_weight: Weight of greeness in the reward function.
        concept_weight: Weight of planning concept in the reward function.
        calculate_road_style: Whether to calculate the road style.

    Returns:
        The RL reward.
        Info dictionary.
    """

    
    travel_distance = travel_distance_weight * mg.travel_distance()

    connect_reward = 0
    implicitConnectReward = 0
    road_cost = 0
    angleReward = 0

    if name == 'connecting':
        
        ######################
        # Check the new road edge is "connected to the existing cul-de-sac"(case1) or "grow like a T-junction"(case2) or "in bettween"(case3)
        caseTag = mg.CheckNewRoadEdgeCase()
        ######################
        road_cost = road_cost_weight * mg.road_cost()                       # The default weight is 0.8 now 
        connect_reward = mg.connected_ration()                              # orginal
        if connect_reward == 0:
            implicitConnectReward = mg.ImplicitConnectReward(connect_reward)                  # implicit connect reward
        else:
            implicitConnectReward = 0

        
        angleReward = angleReward = mg.AngleReward()[1] 

   
        # print ("In reward_info_function:", "name:",name,"connect_reward:",connect_reward,"implicitConnectReward",implicitConnectReward,"culdesacReward:",culdesacReward,"ImplicitCuldesacReward:",implicitCuldesacReward,"angleReward:",angleReward,"case:",caseTag)
        # print ("In reward_info_function:", "road_cost:",road_cost,"angleReward:",angleReward)

        
    elif name == 'full_connected':
        road_cost = road_cost_weight * mg.road_cost()                       # The default weight is 0.8 now 
        connect_reward = 0                                                  # so it wont affect the 'full_connected' stage


    elif name == 'POI_improvement':
        pass 


    interior_parcels_num = len(mg.interior_parcels)
    connecting_steps = mg._get_full_connected_road_num()
    total_road_cost = mg.total_cost()
    
    
    finalReward = connect_reward  + implicitConnectReward + road_cost + angleReward  
    #finalReward = connect_reward  + travel_distance + travel_distance_POI + road_cost  # + culdesacReward    # for complete roadnetwork  
    #print ("finalReward",finalReward)
    #print ("total_road_cost",total_road_cost)
    #print ("-----------------------")

    return finalReward, {

        'connect_reward': connect_reward,
        'travel_distance_reward': travel_distance,
        #'travel_distance_POI_reward': travel_distance_POI,   # New
        'road_cost_reward': road_cost,

        'interior_parcels_num':interior_parcels_num,
        'connecting_steps':connecting_steps,
        'f2f_dis_avg': mg.f2f_avg,        # original is 0
        'total_road_cost': total_road_cost,

        'travel_distance_weight':travel_distance_weight,
        'road_cost_weight':road_cost_weight,

        'f2POI_dis_avg':mg.f2POI_avg        # New
        # 'culdesacReward':culdesacReward,        # New
        # 'f2POI_avg_EachCat_mean':mg.f2POI_avg_EachCat_mean        # New

    }

# Increase T AND L junction reward
def reward_info_function_archive(mg: MyGraph, name: Text,
                         travel_distance_weight: float,
                         road_cost_weight: float,
                         travel_distance_POI_weight: float=0.5,
                         action = None) -> Tuple[float, Dict]:
    """Returns the RL reward and info.

    Args:
        plc: Plan client object.
        name: Reward name, can be land_use, road, or intermediate.
        road_network_weight:  Weight of road network in the reward function.
        life_circle_weight: Weight of 15-min life circle in the reward function.
        greeness_weight: Weight of greeness in the reward function.
        concept_weight: Weight of planning concept in the reward function.
        calculate_road_style: Whether to calculate the road style.

    Returns:
        The RL reward.
        Info dictionary.
    """
    L_juncion_edges = mg.Collect_L_Junction_Edge(ratio = 0)
    T_juncion_edges = mg.Collect_T_Junction_Edge()
    # print ("L_juncion_edges",[mg.edge_list.index(edge) for edge in L_juncion_edges])
    # print ("T_juncion_edges",[mg.edge_list.index(edge) for edge in T_juncion_edges])
    hitBoundaryReward = 0
    TorL_Reward = 0
    travel_distance = travel_distance_weight * mg.travel_distance()
    culdesacReward = 0

    if name == 'connecting':
        
        ######################
        # Check the new road edge is "connected to the existing cul-de-sac"(case1) or "grow like a T-junction"(case2) or "in bettween"(case3)

        caseTag = mg.CheckNewRoadEdgeCase()
        ######################
        travel_distance_POI = 0                                             # it is not the focus in the "connecting" stage

        ##### Length related  !!!
        road_cost = road_cost_weight * mg.road_cost()                       # The default weight is 0.8 now 
        
        ##### Connect related   !!!
        connect_reward = mg.connected_ration()                              # orginal
        if connect_reward < 0:
            implicitConnectReward = mg.ImplicitConnectReward(connect_reward)                  # implicit connect reward
        else:
            implicitConnectReward = 0

  
        ##### Angle related   !!!
        ####  A strict condition to prioritize the angle reward, if the angle is greater than 45, then the connect reward will be 0
        angleReward = mg.AngleReward()[1]                                   # angle   
        originalAngle = mg.AngleReward_OriginalAngle()

        # if caseTag == "case1":
        #     if originalAngle>=45:
        #         angleReward *=5             
  
        # if caseTag == "case2":           # generate between edges
        #     if originalAngle<=45:       #   L or T
        #         angleReward = 0
        #     else:
        #         angleReward *=5

        ##### Culdesac related    !!! 
        culdesacReward = mg.CuldesacReward() 
        if caseTag == "case3":
            # check if one side of edge hit boundary, and it reduces one cul-de-sac, give extra reward
            # this encourage smooth connection to the boundary
            newRoadEdge = mg.road_edges[-1]
            if newRoadEdge.nodes[0].onBoundary or newRoadEdge.nodes[1].onBoundary:
                if culdesacReward>0:
                    if originalAngle<=45:
                        hitBoundaryReward = 1

        

        ##### Encourage generation from T and L junction   !!!
        if caseTag == "case2":
            newRoadEdge = mg.road_edges[-1]
            if newRoadEdge in L_juncion_edges or newRoadEdge in T_juncion_edges:
                if connect_reward> 0 or implicitConnectReward>0:
                    TorL_Reward = 1
                    angleReward = 0
                else:
                    TorL_Reward = 0
                    angleReward *=5
            else:
                TorL_Reward = 0
                angleReward *=5 

        if caseTag == "case1" or caseTag == "case3":
            if originalAngle>=45:
                angleReward *=5  
            
            else:
                angleReward = 0

        
        ringRoadPunishment = mg.RingRoadAreaPunishment()


        
    elif name == 'full_connected':
        caseTag = mg.CheckNewRoadEdgeCase()

        ##### Length related  !!!
        road_cost = road_cost_weight * mg.road_cost()                       # The default weight is 0.8 now 

        ##### Angle related   !!!
        angleReward = mg.AngleReward()[1] *5                                  # angle 
        originalAngle = mg.AngleReward_OriginalAngle()
        if originalAngle<=45:
            angleReward = 0
        ##### Connection related   !!!
        connect_reward = 0                                                  # so it wont affect the 'full_connected' stage
        implicitConnectReward = 0                           

        ##### Culdesac related   !!!
        culdesacReward = mg.CuldesacReward() *1                            # weight is not emphasized      
        #culdesacReward = mg.CuldesacReward() 
        if caseTag == "case3":
            # check if one side of edge hit boundary, and it reduces one cul-de-sac, give extra reward
            # this encourage smooth connection to the boundary
            newRoadEdge = mg.road_edges[-1]
            if newRoadEdge.nodes[0].onBoundary or newRoadEdge.nodes[1].onBoundary:
                if culdesacReward>0:
                    if originalAngle<=45:
                        hitBoundaryReward = 1

        ringRoadPunishment = mg.RingRoadAreaPunishment()
        # implicitCuldesacReward =  mg.ImplicitCuldesacReward(culdesacReward) # implicit culdesac reward  


    elif name == 'POI_improvement':
        pass 



    interior_parcels_num = len(mg.interior_parcels)
    connecting_steps = mg._get_full_connected_road_num()
    total_road_cost = mg.total_cost()
    
    
  

    #print ("name",name)
    # print ("culdesacReward",culdesacReward)
    # print ("connect_reward",connect_reward)
    # print ("road_cost",road_cost)
    # print ("travel_distance_POI",travel_distance_POI)  
    #print("connect_reward", connect_reward)
    # print(face2face_avg,total_road_cost)

    # print ("travel_distance",travel_distance)
    # print ("travel_distance_POI",travel_distance_POI)
    # print ("road_cost",road_cost)
    # print ("angleReward",angleReward)
    
    
    #culdesacReward = 0

    #implicitCuldesacReward
    finalReward = road_cost + connect_reward + implicitConnectReward + angleReward + hitBoundaryReward + TorL_Reward + ringRoadPunishment
    print ("In reward_info_function:", "EDGEID",mg.edge_list.index(mg.road_edges[-1]),"name:",name,"road_cost:",road_cost,"connect_reward",connect_reward,"implicitConnectReward:",implicitConnectReward,"angleReward:",angleReward,"hitBoundaryReward:",hitBoundaryReward,"TorL_Reward",TorL_Reward,"case:",caseTag)
    
    mg.total_angle_cost +=angleReward
    mg.total_hit_boundary_reward += hitBoundaryReward 
    mg.total_explcit_connection_reward += connect_reward 
    mg.total_implcit_connection_reward += implicitConnectReward   
    mg.total_all_connection_reward = mg.total_explcit_connection_reward + mg.total_implcit_connection_reward 
    mg.total_L_T_reward += TorL_Reward
    mg.total_RingRoad_reward += ringRoadPunishment
    print ("mg.total_L_T_reward,",mg.total_L_T_reward)
    mg.accumulated_reward += finalReward
 
    #print ("-----------------------")

    return finalReward, {

        'connect_reward': connect_reward,
        'travel_distance_reward': travel_distance,
        #'travel_distance_POI_reward': travel_distance_POI,   # New
        'road_cost_reward': road_cost,

        'interior_parcels_num':interior_parcels_num,
        'connecting_steps':connecting_steps,
        'f2f_dis_avg': mg.f2f_avg,        # original is 0
        'total_road_cost': total_road_cost,

        'travel_distance_weight':travel_distance_weight,
        'road_cost_weight':road_cost_weight,

        'f2POI_dis_avg':mg.f2POI_avg,        # New
        'culdesacReward':culdesacReward,        # New

        'total_angle_cost':mg.total_angle_cost,
        'total_hit_boundary_reward':mg.total_hit_boundary_reward,
        'total_explcit_connection_reward':mg.total_explcit_connection_reward,
        'total_implcit_connection_reward':mg.total_implcit_connection_reward,
        'total_all_connection_reward':mg.total_all_connection_reward,
        'total_L_T_reward':mg.total_L_T_reward,
        'total_RingRoad_reward':mg.total_RingRoad_reward,   
        'accumulated_reward':mg.accumulated_reward

    }


#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################



def step_save_20241118(self, action: List,
            logger: logging.Logger) -> Tuple[List, float, bool, Dict]:
    """
    Run one timestep of the environment's dynamics. When end of episode
    is reached, you are responsible for calling `reset()` to reset
    the environment's state.

    Accepts an action and returns a tuple (observation, reward, done, info).

    Args:
        action (np.ndarray of size 2): The action to take.
                                        1 is the land_use placement action.
                                        1 is the building road action.
        logger (Logger): The logger.

    Returns:
        observation (object): agent's observation of the current environment
        reward (float) : amount of reward returned after previous action
        done (bool): whether the episode has ended, in which case further step() calls will return undefined results
        info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
    """


    if self._done:
        raise RuntimeError('Action taken after episode is done.')

    else:
        if self._stage == 'connecting':
            self._action_history.append(action)
            self.build_road(action,POIVersionTag = False)  

            self._connecting_steps += 1
            if self._connecting_steps >= math.floor(self._total_road_steps *self.build_ration):    ###### Stop Condition: If it is still in the "connecting" stage, but step reach the maximum road steps ######  
                
                self.transition_stage()                                                             ###### Stage Transition: It will transit to "Done" ######
                                        
                
            elif self._full_connected():                                                             ###### Stage Transition: If it is still in the "connecting" stage, but all the parcels are connected ######
                self.transition_stage()

            
        
        elif self._stage == 'full_connected':                
            self._action_history.append(action)
            self.build_road(action,POIVersionTag = False)   

            self._full_connected_steps += 1

            if (self._mg.CheckCuldesacNum_NotAssign()==0) :                                                                 ###### Stage Transition: It will transit to "Done" ###### 
                self.transition_stage()             

            elif  (self._full_connected_steps + self._connecting_steps > self._total_road_steps * self.build_ration):       ###### Stop Condition: If it is still in the "connecting" stage, but total step reach the maximum road steps ######  
                self.transition_stage()                                                                                     ###### Stage Transition: It will transit to "Done" ######      
            

        reward, info = self.get_reward_info()                   ######  Culdesac Reward is calculated in this function, it will update cul-de-sac number ######
        if self._stage == 'done':
            self.save_step_data()


    
    converted_list = [float(arr) for arr in self._action_history]
    # print ("self._action_history",converted_list)
    # if len(converted_list) > 3:
    #     if converted_list[-2] == 54.0:
    #         print (converted_list +"asda")
    # culdesacNum_Check = self._mg.CheckCuldesacNum_NotAssign()
    if self._stage == 'connecting':
        # print ("IN STEP FUNCTION: self._stage == connecting")
        return self._get_obs(), reward, self._done, info

    # elif self._stage == 'full_connected' and culdesacNum_Check == 0:                                                      ##### This wont happen again, let's check
    #     print ("IN STEP FUNCTION: self._stage == full_connected and culdesacNum_Check == 0")
    #     return self._get_obs(), reward, self._done, info

    elif self._stage == 'full_connected':                                                                                   # and culdesacNum_Check != 0:    ##### for completing network
        # print ("IN STEP FUNCTION: self._stage == full_connected and culdesacNum_Check != 0")
        return self._get_obs_stage2_culdesac(), reward, self._done, info
    
    elif self._stage == 'done':
        return self._get_obs(), reward, self._done, info


#### Original one
def step_Original(self, action: List,
            logger: logging.Logger) -> Tuple[List, float, bool, Dict]:
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, you are responsible for calling `reset()` to reset
        the environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (np.ndarray of size 2): The action to take.
                                        1 is the land_use placement action.
                                        1 is the building road action.
            logger (Logger): The logger.
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # print ("in step,self._stage :",self._stage)
        # print ("in step,self._total_road_steps * self.build_ration:",self._total_road_steps * self.build_ration)
        # print ("in step,self._full_connected_steps:",self._full_connected_steps)
        # print ("in step,self._connecting_steps:",self._connecting_steps)
        # print ("self.stage2edges ",len(self._mg.stage2edges ))
        if self._done:
            raise RuntimeError('Action taken after episode is done.')
        else:
            if self._stage == 'connecting':
                self._action_history.append(action)
                self.build_road(action,POIVersionTag = False)  ########
                self._connecting_steps += 1
                if self._connecting_steps >= math.floor(
                        self._total_road_steps *
                        self.build_ration) or self._full_connected():
                    self.transition_stage()
            elif self._stage == 'full_connected':
                self._action_history.append(action)
                self.build_road(action,POIVersionTag = False)   #######
                self._full_connected_steps += 1
                # if (self._full_connected_steps + self._connecting_steps >        # original
                #         self._total_road_steps * self.build_ration):

                if (self._full_connected_steps + self._connecting_steps >      
                        9):
                                        
                    self.transition_stage()
            # print ("in step:", "total_cost",self._mg.total_cost(),self._stage)
            # print ("in step:",'f2POI_dis_avg',self._mg.f2f_avg)
            reward, info = self.get_reward_info()
            if self._stage == 'done':
                self.save_step_data()
        # print ("reward",reward)
        # converted_list = [float(arr[0]) for arr in self._action_history]
        # print ("self._action_history",converted_list)
        return self._get_obs(), reward, self._done, info


