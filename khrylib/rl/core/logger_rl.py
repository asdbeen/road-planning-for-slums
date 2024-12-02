import math
import itertools
from khrylib.utils.stats_logger import StatsLogger


class LoggerRL:

    def __init__(self, init_stats_logger=True):
        self.num_steps = 0
        self.num_episodes = 0
        self.sample_time = 0
        #self.stats_names = ['episode_len', 'reward', 'episode_reward_avg','episode_reward_dis','episode_reward_cost', 'interior_parcels_num', 'connecting_steps','f2f_dis_avg', 'total_road_cost']
        
        # self.stats_names.append('f2POI_dis_avg')      # "New added"
        self.stats_names = ['episode_len', 
                            'reward', 
                            'episode_reward_avg',
                            'interior_parcels_num',
                            'connecting_steps', 
                            'total_road_cost', 
                            'total_angle_cost',   
                            'total_hit_boundary_reward',
                            'total_explcit_connection_reward', 
                            'total_implcit_connection_reward',
                            'total_all_connection_reward',
                            'total_L_T_reward',
                            'total_RingRoad_reward',
                            'accumulated_reward'
                            ]

        if init_stats_logger:
            self.stats_loggers = {x: StatsLogger(is_nparray=False) for x in self.stats_names}
        self.plans = []

    def start_episode(self, env):
        self.episode_len = 0
        self.episode_reward = 0

    def step(self, env, reward, info):
        self.episode_len += 1
        # self.episode_reward += reward
        self.stats_loggers['reward'].log(reward)

    def end_episode(self, info):
        self.num_steps += self.episode_len
        self.num_episodes += 1
        self.stats_loggers['episode_len'].log(self.episode_len)

        self.stats_loggers['interior_parcels_num'].log(info['interior_parcels_num'])
        self.stats_loggers['connecting_steps'].log(info['connecting_steps'])
        self.stats_loggers['total_road_cost'].log(info['total_road_cost'])
        self.stats_loggers['total_angle_cost'].log(info['total_angle_cost'])
        self.stats_loggers['total_hit_boundary_reward'].log(info['total_hit_boundary_reward'])
        self.stats_loggers['total_explcit_connection_reward'].log(info['total_explcit_connection_reward'])
        self.stats_loggers['total_implcit_connection_reward'].log(info['total_implcit_connection_reward'])
        self.stats_loggers['total_all_connection_reward'].log(info['total_all_connection_reward'])
        self.stats_loggers['total_L_T_reward'].log(info['total_L_T_reward'])
        self.stats_loggers['total_RingRoad_reward'].log(info['total_RingRoad_reward'])

        self.stats_loggers['episode_reward_avg'].log(- 1*info['total_road_cost'])
        self.stats_loggers['accumulated_reward'].log(info['accumulated_reward'])

        #self.stats_loggers['episode_reward_avg'].log(-0.9*info['f2f_dis_avg'] - 0.1*info['total_road_cost'])
        #self.stats_loggers['episode_reward_avg'].log(-info['f2POI_dis_avg'])       # Now consider travel dist to POI also
        #self.stats_loggers['episode_reward_avg'].log(-info['f2POI_avg_EachCat_mean'])       # Now consider travel dist to POI also
        # self.stats_loggers['episode_reward_avg'].log(- 1*info['total_road_cost'])
        # self.stats_loggers['episode_reward_dis'].log(-info['f2f_dis_avg'])            
        # self.stats_loggers['episode_reward_cost'].log(-info['total_road_cost'])

        # self.stats_loggers['interior_parcels_num'].log(info['interior_parcels_num'])
        # 
        # self.stats_loggers['f2f_dis_avg'].log(info['f2f_dis_avg'])
        # self.stats_loggers['total_road_cost'].log(info['total_road_cost'])

        # self.stats_loggers['f2POI_dis_avg'].log(info['f2POI_dis_avg'])          # "New added"




        # print ("------")
        # print ("end_episode")
        # print ("self.num_episodes",self.num_episodes)
        # print (self.stats_loggers['f2f_dis_avg'].total_val)
        # print (self.stats_loggers['total_road_cost'].total_val)
        # print (-0.9*info['f2f_dis_avg'],0.1*info['total_road_cost'],self.stats_loggers['episode_reward_avg'].total_val)
        # print ("------")
    def add_plan(self, info_plan):
        self.plans.append(info_plan)

    @classmethod
    def merge(cls, logger_list, **kwargs):
        logger = cls(init_stats_logger=False, **kwargs)
        logger.num_episodes = sum([x.num_episodes for x in logger_list])
        logger.num_steps = sum([x.num_steps for x in logger_list])
        logger.stats_loggers = {}
        for stats in logger.stats_names:
            logger.stats_loggers[stats] = StatsLogger.merge([x.stats_loggers[stats] for x in logger_list])

        logger.total_reward = logger.stats_loggers['reward'].total()
        logger.avg_episode_len = logger.stats_loggers['episode_len'].avg()

  

        logger.avg_episode_reward = logger.stats_loggers['episode_reward_avg'].avg()
        # logger.dis_episode_reward = logger.stats_loggers['episode_reward_dis'].avg()
        # logger.cost_episode_reward = logger.stats_loggers['episode_reward_cost'].avg()

        # logger.interior_parcels_num = logger.stats_loggers['interior_parcels_num'].avg()
        # logger.connecting_steps = logger.stats_loggers['connecting_steps'].avg()
        # logger.face2face_avg = logger.stats_loggers['f2f_dis_avg'].avg()
        # logger.total_road_cost = logger.stats_loggers['total_road_cost'].avg()
        
        # logger.f2POI_dis_avg = logger.stats_loggers['f2POI_dis_avg'].avg()          # "New added"

        logger.interior_parcels_num = logger.stats_loggers['interior_parcels_num'].avg()
        logger.connecting_steps = logger.stats_loggers['connecting_steps'].avg()
        logger.total_road_cost = logger.stats_loggers['total_road_cost'].avg()
        logger.total_angle_cost = logger.stats_loggers['total_angle_cost'].avg()                            #
        logger.total_hit_boundary_reward = logger.stats_loggers['total_hit_boundary_reward'].avg()          #    
        logger.total_explcit_connection_reward = logger.stats_loggers['total_explcit_connection_reward'].avg()   #
        logger.total_implcit_connection_reward = logger.stats_loggers['total_implcit_connection_reward'].avg()   #
        logger.total_all_connection_reward = logger.stats_loggers['total_all_connection_reward'].avg()           #
        logger.total_L_T_reward = logger.stats_loggers['total_L_T_reward'].avg()           #
        logger.total_RingRoad_reward = logger.stats_loggers['total_RingRoad_reward'].avg()           #

        logger.accumulated_reward = logger.stats_loggers['accumulated_reward'].avg()  




        logger.plans = list(itertools.chain(*[var.plans for var in logger_list]))
        return logger
