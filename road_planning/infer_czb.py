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

else:
    sys.path.append(cwd) 
    sys.path.append(os.path.join(cwd,'road_planning'))
    sys.path.append(os.path.join(cwd,'road_planning/envs'))



######################################################################
#############################  Set up  ###############################
######################################################################
slum_name = 'tengah_1'

iteration = '0'

######################################################################
#########################  Original Code #############################
######################################################################

import setproctitle



import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from absl import app
from absl import flags

from khrylib.utils import *
from road_planning.utils.config import Config
from road_planning.agents.road_planning_agent import RoadPlanningAgent

flags.DEFINE_string('root_dir', os.path.join(cwd,'train_data') , 'Root directory for writing '
                                                                      'logs/summaries/checkpoints.')
flags.DEFINE_string('slum_name', slum_name, 'data_dir')                                                          # this is the name of ymal file 
flags.DEFINE_string('cfg', slum_name, 'Configuration file of rl training.')


flags.DEFINE_bool('tmp', False, 'Whether to use temporary storage.')
flags.DEFINE_bool('infer', True, 'Train or Infer.')                                                         #!!!!!!!!!!!!!!!!!

flags.DEFINE_bool('visualize', True, 'visualize plan.')
flags.DEFINE_enum('agent', 'rl-ngnn',
                  ['rl-sgnn', 'rl-ngnn', 'rl-mlp', 'rl-rmlp',
                   'random', 'road-cost'],
                  'Agent type.')
flags.DEFINE_integer('num_threads', 1, 'The number of threads for sampling trajectories.')
flags.DEFINE_bool('use_nvidia_gpu', True, 'Whether to use Nvidia GPU for acceleration.')
flags.DEFINE_integer('gpu_index', 0,'GPU ID.')
flags.DEFINE_integer('global_seed', 0, 'Used in env and weight initialization, does not impact action sampling.')
flags.DEFINE_string('iteration', '0', 'The start iteration. Can be number or best. If not 0, the agent will load from '
                                      'a saved checkpoint.')
flags.DEFINE_bool('restore_best_rewards', True, 'Whether to restore the best rewards from a saved checkpoint. '
                                                'True for resume training. False for finetune with new reward.')

FLAGS = flags.FLAGS


# def train_one_iteration(agent: RoadPlanningAgent, iteration: int) -> None:
    
#     """Train one iteration"""
#     agent.optimize(iteration)
#     agent.save_checkpoint(iteration)

#     """clean up gpu memory"""
#     torch.cuda.empty_cache()


def main_loop(_):

    setproctitle.setproctitle(f'road_planning_{FLAGS.cfg}_{FLAGS.global_seed}@suhy')

    cfg = Config(FLAGS.cfg, FLAGS.slum_name, FLAGS.global_seed, FLAGS.tmp, FLAGS.root_dir, FLAGS.agent)

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    if FLAGS.use_nvidia_gpu and torch.cuda.is_available():
        device = torch.device('cuda', index=FLAGS.gpu_index)
    else:
        device = torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(FLAGS.gpu_index)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    checkpoint = int(FLAGS.iteration) if FLAGS.iteration.isnumeric() else FLAGS.iteration

    #specificCheckPointPath = "/Users/chenzebin/Documents/GitHub/road-planning-for-slums/train_data/punggol/rl-ngnn/punggol/0/models/best.p"
    #specificCheckPointPath = "/Users/chenzebin/Documents/GitHub/road-planning-for-slums/train_data/punggol/rl-ngnn/punggol/0/models/iteration_0024.p"
    checkpoint = 20
    specificCheckPointPath = "/Users/chenzebin/Documents/GitHub/road-planning-for-slums/train_data/tengah_1/rl-ngnn/tengah_1/0/models/iteration_0021.p"
    #specificCheckPointPath = "/Users/chenzebin/Documents/GitHub/road-planning-for-slums/train_data/punggol/rl-ngnn/punggol/0/models/run2/iteration_0001.p"
    
    """create agent"""
    agent = RoadPlanningAgent(cfg=cfg, dtype=dtype, device=device, num_threads=FLAGS.num_threads,
                               training=True, checkpoint=checkpoint, restore_best_rewards=FLAGS.restore_best_rewards, specificCheckPointPath = specificCheckPointPath)

    agent.infer(visualize=FLAGS.visualize)
    print ("Done")

 

if __name__ == '__main__':
    # flags.mark_flags_as_required([
    #   'cfg'
    # ])
    #print ("hi")
    # print (FLAGS.iteration)
    app.run(main_loop)




#tensorboard --logdir=/Users/chenzebin/Documents/GitHub/road-planning-for-slums/train_data/punggol/rl-ngnn/punggol/0/tb
#ps aux | grep tensorboard
#kill <PID>