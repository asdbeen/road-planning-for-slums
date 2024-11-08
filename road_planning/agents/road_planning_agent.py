import math
import pickle
import time

import torch

from khrylib.utils import *
from khrylib.utils.torch import *
from khrylib.rl.agents import AgentPPO
from khrylib.rl.core import estimate_advantages, LoggerRL
from torch.utils.tensorboard import SummaryWriter
from road_planning.envs import RoadEnv
from road_planning.models.model import create_sgnn_model, create_dgnn_model, create_ngnn_model, create_mlp_model, create_rmlp_model, ActorCritic
from road_planning.models.baseline import RandomPolicy, RoadCostPolicy,  GAPolicy, NullModel
from road_planning.utils.tools import TrajBatchDisc
from road_planning.utils.config import Config

import os
import sys

def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) for x in y] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]


class RoadPlanningAgent(AgentPPO):


    def print_model_parameters_grad(self,model, file_path):
        # Print Gradient Values
        # print ("print_model_parameters_grad")
        with open(file_path, 'a') as f:
            for name, param in model.named_parameters():
  
                if param.grad is not None:
                    f.write(f"Parameter name: {name}\n")
                    f.write(f"Gradient norm: {param.grad.norm().item()}")
                    f.write("\n") 
                    #print (f"Gradient norm: {param.grad.norm().item()}")

    def print_model_parameters(self,model, file_path):

        """Print the parameters of a given model to a file."""
        #print ("print_model_parameters_grad")
        with open(file_path, 'a') as f:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    f.write(f"Parameter name: {name}\n")
                    f.write(f"Shape: {param.shape}\n")
                    f.write(f"Values: {param}\n")
                    f.write("\n")  # Add a blank line between parameters
    
    def __init__(self,
                 cfg: Config,
                 dtype: torch.dtype,
                 device: torch.device,
                 num_threads: int,
                 training: bool = True,
                 checkpoint: Union[int, Text] = 0,
                 restore_best_rewards: bool = True,
                 specificCheckPointPath = None
                 ):
      
        if cfg.train_file_num == 1:
            self.cfg = cfg
            self.training = training
            self.device = device
            self.loss_iter = 0
            self.setup_logger(num_threads)

            self.setup_env()
            self.setup_model()
            self.setup_optimizer()

         
            if checkpoint != 0:
                self.start_iteration = self.load_checkpoint(
                    checkpoint, restore_best_rewards,specificCheckPointPath)
           
            else:
                self.start_iteration = 0
            super().__init__(env=self.env,
                            dtype=dtype,
                            device=device,
                            logger_cls=LoggerRL,
                            traj_cls=TrajBatchDisc,
                            num_threads=num_threads,
                            policy_net=self.policy_net,
                            value_net=self.value_net,
                            optimizer=self.optimizer,
                            opt_num_epochs=cfg.num_optim_epoch,
                            gamma=cfg.gamma,
                            tau=cfg.tau,
                            clip_epsilon=cfg.clip_epsilon,
                            value_pred_coef=cfg.value_pred_coef,
                            entropy_coef=cfg.entropy_coef,
                            policy_grad_clip=[(self.policy_net.parameters(), 1),
                                            (self.value_net.parameters(), 1)],
                            mini_batch_size=cfg.mini_batch_size)

        elif cfg.train_file_num != 1:
            self.cfg = cfg
            self.training = training
            self.device = device
            self.loss_iter = 0
            self.setup_logger(num_threads)

            self.setup_env_multi()    # it defaultly set the first env
            self.setup_model()
            self.setup_optimizer()
            if checkpoint != 0:
                self.start_iteration = self.load_checkpoint(
                    checkpoint, restore_best_rewards,specificCheckPointPath)
            else:
                self.start_iteration = 0
            super().__init__(env=self.env,
                            dtype=dtype,
                            device=device,
                            logger_cls=LoggerRL,
                            traj_cls=TrajBatchDisc,
                            num_threads=num_threads,
                            policy_net=self.policy_net,
                            value_net=self.value_net,
                            optimizer=self.optimizer,
                            opt_num_epochs=cfg.num_optim_epoch,
                            gamma=cfg.gamma,
                            tau=cfg.tau,
                            clip_epsilon=cfg.clip_epsilon,
                            value_pred_coef=cfg.value_pred_coef,
                            entropy_coef=cfg.entropy_coef,
                            policy_grad_clip=[(self.policy_net.parameters(), 1),
                                            (self.value_net.parameters(), 1)],
                            mini_batch_size=cfg.mini_batch_size)
        
        # print ("---")

        
        cwd = os.getcwd()
        if "road_planning" in cwd:      # for ssh remote terminal
            cwd = os.path.dirname(cwd)
        else:
            cwd = cwd
        

        #### Check the model parameters
        
        file_path = os.path.join(cwd,'debug','debug_init.txt')
        self.print_model_parameters(self.actor_critic_net.actor_net.policy_road_head,file_path)
        self.print_model_parameters(self.actor_critic_net.value_net.value_head,file_path)

        #### Check the model parameters grad
        file_path = os.path.join(cwd,'debug','debug_init_grad.txt')
        self.print_model_parameters_grad(self.actor_critic_net.actor_net.policy_road_head,file_path)
        self.print_model_parameters_grad(self.actor_critic_net.value_net.value_head,file_path)




 
    def sample_worker(self, pid, queue, num_samples, mean_action):  # This is the one in use
        self.seed_worker(pid)
        memory = Memory()
        logger = self.logger_cls(**self.logger_kwargs)
       
        while logger.num_steps < num_samples:
            state = self.env.reset()
            
            last_info = dict()
            episode_success = False
            logger_messages = []
            memory_messages = []
            for t in range(10000):   
              
                state_var = tensorfy([state])
                #### self.noise_rate == 1 meaning it is totally depend on mean_action
                use_mean_action = mean_action or torch.bernoulli(
                    torch.tensor([1 - self.noise_rate])).item()
                # action = self.policy_net.select_action(state_var, use_mean_action).numpy().squeeze(0)
                
                #print ("select_action")
                action = self.policy_net.select_action(
                    state_var, use_mean_action).numpy().squeeze(0)
                
                #print ("before step")
                #print ("in sample_worker_action",action)
                next_state, reward, done, info = self.env.step(
                    action, self.thread_loggers[pid])
                
                # print ("after step")
                # print ("-----------")
                # cache logging
                logger_messages.append([reward, info])

                mask = 0 if done else 1     
                exp = 1 - use_mean_action
                # cache memory
                memory_messages.append(
                    [state, action, mask, next_state, reward, exp])
                
                if done:
                    episode_success = (reward != self.env.FAILURE_REWARD) and (
                        reward != self.env.INTERMEDIATE_REWARD)
                    last_info = info
                    break
                state = next_state


            print ("sample_worker_episode_success",episode_success, "current logger.num_steps",logger.num_steps)

            # try:
            #     print ("self._action_history", [arr.item() for arr in self.env._action_history])
             
            # except:
            #     pass

            if episode_success:
                logger.start_episode(self.env)
                for var in range(len(logger_messages)):
                    logger.step(self.env, *logger_messages[var])
                    self.push_memory(memory, *memory_messages[var])
                logger.end_episode(last_info)
                
            #     self.thread_loggers[pid].info(
            #         'worker {} finished episode {}.'.format(
            #             pid, logger.num_episodes))
            # else:
            #     self.thread_loggers[pid].info(
            #         '[!]worker {} failed episode {}.'.format(
            #             pid, logger.num_episodes))

        print ("logger.num_steps",logger.num_steps)   
        if queue is not None:
            queue.put([pid, memory, logger])
        else:
            return memory, logger

    def setup_env(self):
        self.env = env = RoadEnv(self.cfg)
        self.numerical_feature_size = env.get_numerical_feature_size()
        self.node_dim = env.get_node_dim()

    def setup_env_multi(self):
        multi_envs = []
        multi_numerical_feature_size = []
        multi_node_dim = []
        originalSlumName = self.cfg.slum
        for i in range(1,self.cfg.train_file_num+1,1):   # becasue it counts from 1
            self.cfg.slum += "_" + str(i)
            env = RoadEnv(self.cfg)
            multi_envs.append(env)
            multi_numerical_feature_size.append(env.get_numerical_feature_size())
            multi_node_dim.append(env.get_node_dim())

            self.cfg.slum = originalSlumName

        self.multi_envs = multi_envs
        self.multi_numerical_feature_size = multi_numerical_feature_size
        self.multi_node_dim = multi_node_dim

        self.env = multi_envs[0]
        self.numerical_feature_size = multi_numerical_feature_size[0]
        self.node_dim = multi_node_dim[0]

    def setup_logger(self, num_threads):
        cfg = self.cfg
        self.tb_logger = SummaryWriter(cfg.tb_dir) if self.training else None
        self.logger = create_logger(os.path.join(
            cfg.log_dir, f'log_{"train" if self.training else "eval"}.txt'),
                                    file_handle=True)
        self.reward_offset = 0.0
        self.best_rewards = -1000.0
        self.best_plans = []
        self.current_rewards = -1000.0
        self.current_plans = []
        self.save_best_flag = False
        cfg.log(self.logger, self.tb_logger)

        self.thread_loggers = []
        for i in range(num_threads):
            self.thread_loggers.append(
                create_logger(os.path.join(
                    cfg.log_dir,
                    f'log_{"train" if self.training else "eval"}_{i}.txt'),
                              file_handle=True))

    def setup_model(self):
        cfg = self.cfg
        if cfg.agent == 'rl-sgnn':
            self.policy_net, self.value_net = create_sgnn_model(cfg, self)
            self.actor_critic_net = ActorCritic(self.policy_net,
                                                self.value_net)
            to_device(self.device, self.actor_critic_net)
        elif cfg.agent == 'rl-dgnn':
            self.policy_net, self.value_net = create_dgnn_model(cfg, self)
            self.actor_critic_net = ActorCritic(self.policy_net,
                                                self.value_net)
            to_device(self.device, self.actor_critic_net)
        elif cfg.agent == 'rl-ngnn':                                            #using this one
            self.policy_net, self.value_net = create_ngnn_model(cfg, self)
            self.actor_critic_net = ActorCritic(self.policy_net,
                                                self.value_net)
            to_device(self.device, self.actor_critic_net)
        elif cfg.agent == 'rl-mlp':
            self.policy_net, self.value_net = create_mlp_model(cfg, self)
            self.actor_critic_net = ActorCritic(self.policy_net,
                                                self.value_net)
            to_device(self.device, self.actor_critic_net)
        elif cfg.agent == 'rl-rmlp':
            self.policy_net, self.value_net = create_rmlp_model(cfg, self)
            self.actor_critic_net = ActorCritic(self.policy_net,
                                                self.value_net)
            to_device(self.device, self.actor_critic_net)
        elif cfg.agent == 'random':
            self.policy_net = RandomPolicy()
            self.value_net = NullModel()
        # elif cfg.agent == 'travel_distance':
        #     self.policy_net = TravelDistancePolicy()
        #     self.value_net = NullModel()
        elif cfg.agent == 'road-cost':
            self.policy_net = RoadCostPolicy()
            self.value_net = NullModel()
        elif cfg.agent == 'ga':
            self.policy_net = GAPolicy()
            self.value_net = NullModel()
        else:
            raise NotImplementedError()

    def setup_optimizer(self):
        cfg = self.cfg
        if cfg.agent in ['rl-sgnn', 'rl-dgnn', 'rl-ngnn', 'rl-mlp', 'rl-rmlp']:
            self.optimizer = torch.optim.Adam(
                self.actor_critic_net.parameters(),
                lr=cfg.lr,
                eps=cfg.eps,
                weight_decay=cfg.weightdecay)
        else:
            self.optimizer = None

    def load_checkpoint(self, checkpoint, restore_best_rewards,specificCheckPointPath = None):
        cfg = self.cfg
        if specificCheckPointPath == None:
            if isinstance(checkpoint, int):
                cp_path = '%s/iteration_%04d.p' % (cfg.model_dir, checkpoint)
            else:
                assert isinstance(checkpoint, str)
                cp_path = '%s/%s.p' % (cfg.model_dir, checkpoint)
        else:
            cp_path = specificCheckPointPath

        self.logger.info('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        self.actor_critic_net.load_state_dict(model_cp['actor_critic_dict'])
        self.loss_iter = model_cp['loss_iter']
        if restore_best_rewards:
            self.best_rewards = model_cp.get('best_rewards', self.best_rewards)
            self.best_plans = model_cp.get('best_plans', self.best_plans)
        self.current_rewards = model_cp.get('current_rewards',
                                            self.current_rewards)
        self.current_plans = model_cp.get('current_plans', self.current_plans)
        start_iteration = model_cp['iteration'] + 1

        #print ("in load_checkpoint",type(self.actor_critic_net))

  
        return start_iteration

    def save_checkpoint(self, iteration):

        def save(cp_path):
            with to_cpu(self.policy_net, self.value_net):
                model_cp = {
                    'actor_critic_dict': self.actor_critic_net.state_dict(),
                    'loss_iter': self.loss_iter,
                    'best_rewards': self.best_rewards,
                    'best_plans': self.best_plans,
                    'current_rewards': self.current_rewards,
                    'current_plans': self.current_plans,
                    'iteration': iteration
                }
                pickle.dump(model_cp, open(cp_path, 'wb'))

        cfg = self.cfg

        if cfg.save_model_interval > 0 and (iteration +
                                            1) % cfg.save_model_interval == 0:
            self.tb_logger.flush()
            save('{}/iteration_{:04d}.p'.format(cfg.model_dir, iteration + 1))
        if self.save_best_flag:
            self.tb_logger.add_scalar('best_reward/best_reward',
                                      self.best_rewards, iteration)
            self.tb_logger.flush()
            self.logger.info(
                f'save best checkpoint with rewards {self.best_rewards:.2f}!')
            save('{}/best.p'.format(cfg.model_dir))
            save('{}/best_reward{:.2f}_iteration_{:04d}.p'.format(
                cfg.model_dir, self.best_rewards, iteration + 1))

    def save_plan(self, log_eval: LoggerRL) -> None:
        """
        Save the current plan to file.

        Args:
            log_eval: LoggerRL object.
        """
        cfg = self.cfg
        # self.logger.info(f'save plan to file: {cfg.plan_dir}\plan.p')
        # with open(f'{cfg.plan_dir}\plan.p', 'wb') as f:
        plan_file_path = os.path.join(cfg.plan_dir,'plan.p')
        self.logger.info(f'save plan to file: {plan_file_path}')
        with open(plan_file_path, 'wb') as f:
            pickle.dump(log_eval.plans, f)


    def optimize(self, iteration):
        info = self.optimize_policy(iteration)
        self.log_optimize_policy(iteration, info)

    def optimize_policy(self, iteration):
        """generate multiple trajectories that reach the minimum batch_size"""
        t0 = time.time()
        
        num_samples = self.cfg.num_episodes_per_iteration * self.cfg.max_sequence_length
        print ("optimize_policy_ start sample")    
        print ("self.env._mg.road_edges",len(self.env._mg.road_edges))
        if self.cfg.train_file_num != 1:
            for i in range (len(self.multi_envs)):     # shift the environment
                self.env = self.multi_envs[i]    
                self.numerical_feature_size = self.multi_numerical_feature_size[i]
                self.node_dim = self.multi_node_dim[i] 
                batch, log = self.sample(num_samples)
                print ("i am using this env",self.env._mg)
        else:
            batch, log = self.sample(num_samples)

            
        # print ("self.env._connecting_steps",self.env._connecting_steps)
        # print ("self.env._full_connected_steps",self.env._full_connected_steps)

       
        """update networks"""
        t1 = time.time()


    
        print ("optimize_policy_ start update_params")  
        print ("self.env._mg.road_edges",len(self.env._mg.road_edges))
        self.update_params(batch, iteration)

        
        ######################
        ### Save the network
        ###################### 

        #### Check the model parameters
        cwd = os.getcwd()   
        if "road_planning" in cwd:      # for ssh remote terminal
            cwd = os.path.dirname(cwd)
        else:
            cwd = cwd
        file_path = os.path.join(cwd,'debug',f'debug{iteration}.txt')
        self.print_model_parameters(self.actor_critic_net.actor_net.policy_road_head,file_path)
        self.print_model_parameters(self.actor_critic_net.value_net.value_head,file_path)

        #### Check the model parameters grad
        file_path = os.path.join(cwd,'debug',f'debug{iteration}_grad.txt')
        self.print_model_parameters_grad(self.actor_critic_net.actor_net.policy_road_head,file_path)
        self.print_model_parameters_grad(self.actor_critic_net.value_net.value_head,file_path)

        t2 = time.time()
        """evaluate policy"""
        print ("optimize_policy_ start eval_agent")  
        if self.cfg.train_file_num != 1:
            self.env = self.multi_envs[i]    
            self.numerical_feature_size = self.multi_numerical_feature_size[i]
            self.node_dim = self.multi_node_dim[i] 
       
        else:
            pass

        log_eval = self.eval_agent(num_samples=1, mean_action=True,visualize=True,iteration=iteration)
        t3 = time.time()
        

        info = {
            'log': log,
            'log_eval': log_eval,
            'T_sample': t1 - t0,
            'T_update': t2 - t1,
            'T_eval': t3 - t2,
            'T_total': t3 - t0,
            'f2POI_avg':self.env._mg.f2POI_avg,
            #'f2POI_avg_EachCat_mean':self.env._mg.f2POI_avg_EachCat_mean
        }
        return info

    def update_params(self, batch, iteration):
        t0 = time.time()
        to_train(*self.update_modules)
        states = batch.states
        actions = torch.from_numpy(batch.actions).to(self.dtype)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype)
        masks = torch.from_numpy(batch.masks).to(self.dtype)
        exps = torch.from_numpy(batch.exps).to(self.dtype)
       
        with to_test(*self.update_modules):
            with torch.no_grad():
                values = []
                chunk = self.cfg.mini_batch_size
                for i in range(0, len(states), chunk):
                    states_i = tensorfy(states[i:min(i + chunk, len(states))],
                                        self.device)
                    values_i = self.value_net(self.trans_value(states_i))
                    values.append(values_i.cpu())
                values = torch.cat(values)
        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values,
                                                  self.gamma, self.tau)
        
     
        self.update_policy(states, actions, returns, advantages, exps,
                           iteration)

        return time.time() - t0

    def get_perm_batch_stage(self, states):
        inds = [[], []]
        for i, x in enumerate(states):
            stage = x[-1]
            inds[stage.argmax()].append(i)
        perm = np.array(inds[0] + inds[1])
        return perm, LongTensor(perm)

    def update_policy(self, states, actions, returns, advantages, exps,
                      iteration):


        """update policy"""
        with to_test(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = []
                chunk = self.cfg.mini_batch_size
                for i in range(0, len(states), chunk):
                    states_i = tensorfy(states[i:min(i + chunk, len(states))],
                                        self.device)
                    actions_i = actions[i:min(i + chunk, len(states))].to(
                        self.device)
                    fixed_log_probs_i, _ = self.policy_net.get_log_prob_entropy(
                        self.trans_policy(states_i), actions_i)
                    fixed_log_probs.append(fixed_log_probs_i.cpu())
                fixed_log_probs = torch.cat(fixed_log_probs)
        num_state = len(states)

        tb_logger = self.tb_logger
        total_loss = 0.0
        total_value_loss = 0.0
        total_surr_loss = 0.0
        total_entropy_loss = 0.0
        for epoch in range(self.opt_num_epochs):

            epoch_loss = 0.0
            epoch_value_loss = 0.0
            epoch_surr_loss = 0.0
            epoch_entropy_loss = 0.0

            perm_np = np.arange(num_state)
            np.random.shuffle(perm_np)
            perm = LongTensor(perm_np)

            states, actions, returns, advantages, fixed_log_probs, exps = \
                index_select_list(states, perm_np), actions[perm].clone(), returns[perm].clone(), \
                advantages[perm].clone(), fixed_log_probs[perm].clone(), exps[perm].clone()

            if self.cfg.agent_specs.get('batch_stage', False):

                perm_stage_np, perm_stage = self.get_perm_batch_stage(states)
                states, actions, returns, advantages, fixed_log_probs, exps = \
                    index_select_list(states, perm_stage_np), actions[perm_stage].clone(), \
                    returns[perm_stage].clone(), advantages[perm_stage].clone(), \
                    fixed_log_probs[perm_stage].clone(), exps[perm_stage].clone()

            # optim_batch_num = int(math.floor(num_state / self.mini_batch_size))
            # print ("optim_batch_num ",optim_batch_num )
            optim_batch_num  =  1 # Temporarily set to 1
            for i in range(optim_batch_num):
                
                ind = slice(i * self.mini_batch_size,
                            min((i + 1) * self.mini_batch_size, num_state))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b, exps_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind], exps[ind]
                ind = exps_b.nonzero(as_tuple=False).squeeze(1)
                states_b = tensorfy(states_b, self.device)
                actions_b, advantages_b, returns_b, fixed_log_probs_b, ind = batch_to(
                    self.device, actions_b, advantages_b, returns_b,
                    fixed_log_probs_b, ind)
                value_loss = self.value_loss(states_b, returns_b)
                surr_loss, entropy_loss = self.ppo_entropy_loss(
                    states_b, actions_b, advantages_b, fixed_log_probs_b, ind)
                loss = surr_loss + self.value_pred_coef * value_loss + self.entropy_coef * entropy_loss
                self.optimizer.zero_grad()
                
                
                # print ("loss",loss)
                # print("Before backward:")
                # print ("actor_net")
                # for name, param in self.actor_critic_net.actor_net.policy_road_head.named_parameters():
                #     if param.requires_grad:
                #         print(f"Parameter name: {name}\n")
                #         print(f"Shape: {param.shape}\n")
                #         print(f"Values: {param}\n")
                #         print("\n")  # Add a blank line between parameters
                # print ("value_net")
                # for name, param in self.actor_critic_net.value_net.value_head.named_parameters():
                #     if param.requires_grad:
                #         print(f"Parameter name: {name}\n")
                #         print(f"Shape: {param.shape}\n")
                #         print(f"Values: {param}\n")
                #         print("\n")  # Add a blank line between parameters   
                loss.backward()

                # print ("----------")
                # print("Before backward:")
                # print ("actor_net")
                # for name, param in self.actor_critic_net.actor_net.policy_road_head.named_parameters():
                #     if param.requires_grad:
                #         print(f"Parameter name: {name}\n")
                #         print(f"Shape: {param.shape}\n")
                #         print(f"Values: {param}\n")
                #         print("\n")  # Add a blank line between parameters
                # print ("value_net")
                # for name, param in self.actor_critic_net.value_net.value_head.named_parameters():
                #     if param.requires_grad:
                #         print(f"Parameter name: {name}\n")
                #         print(f"Shape: {param.shape}\n")
                #         print(f"Values: {param}\n")
                #         print("\n")  # Add a blank line between parameters   
                


                self.clip_policy_grad()
                self.optimizer.step()



                  

                epoch_loss += loss.item()
                epoch_value_loss += value_loss.item()
                epoch_surr_loss += surr_loss.item()
                epoch_entropy_loss += entropy_loss.item()
                tb_logger.add_scalar('loss/loss', loss.item(), self.loss_iter)
                tb_logger.add_scalar('loss/value_loss', value_loss.item(),
                                     self.loss_iter)
                tb_logger.add_scalar('loss/surr_loss', surr_loss.item(),
                                     self.loss_iter)
                tb_logger.add_scalar('loss/entropy_loss', entropy_loss.item(),
                                     self.loss_iter)
                self.loss_iter += 1

                #print ("loss.item()",loss.item(), "value_loss.item()",value_loss.item(), "surr_loss.item()",surr_loss.item(),"entropy_loss.item()",entropy_loss.item())
            total_loss += epoch_loss
            total_value_loss += epoch_value_loss
            total_surr_loss += epoch_surr_loss
            total_entropy_loss += epoch_entropy_loss
            global_epoch = iteration * self.opt_num_epochs + epoch
            tb_logger.add_scalar('loss/epoch_loss', epoch_loss, global_epoch)
            tb_logger.add_scalar('loss/epoch_value_loss', epoch_value_loss,
                                 global_epoch)
            tb_logger.add_scalar('loss/epoch_surr_loss', epoch_surr_loss,
                                 global_epoch)
            tb_logger.add_scalar('loss/epoch_entropy_loss', epoch_entropy_loss,
                                 global_epoch)

        tb_logger.add_scalar('loss/total_loss',
                             total_loss / self.opt_num_epochs, iteration)
        tb_logger.add_scalar('loss/total_value_loss',
                             total_value_loss / self.opt_num_epochs, iteration)
        tb_logger.add_scalar('loss/total_surr_loss',
                             total_surr_loss / self.opt_num_epochs, iteration)
        tb_logger.add_scalar('loss/total_entropy_loss',
                             total_entropy_loss / self.opt_num_epochs,
                             iteration)
        



    
    def ppo_entropy_loss(self, states, actions, advantages, fixed_log_probs,
                         ind):
        log_probs, entropy = self.policy_net.get_log_prob_entropy(
            self.trans_policy(states), actions)
        ratio = torch.exp(log_probs[ind] - fixed_log_probs[ind])
        advantages = advantages[ind]
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                            1.0 + self.clip_epsilon) * advantages
        surr_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -entropy[ind].mean()
        return surr_loss, entropy_loss

    def log_optimize_policy(self, iteration, info):
        cfg = self.cfg
  
        log, log_eval = info['log'], info['log_eval']
       
        f2POI_dis_avg = info['f2POI_avg']
        #f2POI_avg_EachCat_mean = info['f2POI_avg_EachCat_mean']
        logger, tb_logger = self.logger, self.tb_logger

        log_str = f'{iteration}\tT_sample {info["T_sample"]:.2f}\tT_update {info["T_update"]:.2f}\t' \
                  f'T_eval {info["T_eval"]:.2f}\t' \
                  f'ETA {get_eta_str(iteration, cfg.max_num_iterations, info["T_total"])}\t' \
                  f'train_R_eps {log.avg_episode_reward + self.reward_offset:.2f}\t'\
                  f'eval_R_eps {log_eval.avg_episode_reward + self.reward_offset:.2f}\t{cfg.id}\t'\
                  #f' {f2POI_avg_EachCat_mean}\t'
                  

        logger.info(log_str)

        self.current_rewards = log_eval.avg_episode_reward + self.reward_offset
        self.current_plans = log_eval.plans
        if log_eval.avg_episode_reward + self.reward_offset > self.best_rewards:
            self.best_rewards = log_eval.avg_episode_reward + self.reward_offset
            self.best_plans = log_eval.plans
            self.save_best_flag = True
        else:
            self.save_best_flag = False
        
        tb_logger.add_scalar('train/train_R_eps_avg',
                             log.avg_episode_reward + self.reward_offset,
                             iteration)
        tb_logger.add_scalar('train/interior_parcels_num',
                             log.interior_parcels_num, iteration)
        tb_logger.add_scalar('train/connecting_steps', log.connecting_steps,
                             iteration)
        tb_logger.add_scalar('train/f2f_dis_avg', log.face2face_avg, iteration)
        tb_logger.add_scalar('train/total_road_cost', log.total_road_cost,
                             iteration)

        tb_logger.add_scalar('train/f2POI_dis_avg', log_eval.f2POI_dis_avg,   # new added
                    iteration)
        #####
        tb_logger.add_scalar('eval/eval_R_eps_avg',
                             log_eval.avg_episode_reward + self.reward_offset,   #avag_episode_reward
                             iteration)
        tb_logger.add_scalar('eval/eval_R_eps_dis',
                             log_eval.dis_episode_reward + self.reward_offset,
                             iteration)
        tb_logger.add_scalar('eval/eval_R_eps_cost',
                             log_eval.cost_episode_reward + self.reward_offset,
                             iteration)

        tb_logger.add_scalar('eval/interior_parcels_num',
                             log_eval.interior_parcels_num, iteration)
        tb_logger.add_scalar('eval/connecting_steps',
                             log_eval.connecting_steps, iteration)
        tb_logger.add_scalar('eval/f2f_dis_avg', log_eval.face2face_avg,
                             iteration)
        tb_logger.add_scalar('eval/total_road_cost', log_eval.total_road_cost,
                             iteration)
  
        tb_logger.add_scalar('eval/f2POI_dis_avg', log_eval.f2POI_dis_avg,   # new added
                            iteration)
        
        ####### Each category POI distance
        # tb_logger.add_scalar('EachPOIDist/f2POI_avg_EachCat_A',
        #                      self.env._mg.f2POI_avg_EachCat_A, iteration)
        # tb_logger.add_scalar('EachPOIDist/f2POI_avg_EachCat_B',
        #                     self.env._mg.f2POI_avg_EachCat_B, iteration)
        # tb_logger.add_scalar('EachPOIDist/f2POI_avg_EachCat_C',
        #                      self.env._mg.f2POI_avg_EachCat_C, iteration)
        # tb_logger.add_scalar('EachPOIDist/f2POI_avg_EachCat_mean',
        #                      self.env._mg.f2POI_avg_EachCat_mean, iteration)
        
        # print("log 属性:")
        # for attr in vars(log):
        #     print(f"{attr}: {getattr(log, attr)}")

        # print ("------")
        # print("log_eval 属性:")
        # for attr in vars(log_eval):
        #     print(f"{attr}: {getattr(log_eval, attr)}")

        # print ("------")
        # print ("log.stats_loggers.n",log.stats_loggers['reward'].n)


    def eval_agent(self, num_samples=1, mean_action=True, visualize=True,iteration = None):
        print ("eval_agent",iteration)
        t_start = time.time()
        to_test(*self.sample_modules)
        self.env.eval()
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                logger = self.logger_cls(**self.logger_kwargs)
                
                while logger.num_steps < num_samples:
                    state = self.env.reset()
                  
                    if visualize:
                        if iteration == None:
                            os.makedirs(os.path.join(self.cfg.plan_dir,"eva_"), exist_ok=True)
                            self.env.visualize(save_fig=True,
                                            path=os.path.join(
                                                self.cfg.plan_dir,"eva_",
                                                'origin.svg'))
                        else:
                            os.makedirs(os.path.join(self.cfg.plan_dir,"eva_"+ str(iteration)), exist_ok=True)
                            self.env.visualize(save_fig=True,
                                            path=os.path.join(
                                                self.cfg.plan_dir,"eva_"+ str(iteration),  # add a folder for each iteration
                                                'origin.svg'))
            
                    logger.start_episode(self.env)

                    info_plan = dict()
                    episode_success = False
                    for t in range(1, 10000):  
                        state_var = tensorfy([state])
                        action = self.policy_net.select_action(
                            state_var, mean_action).numpy()
                        next_state, reward, done, info = self.env.step(
                            action, self.logger)
                        logger.step(self.env, reward, info)
                        # self.logger.info(f'reward:{reward}  step:{t:02d}')
              
                        if visualize:

                            if iteration == None:
                                os.makedirs(os.path.join(self.cfg.plan_dir,"eva_"), exist_ok=True)
                                self.env.visualize(save_fig=True,
                                                path=os.path.join(
                                                    self.cfg.plan_dir,"eva_",
                                                    f'step_all_{t:04d}.svg'))
                            else:
                                os.makedirs(os.path.join(self.cfg.plan_dir,"eva_"+ str(iteration)), exist_ok=True)
                                self.env.visualize(save_fig=True,
                                                path=os.path.join(
                                                    self.cfg.plan_dir,"eva_"+ str(iteration),  # add a folder for each iteration
                                                    f'step_all_{t:04d}.svg'))
                        
                        
                  
                        if done:
                            episode_success = (reward != self.env.FAILURE_REWARD) and \
                                              (reward != self.env.INTERMEDIATE_REWARD)
                            info_plan = info
                            break
                        state = next_state

                    logger.add_plan(info_plan)
                    logger.end_episode(info_plan)
                    if not episode_success:
                        self.logger.info('Plan fails during eval.')
                    else:
                        if iteration == None:
                            os.makedirs(os.path.join(self.cfg.plan_dir,"eva_"), exist_ok=True)
                            self.env.visualize(save_fig=True,
                                            path=os.path.join(
                                                self.cfg.plan_dir,"eva_",
                                                'final.svg'))
                        else:
                            os.makedirs(os.path.join(self.cfg.plan_dir,"eva_"+ str(iteration)), exist_ok=True)
                            self.env.visualize(save_fig=True,
                                            path=os.path.join(
                                                self.cfg.plan_dir,"eva_"+ str(iteration),  # add a folder for each iteration
                                                'final.svg'))

                logger = self.logger_cls.merge([logger], **self.logger_kwargs)
   

        self.env.train()
        logger.sample_time = time.time() - t_start
        return logger

    def eval_agent_infer(self, num_samples=1, mean_action=True, visualize=True,iteration = None):
        print ("eval_agent_infer",iteration)

        #print ("road_",self.env._mg.edge_list)
        # info = []
        # for edge in self.env._mg.edge_list:
        #     for node in edge.nodes:
        #         info.append(node.x)
        #         info.append(node.y)

        # print (info)

        t_start = time.time()
        to_test(*self.sample_modules)
        self.env.eval()

 

        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                logger = self.logger_cls(**self.logger_kwargs)
                
                while logger.num_steps < num_samples:
                    state = self.env.reset()
            
                    if visualize:
                        if iteration == None:
                            os.makedirs(os.path.join(self.cfg.plan_dir,"eva_infer"), exist_ok=True)
                            self.env.visualize(save_fig=True,
                                            path=os.path.join(
                                                self.cfg.plan_dir,"eva_infer",
                                                'origin.svg'))
                        else:
                            os.makedirs(os.path.join(self.cfg.plan_dir,"eva_infer"+ str(iteration)), exist_ok=True)
                            self.env.visualize(save_fig=True,
                                            path=os.path.join(
                                                self.cfg.plan_dir,"eva_infer"+ str(iteration),  # add a folder for each iteration
                                                'origin.svg'))
            
                    logger.start_episode(self.env)

                    info_plan = dict()
                    episode_success = False
                    for t in range(1, 10000):  
                        state_var = tensorfy([state])
                        action = self.policy_net.select_action(
                            state_var, mean_action).numpy()
                        
                        next_state, reward, done, info = self.env.step(
                            action, self.logger)
                        logger.step(self.env, reward, info)
                        # self.logger.info(f'reward:{reward}  step:{t:02d}')
              
                        if visualize:

                            if iteration == None:
                                os.makedirs(os.path.join(self.cfg.plan_dir,"eva_infer"), exist_ok=True)
                                self.env.visualize(save_fig=True,
                                                path=os.path.join(
                                                    self.cfg.plan_dir,"eva_infer",
                                                    f'step_all_{t:04d}.svg'))
                            else:
                                os.makedirs(os.path.join(self.cfg.plan_dir,"eva_infer"+ str(iteration)), exist_ok=True)
                                self.env.visualize(save_fig=True,
                                                path=os.path.join(
                                                    self.cfg.plan_dir,"eva_infer"+ str(iteration),  # add a folder for each iteration
                                                    f'step_all_{t:04d}.svg'))
                  
                        if done:
                            episode_success = (reward != self.env.FAILURE_REWARD) and \
                                              (reward != self.env.INTERMEDIATE_REWARD)
                            info_plan = info
                            break
                        state = next_state

                    logger.add_plan(info_plan)
                    logger.end_episode(info_plan)
                    if not episode_success:
                        self.logger.info('Plan fails during eval.')
                    else:
                        if iteration == None:
                            os.makedirs(os.path.join(self.cfg.plan_dir,"eva_infer"), exist_ok=True)
                            self.env.visualize(save_fig=True,
                                            path=os.path.join(
                                                self.cfg.plan_dir,"eva_infer",
                                                'final.svg'))
                        else:
                            os.makedirs(os.path.join(self.cfg.plan_dir,"eva_infer"+ str(iteration)), exist_ok=True)
                            self.env.visualize(save_fig=True,
                                            path=os.path.join(
                                                self.cfg.plan_dir,"eva_infer"+ str(iteration),  # add a folder for each iteration
                                                'final.svg'))
                logger = self.logger_cls.merge([logger], **self.logger_kwargs)

        self.env.train()
        logger.sample_time = time.time() - t_start

        return logger
    
    def infer(self,
              num_samples=1,
              mean_action=True,
              visualize=False,
              save_video=False,
              only_road=False,
              ):
        
        t_start = time.time()
        log_eval = self.eval_agent_infer(num_samples,
                                   mean_action=mean_action,
                                   visualize=visualize)
        t_eval = time.time() - t_start

        logger = self.logger
        logger.info(f'Infer time: {t_eval:.2f}')
        logger.info(f'dis: {log_eval.face2face_avg}')
        logger.info(f'cost: {log_eval.total_road_cost}')
        logger.info(f'eval_0.9&0.1: {log_eval.avg_episode_reward}')


        self.save_plan(log_eval)
        if save_video:
            if only_road:
                save_video_ffmpeg(f'{self.cfg.plan_dir}/step_road_%04d.svg',
                                  f'{self.cfg.plan_dir}/plan_road.mp4',
                                  fps=10)
            else:
                save_video_ffmpeg(
                    f'{self.cfg.plan_dir}/step_land_use_%04d.svg',
                    f'{self.cfg.plan_dir}/plan_land_use.mp4',
                    fps=10)
            save_video_ffmpeg(f'{self.cfg.plan_dir}/step_all_%04d.svg',
                              f'{self.cfg.plan_dir}/plan_all.mp4',
                              fps=10)

    def eval_agent_ga(self,
                      gene,
                      num_samples=1,
                      mean_action=True,
                      visualize=True):
        t_start = time.time()
        to_test(*self.sample_modules)
        self.env.eval()
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                logger = self.logger_cls(**self.logger_kwargs)

                while logger.num_steps < num_samples:
                    state = self.env.reset()
                    if visualize:
                        os.makedirs(os.path.join(self.cfg.plan_dir,"eva_ga_"), exist_ok=True)
                        self.env.visualize(save_fig=True,
                                           path=os.path.join(
                                               self.cfg.plan_dir,
                                               'origin.svg'))
                    logger.start_episode(self.env)

                    info_plan = dict()
                    episode_success = False
                    for t in range(1, 10000):
                        state_var = tensorfy([state])
                        action = self.policy_net.select_action(
                            state_var, gene, mean_action).numpy()
                        next_state, reward, done, info = self.env.step(
                            action, self.logger)
                        logger.step(self.env, reward, info)
                        # self.logger.info(f'reward:{reward}  step:{t:02d}')
                        if visualize:
                            os.makedirs(os.path.join(self.cfg.plan_dir,"eva_ga_"), exist_ok=True)
                            self.env.visualize(save_fig=True,
                                               path=os.path.join(
                                                   self.cfg.plan_dir,
                                                   f'step_all_{t:04d}.svg'))
                        if done:
                            episode_success = (reward != self.env.FAILURE_REWARD) and \
                                              (reward != self.env.INTERMEDIATE_REWARD)
                            info_plan = info
                            break
                        state = next_state

                    logger.add_plan(info_plan)
                    logger.end_episode(info_plan)
                    if not episode_success:
                        self.logger.info('Plan fails during eval.')
                    else:
                        os.makedirs(os.path.join(self.cfg.plan_dir,"eva_ga_"), exist_ok=True)
                        self.env.visualize(save_fig=True,
                                           path=os.path.join(
                                               self.cfg.plan_dir,
                                               f'final.svg'))
                logger = self.logger_cls.merge([logger], **self.logger_kwargs)

        
        self.env.train()
        logger.sample_time = time.time() - t_start
        return logger

    def fitness_ga(self,
                   gene,
                   num_samples=1,
                   mean_action=True,
                   visualize=False) -> Tuple[float, Dict]:
        log_eval = self.eval_agent_ga(gene, num_samples, mean_action,
                                      visualize)
        return log_eval.avg_episode_reward, log_eval.plans[0]

    def save_ga(self, best_solution, best_solution_fitness):
        solution = {
            'best_solution': best_solution,
            'best_solution_fitness': best_solution_fitness,
        }
        cfg = self.cfg
        self.logger.info(f'save ga solution to file: {cfg.model_dir}/best.p')
        with open(f'{cfg.model_dir}/best.p', 'wb') as f:
            pickle.dump(solution, f)

    def load_ga(self):
        cfg = self.cfg
        self.logger.info(f'load ga solution from file: {cfg.model_dir}/best.p')
        with open(f'{cfg.model_dir}/best.p', 'rb') as f:
            solution = pickle.load(f)
        return solution['best_solution'], solution['best_solution_fitness']
