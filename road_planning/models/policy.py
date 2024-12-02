import torch
import torch.nn as nn


class UrbanPlanningPolicy(nn.Module):
    """
    Policy network for urban planning.
    """
    def __init__(self, cfg, agent, shared_net):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.shared_net = shared_net
        self.policy_road_head = self.create_policy_head(
            self.shared_net.output_policy_road_size, cfg['policy_road_head_hidden_size'], 'road')

    def create_policy_head(self, input_size, hidden_size, name):
        """Create the policy land_use head."""
        policy_head = nn.Sequential()
        for i in range(len(hidden_size)):
            if i == 0:
                policy_head.add_module(
                    '{}_linear_{}'.format(name, i),
                    nn.Linear(input_size, hidden_size[i])
                )
            else:
                policy_head.add_module(
                    '{}_linear_{}'.format(name, i),
                    nn.Linear(hidden_size[i - 1], hidden_size[i], bias=False)
                )
            if i < len(hidden_size) - 1:
                policy_head.add_module(
                    '{}_tanh_{}'.format(name, i),
                    nn.Tanh()
                )
            elif hidden_size[i] == 1:
                policy_head.add_module(
                    '{}_flatten_{}'.format(name, i),
                    nn.Flatten()
                )
        return policy_head

    # def create_policy_head(self, input_size, hidden_size, name):
    #     """Create the policy land_use head with BatchNorm and ELU."""
    #     policy_head = nn.Sequential()
    #     print("input_size", input_size)
    #     print("hidden_size", hidden_size)
        
    #     for i in range(len(hidden_size)):
    #         if i == 0:
    #             # First layer: input_size -> hidden_size[0]
    #             policy_head.add_module(
    #                 '{}_linear_{}'.format(name, i),
    #                 nn.Linear(input_size, hidden_size[i])
    #             )
    #         else:
    #             # Other layers: hidden_size[i-1] -> hidden_size[i]
    #             policy_head.add_module(
    #                 '{}_linear_{}'.format(name, i),
    #                 nn.Linear(hidden_size[i - 1], hidden_size[i], bias=False)
    #             )

    #         # Add BatchNorm after Linear
    #         policy_head.add_module(
    #             '{}_batchnorm_{}'.format(name, i),
    #             nn.BatchNorm1d(hidden_size[i])
    #         )
            
        
    #         if i < len(hidden_size) - 1:
    #             # Add ELU instead of Tanh
    #             policy_head.add_module(
    #                 '{}_elu_{}'.format(name, i),
    #                 nn.ELU()
    #             )
    #         elif hidden_size[i] == 1:
    #             # Flatten if the last layer has output size of 1
    #             policy_head.add_module(
    #                 '{}_flatten_{}'.format(name, i),
    #                 nn.Flatten()
    #             )
        
    #     return policy_head


    def forward(self, x):
        
        state_policy_road, _, edge_mask, stage = self.shared_net(x)
        # print ("forward_edge_mask",edge_mask)

        if stage[:, 2].sum() == 0:
            road_logits = self.policy_road_head(state_policy_road)
            # print ("forward_road_logits",road_logits)
            road_paddings = torch.ones_like(edge_mask, dtype=self.agent.dtype)*(-2.**32 + 1)
            # print ("forward_road_paddings",road_paddings)
            masked_road_logits = torch.where(edge_mask.bool(), road_logits, road_paddings)
            # print ("forward_masked_road_logits",masked_road_logits)
            road_dist = torch.distributions.Categorical(logits=masked_road_logits)
            # print ("forward_road_dist",road_dist)
        else:
            road_dist = None

        return road_dist, stage
    

    # def forward(self, x):
    #     with open('/home/chenzebin/road-planning-for-slums/road_planning/models/output500.txt', 'a') as f:
    #         # f.write("state_policy_road输入通过shared_net前:")
    #         # f.write("\n")
    #         for batch in x:
    #             for list in batch:
                 
    #                 # x_list_converted = list.cpu().numpy().tolist()  
    #                 # f.write("{}\n : ".format(list.shape))
    #                 # f.write( str(x_list_converted))
    #                 # f.write("\n")
    #                 pass
    #         state_policy_road, _, edge_mask, stage = self.shared_net(x)
    #         f.write("state_policy_road输入通过shared_net后:")
    #         f.write("{}\n : ".format(state_policy_road.shape))
    #         f.write(str(state_policy_road.tolist()) + "\n")
    #         f.write("-----------------\n")
    #         f.write("\n")
    #         if stage[:, 2].sum() == 0:
    #             #print ("forward")

    #             #state_policy_road_list = state_policy_road.cpu().numpy().tolist()
    #             # f.write("state_policy_road输入通过第一个线性层前: {}\n".format(state_policy_road_list))
    #             # f.write("\n")


    #             #print ("state_policy_road输入通过第一个线性层前",state_policy_road.shape,state_policy_road)
    #             x = self.policy_road_head[0](state_policy_road)    # 输入通过第一个线性层
    #             # f.write("state_policy_road输入通过第一个线性层后: {}\n".format(x))
    #             # f.write("\n")

    #             #print ("state_policy_road输入通过第一个线性层后",x)
        

    #             x = self.policy_road_head[1](x)       # 经过Tanh激活函数
    #             # f.write("经过Tanh激活函数后: {}\n".format(x))
    #             # f.write("\n")
    #             f.write("====================================================================")
    #             f.write("\n")
    #             f.write("\n")

    #             #print ("经过Tanh激活函数后",x.shape,x)

                
    #             x = self.policy_road_head[2](x)    # 输入通过第二个线性层
    #             x = self.policy_road_head[3](x)    # 展平为一维向量

    #             road_logits = x
    #             road_logits = self.policy_road_head(state_policy_road)
    #             road_paddings = torch.ones_like(edge_mask, dtype=self.agent.dtype)*(-2.**32 + 1)
    #             masked_road_logits = torch.where(edge_mask.bool(), road_logits, road_paddings)
    #             road_dist = torch.distributions.Categorical(logits=masked_road_logits)
    #         else:
    #             road_dist = None

    #     return road_dist, stage
    
    # def forward(self, x):
    
    #     state_policy_road, _, edge_mask, stage = self.shared_net(x)

    #     if stage[:, 2].sum() == 0:

    #         x = self.policy_road_head[0](state_policy_road)                # [1, 523, 16] -> [1, 523, 32]
    #         x = x.transpose(1, 2)              # 转置为 [1, 32, 523] 以适应BatchNorm1d
    #         x = self.policy_road_head[1](x)                  # 归一化
    #         x = x.transpose(1, 2)              # 再次转置回 [1, 523, 32]
    #         x = self.policy_road_head[2](x)            # 应用tanh激活函数
            
    #         x = self.policy_road_head[3](x)                # [1, 523, 32] -> [1, 523, 1]
    #         x = x.transpose(1, 2)              # 转置为 [1, 1, 523]
    #         x = self.policy_road_head[4](x)                  # 归一化
            
    #         x = x.view(x.size(0), -1)          # 展平为 [1, 523]

    #         road_logits = x



    #         road_paddings = torch.ones_like(edge_mask, dtype=self.agent.dtype)*(-2.**32 + 1)
    #         masked_road_logits = torch.where(edge_mask.bool(), road_logits, road_paddings)
    #         road_dist = torch.distributions.Categorical(logits=masked_road_logits)
    #     else:
    #         road_dist = None

    #     return road_dist, stage
    
        

    def select_action(self, x, mean_action=False):
       
        road_dist, stage = self.forward(x)
        # print ("road_dist",road_dist.probs)
        # print ("mean_action",mean_action)
        batch_size = stage.shape[0]
        action = torch.zeros(batch_size, 1, dtype=self.agent.dtype, device=stage.device)
        #print ("select_action_action:",action)
   
        if mean_action:
            road_action = road_dist.probs.argmax(dim=1).to(self.agent.dtype)
            
        else:
            road_action = road_dist.sample().to(self.agent.dtype)
        action = road_action
        
        return action

    def get_log_prob_entropy(self, x, action):
        road_dist, stage = self.forward(x)
        batch_size = stage.shape[0]
        log_prob = torch.zeros(batch_size, dtype=self.agent.dtype, device=stage.device)
        entropy = torch.zeros(batch_size, dtype=self.agent.dtype, device=stage.device)

        road_action = action

        road_log_prob = road_dist.log_prob(road_action)
        log_prob = road_log_prob
        entropy = road_dist.entropy()

        return log_prob.unsqueeze(1), entropy.unsqueeze(1)
