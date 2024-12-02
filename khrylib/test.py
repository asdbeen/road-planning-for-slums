
state_policy_road, _, edge_mask, stage = self.shared_net(x)


if stage[:, 2].sum() == 0:

    road_logits = self.policy_road_head(state_policy_road)
    road_paddings = torch.ones_like(edge_mask, dtype=self.agent.dtype)*(-2.**32 + 1)
    masked_road_logits = torch.where(edge_mask.bool(), road_logits, road_paddings)
    road_dist = torch.distributions.Categorical(logits=masked_road_logits)
else:
    road_dist = None


road_dist, stage = self.forward(x)
# print ("road_dist",road_dist.probs)
# print ("mean_action",mean_action)
batch_size = stage.shape[0]
action = torch.zeros(batch_size, 1, dtype=self.agent.dtype, device=stage.device)
if mean_action:
    road_action = road_dist.probs.argmax(dim=1).to(self.agent.dtype)
    
else:
    road_action = road_dist.sample().to(self.agent.dtype)
action = road_action 