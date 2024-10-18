import pickle
from road_planning.models.model import Model


# 文件路径
file_path = '/home/chenzebin/road-planning-for-slums/train_data/punggol_1_withShortcut_withConfigAll_StraightSkeletonSinglePOI/rl-ngnn/punggol_1_withShortcut_withConfigAll_StraightSkeletonSinglePOI/0/models/best_reward-1.34_iteration_0001.p'

# 加载文件
with open(file_path, 'rb') as file:
    data = pickle.load(file)

model_cp = pickle.load(open(file_path, "rb"))


