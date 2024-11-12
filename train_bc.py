import torch


data = torch.load('data/expert_policy.pt', weights_only = True)
data_points = len(data[0])
print(f"Number of data points: {data_points}")
