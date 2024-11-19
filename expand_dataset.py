import torch

def main():
    state_dict = torch.load("data/expert_policy.pt")
    expanded_data = torch.empty((0,))
    expanded_labels = torch.empty((0,))

    for i in range(len(state_dict[0])):
        state = state_dict[0][i]
        pred = state_dict[1][i]
        perms = get_perms(state)
        for perm in perms:
            expanded_data = torch.cat((expanded_data, perm))
            expanded_labels = torch.cat((expanded_labels, pred))
    torch.save((expanded_data, expanded_labels), 'data/expanded_expert_data.pt')

def get_perms(state):
    return [state]

if __name__ == "__main__":
    main()