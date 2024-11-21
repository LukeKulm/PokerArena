import torch



def main():
    state_dict = torch.load("data/expert_policy.pt")
    data, labels = state_dict
    expanded_data = torch.empty((0,) + data[0].shape)
    expanded_labels = torch.empty((0,) + labels[0].shape)

    for state, pred in zip(data, labels):
        perms = get_perms(state)
        for perm in perms:
            expanded_data = torch.cat((expanded_data, perm.unsqueeze(0)), dim=0)
            expanded_labels = torch.cat((expanded_labels, pred.unsqueeze(0)), dim=0)
    
    torch.save((expanded_data, expanded_labels), 'data/expanded_expert_data.pt')

def swap(arr, i, j, k, m):
    ans = arr.clone()
    ans[i] = arr[j]
    ans[j] = arr[i]
    ans[k] = arr[m]
    ans[m] = arr [k]
    return ans

def get_perms(state):
    ans = [state]
    backwards_hand = swap(state, 10, 12, 11, 13)
    
    ans.append(backwards_hand)

    if state[7] > 0 : # flop or later
        old = ans
        ans = []
        for first in old:
            ans.append(first)
            ans.append(swap(first, 14, 12, 15, 13))
            second = (swap(first, 10, 12, 11, 13))
            ans.append(second)
            ans.append(swap(second, 14, 12, 15, 13))
            third = (swap(first, 10, 14, 11, 15))
            ans.append(third)
            ans.append(swap(third, 14, 12, 15, 13))
    if state[7] > 1: # turn or later
        old = ans
        ans = []
        for first in old:
            ans.append(first)
            ans.append(swap(first, 16, 10, 17, 11))
            ans.append(swap(first, 16, 12, 17, 13))
            ans.append(swap(first, 16, 14, 17, 15))
    if state[7] > 2: # river
        old = ans
        ans = []
        for first in old:
            ans.append(first)
            ans.append(swap(first, 18, 10, 19, 11))
            ans.append(swap(first, 18, 12, 19, 13))
            ans.append(swap(first, 18, 14, 19, 15))
            ans.append(swap(first, 18, 16, 19, 17))
           

            


    return ans
    


if __name__ == "__main__":
    main()