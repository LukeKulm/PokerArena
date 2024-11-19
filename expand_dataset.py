import torch

def main():
    state_dict = torch.load("data/expert_policy.pt")
    data, labels = state_dict
    print(len(data))
    expanded_data = torch.empty((0,))
    expanded_labels = torch.empty((0,))

    for state, pred in zip(data, labels):
        perms = get_perms(state)
        for perm in perms:
            expanded_data = torch.cat((expanded_data, perm))
            expanded_labels = torch.cat((expanded_labels, pred))
    
    torch.save((expanded_data, expanded_labels), 'data/expanded_expert_data.pt')

def swap(arr, i, j):
    temp = arr[i]
    arr[i] = j
    arr[j] = temp
    return arr

def get_perms(state):
    ans = [state]
    backwards_hand = swap(state, 2, 4)
    backwards_hand = swap(backwards_hand, 3, 5)
    ans.append(backwards_hand)


    return ans
    


if __name__ == "__main__":
    main()