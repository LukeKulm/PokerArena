from bc import NN
import torch

if __name__ == "__main__":
    data, labels = torch.load('data/improved_expert_data.pt', weights_only = True)
    data_points = len(data)
    print(f"Number of data points: {data_points}")


    model = NN(input_size=len(data[0]))


    model.train_model(train_data=data, train_targets= labels, num_epochs=3000, learning_rate=0.001)


    # Evaluate the model
    model.evaluate(test_data=data, test_targets=labels)

    model.save_checkpoint()