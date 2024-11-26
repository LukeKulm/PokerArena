import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ai_models.bc import NN


def training_fn(data_name, model_name):
    data, labels = torch.load(data_name, weights_only = True)
    data_points = len(data)
    print(f"Number of data points: {data_points}")


    model = NN(input_size=len(data[0]))


    model.train_model(train_data=data, train_targets= labels, num_epochs=15000, learning_rate=0.001)


    # Evaluate the model
    model.evaluate(test_data=data, test_targets=labels)

    model.save_checkpoint(file_path=model_name)

if __name__ == "__main__":
    training_fn('data/improved_expert_data.pt', "ai_models/smart_bc_checkpoint.pth")