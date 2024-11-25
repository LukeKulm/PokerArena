import ai_models.q_learning_reinforcement_learning_model as model
import numpy as np
import torch


class TestDataBuffer:
    def test_does_not_exceed_max_len(self):
        buffer = model.DataBuffer(maxlen=1)
        buffer.add(0, 0, 0, 0)
        buffer.add(1, 1, 1, 1)
        assert len(buffer.buffer) == 1
        assert buffer.buffer[0] == (1, 1, 1, 1)

    def test_weighted_sample(self):
        buffer = model.DataBuffer(maxlen=2)
        buffer.add(np.array([0]), 0, 0, np.array([0]))
        buffer.add(np.array([1]), 1, 1, np.array([1]))
        assert len(buffer.buffer) == 2
        assert len(buffer.weighted_sample(1)[0]) == 1


class TestPokerQNetwork:
    def test_select_action(self):
        network = model.PokerQNetwork(8, 10)
        state = np.random.rand(8)
        epsilon = 0.01

        prediction = network.select_action(state, epsilon)
        assert isinstance(prediction, int)
        assert prediction >= 0
        assert prediction <= 9

    def test_select_action_softmax_activated(self):
        network = model.PokerQNetwork(8, 10, softmax=True)
        state = np.random.rand(8)
        epsilon = 0.01

        prediction = network.select_action(state, epsilon)
        assert isinstance(prediction, int)
        assert prediction >= 0
        assert prediction <= 9

    def test_forward(self):
        network = model.PokerQNetwork(8, 10)
        state = torch.rand(8)
        output = network.forward(state)
        assert output.size() == torch.Size([10])
