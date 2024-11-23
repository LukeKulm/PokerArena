import ai_models.q_learning_reinforcement_learning_model as model


class TestDataBuffer:
    def test_does_not_exceed_max_len(self):
        buffer = model.DataBuffer(maxsize=1)
        buffer.add(0, 0, 0, 0)
        buffer.add(1, 1, 1, 1)
        assert len(buffer.buffer) == 1
        assert buffer.buffer[0] == (1, 1, 1, 1)
