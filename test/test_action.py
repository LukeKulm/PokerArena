import action


def test_action_info_convert_to_tuple():
    action_info = action.ActionInformation(action.Action.CALL.value, 1, 0)
    assert action_info.convert_to_tuple() == (1, 1, 0)
