import pytest

from grpo import advantages


def test_advantages_normalized_per_group():
    adv = advantages([1.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5], group_size=4)
    assert adv[:4] == pytest.approx([1.0, -1.0, 1.0, -1.0])
    assert adv[4:] == [None] * 4  # même reward partout : rien à apprendre


def test_advantages_sum_to_zero():
    adv = advantages([-1.0, 0.2, 0.7, 1.0], group_size=4)
    assert sum(adv) == pytest.approx(0.0)
    assert adv[0] < adv[1] < adv[2] < adv[3]
