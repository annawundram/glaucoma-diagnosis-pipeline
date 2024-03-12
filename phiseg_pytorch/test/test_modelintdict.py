from src.utils import ModuleIntDict
import torch.nn as nn


def test_setitem_getitem():
    d = ModuleIntDict()
    op = nn.Conv2d(2, 2, 3)
    d[2] = op
    op_ = d[2]
    assert len(d) == 1
    assert op is op_


def test_init_from_dict():
    op = nn.Conv2d(2, 2, 3)
    d = ModuleIntDict({2: op})
    assert len(d) == 1


def test_get_keys():
    d = ModuleIntDict()
    d[1] = nn.Conv2d(2, 2, 3)
    d[3] = nn.Conv2d(4, 4, 3)
    assert list(d.keys()) == [1, 3]


def test_get_items():
    d = ModuleIntDict()
    op = nn.Conv2d(2, 2, 3)
    d[1] = op
    d[3] = nn.Conv2d(4, 4, 3)
    item_list = list(d.items())
    assert len(item_list) == 2
    assert item_list[0][1] == op
