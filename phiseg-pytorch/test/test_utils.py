import torch
from utils import convert_to_onehot


def test_onehot_conversion_2d():
    label_batch = torch.randint(0, 3, size=(32, 64, 64))
    label_onehot = convert_to_onehot(label_batch, num_classes=3)
    label_argmax = torch.argmax(label_onehot, dim=1, keepdim=False)
    assert torch.all(label_batch == label_argmax)


def test_onehot_conversion_3d():
    label_batch = torch.randint(0, 3, size=(32, 64, 64, 64))
    label_onehot = convert_to_onehot(label_batch, num_classes=3)
    label_argmax = torch.argmax(label_onehot, dim=1, keepdim=False)
    assert torch.all(label_batch == label_argmax)


def test_onehot_conversion_2d_single_image():
    label_image = torch.randint(0, 3, size=(64, 64))
    label_onehot = convert_to_onehot(label_image, num_classes=3, channel_dim=0)
    label_argmax = torch.argmax(label_onehot, dim=0, keepdim=False)
    assert torch.all(label_image == label_argmax)


def test_onehot_conversion_2d_sample_batch():
    label_batch = torch.randint(0, 3, size=(32, 5, 64, 64))
    label_onehot = convert_to_onehot(label_batch, num_classes=3, channel_dim=2)
    label_argmax = torch.argmax(label_onehot, dim=2, keepdim=False)
    assert torch.all(label_batch == label_argmax)


def test_tensor_is_onehot_single_image():
    from utils import find_onehot_dimension

    label_image = torch.randint(0, 3, size=(64, 64))
    assert find_onehot_dimension(label_image) is None

    label_image_oh = convert_to_onehot(label_image, num_classes=3, channel_dim=0)
    assert find_onehot_dimension(label_image_oh) == 0


def test_tensor_is_onehot_batch():
    from utils import find_onehot_dimension

    label_image = torch.randint(0, 3, size=(32, 64, 64))
    assert find_onehot_dimension(label_image) is None

    label_image_oh = convert_to_onehot(label_image, num_classes=3, channel_dim=1)
    assert find_onehot_dimension(label_image_oh) == 1


def test_harden_softmax_outputs():
    from utils import harden_softmax_outputs
    from torch.nn.functional import softmax

    label_batch = torch.randint(0, 3, size=(32, 64, 64))
    label_onehot = convert_to_onehot(label_batch, num_classes=3)
    label_softmax = label_onehot.float() + 0.01 * torch.rand(label_onehot.shape)
    label_softmax = softmax(label_softmax)
    label_hard = harden_softmax_outputs(label_softmax, dim=1)

    assert torch.all(label_hard == label_onehot)
