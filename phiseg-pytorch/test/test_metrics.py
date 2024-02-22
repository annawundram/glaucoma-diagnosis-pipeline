from src.metrics import per_label_dice, generalised_energy_distance
import torch
from utils import convert_to_onehot


def _test_all_dice_combinations(
    target,
    input,
    background,
    num_classes,
    input_is_batch,
):
    dice_target = per_label_dice(
        input=target,
        target=target,
        input_is_batch=input_is_batch,
    )
    assert dice_target.dim() == 1
    assert dice_target.shape[0] == num_classes
    assert torch.all(dice_target == 1)

    dice_normal = per_label_dice(
        input=input,
        target=target,
        input_is_batch=input_is_batch,
    )
    assert torch.all((dice_normal <= 1) & (dice_normal >= 0))

    dice_background = per_label_dice(
        input=background,
        target=target,
        input_is_batch=input_is_batch,
    )
    assert torch.all(dice_background[1:] == 0)

    dice_background = per_label_dice(
        input=background,
        target=background,
        input_is_batch=input_is_batch,
    )
    assert torch.all(dice_background[1:] == 1.0)


def test_per_label_dice_for_batch():
    num_classes = 3

    target = torch.randint(0, num_classes, size=(32, 256, 256))
    input = torch.randint(0, num_classes, size=(32, 256, 256))
    target_oh = convert_to_onehot(target, num_classes=num_classes)
    input_oh = convert_to_onehot(input, num_classes=num_classes)

    background = torch.zeros_like(target)
    background_oh = convert_to_onehot(background, num_classes=num_classes)

    _test_all_dice_combinations(
        target_oh,
        input_oh,
        background_oh,
        num_classes,
        input_is_batch=True,
    )


def test_per_label_dice_single_image():
    num_classes = 3

    target = torch.randint(0, num_classes, size=(256, 256))
    input = torch.randint(0, num_classes, size=(256, 256))
    target_oh = convert_to_onehot(target, num_classes=num_classes, channel_dim=0)
    input_oh = convert_to_onehot(input, num_classes=num_classes, channel_dim=0)

    background = torch.zeros_like(target)
    background_oh = convert_to_onehot(
        background, num_classes=num_classes, channel_dim=0
    )

    _test_all_dice_combinations(
        target_oh,
        input_oh,
        background_oh,
        num_classes,
        input_is_batch=False,
    )


def test_ged_single_image():
    num_classes = 3

    target = torch.randint(0, num_classes, size=(20, 64, 64))
    input = torch.randint(0, num_classes, size=(20, 64, 64))
    target_oh = convert_to_onehot(target, num_classes=num_classes)
    input_oh = convert_to_onehot(input, num_classes=num_classes)

    ged0 = generalised_energy_distance(input_oh, input_oh)
    assert ged0.dim() == 1
    assert ged0.shape[0] == num_classes
    assert torch.all(ged0 == 0.0)

    ged1 = generalised_energy_distance(input_oh, target_oh)
    assert torch.all(ged1 > 0.0) and torch.all(ged1 < 2.0)

    ged2 = generalised_energy_distance(target_oh, input_oh)
    assert torch.all((ged1 - ged2) ** 2 <= 1e-5), "Metric must be symmetric"


def test_ged_batch():
    num_classes = 3
    batch_size = 8
    N = 5

    target = torch.randint(0, num_classes, size=(batch_size, N, 64, 64))
    input = torch.randint(0, num_classes, size=(batch_size, N, 64, 64))
    target_oh = convert_to_onehot(target, num_classes=num_classes, channel_dim=2)
    input_oh = convert_to_onehot(input, num_classes=num_classes, channel_dim=2)

    ged0 = generalised_energy_distance(input_oh, input_oh)
    assert ged0.dim() == 2
    assert ged0.shape[0] == batch_size
    assert ged0.shape[1] == num_classes
    assert torch.all(ged0 == 0.0)

    ged1 = generalised_energy_distance(input_oh, target_oh)
    assert torch.all(ged1 > 0.0) and torch.all(ged1 < 2.0)
