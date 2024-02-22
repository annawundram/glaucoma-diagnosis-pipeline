import torch


def _check_downsampling(list_of_activations, input_shape):
    for ii, activation in list_of_activations.items():
        assert activation.shape[0] == input_shape[0]
        assert activation.shape[2] == input_shape[2] / 2**ii
        assert activation.shape[3] == input_shape[3] / 2**ii


def test_unet_encoder():
    from src.components.unet import UNetEncoder

    enc = UNetEncoder(total_levels=7, input_channels=1, n0=32)
    inp = torch.randn(4, 1, 256, 256)
    enc_features = enc(inp)

    assert type(enc_features) == dict
    assert len(enc_features) == 7
    _check_downsampling(enc_features, inp.shape)


def test_unet_decoder():
    from src.components.unet import UNetDecoder

    dec = UNetDecoder(total_levels=3, n0=32)
    enc_features = [
        torch.randn(4, 32, 64, 64),
        torch.randn(4, 64, 32, 32),
        torch.randn(4, 128, 16, 16),
    ]
    out = dec(enc_features)
    assert out.shape[0] == 4
    assert out.shape[2] == 64
    assert out.shape[3] == 64


def test_unet():
    from src.components.unet import UNetBase

    unet = UNetBase(total_levels=7, input_channels=1, n0=32)
    inp = torch.randn(4, 1, 256, 256)
    out = unet(inp)

    assert out.shape[0] == inp.shape[0]
    assert [*out.shape[2:4]] == [*inp.shape[2:4]]
