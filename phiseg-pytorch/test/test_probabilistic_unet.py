import torch
from src.network_blocks import gauss_sampler


def test_probabilistic_unet_likelihood():
    from src.components.probabilistic_unet import ProbUNetLikelihood

    latent_dim = 10

    likelihood_net = ProbUNetLikelihood(
        total_levels=7, zdim=latent_dim, num_classes=4, n0=32
    )

    x = torch.randn(4, 1, 256, 256)
    z = {0: torch.randn(4, latent_dim)}

    out = likelihood_net(z, x)

    assert type(out) == dict
    assert len(out) == 1
    assert [*out[0].shape] == [4, 4, 256, 256]


def test_probabilistic_unet_prior():
    from src.components.probabilistic_unet import ProbUNetPrior

    latent_dim = 10
    prior_net = ProbUNetPrior(
        sampler=gauss_sampler, total_levels=7, zdim=latent_dim, n0=32
    )

    inp = torch.randn(4, 1, 256, 256)

    mu, sigma, samples = prior_net(inp)
    assert (type(mu), type(sigma), type(samples)) == (dict, dict, dict)
    assert (len(mu), len(sigma), len(samples)) == (1, 1, 1)
    assert [*mu[0].shape] == [4, latent_dim, 1, 1]
    assert [*sigma[0].shape] == [4, latent_dim, 1, 1]
    assert [*samples[0].shape] == [4, latent_dim, 1, 1]


def test_probabilistic_unet_posterior():
    from src.components.probabilistic_unet import ProbUNetPosterior

    latent_dim = 10
    num_classes = 3

    posterior_net = ProbUNetPosterior(
        sampler=gauss_sampler,
        total_levels=7,
        zdim=latent_dim,
        n0=32,
        num_classes=num_classes,
    )

    x = torch.randn(4, 1, 256, 256)
    y = torch.randint(0, num_classes, size=(4, 256, 256))

    mu, sigma, samples = posterior_net(x, y)
    assert (type(mu), type(sigma), type(samples)) == (dict, dict, dict)
    assert (len(mu), len(sigma), len(samples)) == (1, 1, 1)
    assert [*mu[0].shape] == [4, latent_dim, 1, 1]
    assert [*sigma[0].shape] == [4, latent_dim, 1, 1]
    assert [*samples[0].shape] == [4, latent_dim, 1, 1]


def test_probabilistic_unet_forward():
    from src.models import ProbUNet

    x = torch.randn(16, 1, 256, 256)
    model = ProbUNet(total_levels=7, zdim=2, num_classes=3)

    y_hat = model(x)
    assert [*y_hat.shape] == [16, 3, 256, 256]


def test_probabilistic_unet_training_step():
    from src.models import ProbUNet

    x = torch.randn(16, 1, 256, 256)
    y = torch.randint(0, 3, size=(16, 256, 256))
    all_ys = torch.randint(0, 3, size=(16, 4, 256, 256))
    model = ProbUNet(total_levels=7, zdim=2, num_classes=3)
    model.training_step((x, y, all_ys), 0)


def test_probabilistic_unet_predict_single_image():
    from src.models import ProbUNet

    x = torch.randn(1, 1, 256, 256)
    model = ProbUNet(total_levels=7, zdim=2, num_classes=3)
    y_hat = model.predict(x, N=4)
    assert [*y_hat.shape] == [1, 3, 256, 256]


def test_probabilistic_unet_predict_single_batch():
    from src.models import ProbUNet

    x = torch.randn(4, 1, 256, 256)
    model = ProbUNet(total_levels=7, zdim=2, num_classes=3)
    y_hat = model.predict(x, N=5)
    assert [*y_hat.shape] == [4, 3, 256, 256]
