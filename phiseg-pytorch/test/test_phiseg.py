import torch

from src.network_blocks import gauss_sampler


def _check_z_shapes_for_phiseg_7_5(mus, sigmas, samples):
    assert type(mus) == type(sigmas) == type(samples) == dict
    assert len(mus) == len(sigmas) == len(samples) == 5
    assert mus[0].shape == sigmas[0].shape == samples[0].shape
    assert [*mus[4].shape] == [16, 2, 4, 4]
    assert [*mus[3].shape] == [16, 2, 8, 8]
    assert [*mus[2].shape] == [16, 2, 16, 16]
    assert [*mus[1].shape] == [16, 2, 32, 32]
    assert [*mus[0].shape] == [16, 2, 64, 64]


def test_phiseg_7_5_encoder():
    from src.components.phiseg import PHISegEncoder

    enc = PHISegEncoder(total_levels=7, latent_levels=5, zdim=2, input_channels=4)
    inp = torch.randn(16, 4, 256, 256)
    mus, sigmas, samples = enc(inp, gauss_sampler)

    _check_z_shapes_for_phiseg_7_5(mus, sigmas, samples)

    for l in range(5):
        assert torch.all(sigmas[l] >= 0)


def test_phiseg_7_1_encoder():
    from src.components.phiseg import PHISegEncoder

    enc = PHISegEncoder(total_levels=7, latent_levels=1, zdim=2, input_channels=4)
    inp = torch.randn(16, 4, 256, 256)
    mus, sigmas, samples = enc(inp, gauss_sampler)

    assert type(mus) == type(sigmas) == type(samples) == dict
    assert len(mus) == len(sigmas) == len(samples) == 1

    assert mus[0].shape == sigmas[0].shape == samples[0].shape
    assert [*mus[0].shape] == [16, 2, 4, 4]

    assert torch.all(sigmas[0] >= 0)


def _likelihood_check(latent_levels):
    from src.components.phiseg import PHISegLikelihood

    dec = PHISegLikelihood(
        total_levels=7, latent_levels=latent_levels, zdim=2, num_classes=3
    )

    samples = {}
    for l in range(latent_levels):
        samples[l] = torch.randn(
            16, 2, 4 * 2 ** (latent_levels - l - 1), 4 * 2 ** (latent_levels - l - 1)
        )

    out = dec(samples)
    assert type(out) == dict
    assert len(out) == latent_levels

    for l in range(latent_levels):
        assert [*out[l].shape] == [16, 3, 256, 256], f"Shape mismatch a level {l}."


def test_phiseg_7_5_likelihood():
    _likelihood_check(latent_levels=5)


def test_phiseg_7_1_likelihood():
    _likelihood_check(latent_levels=1)


def test_phiseg_prior():
    from src.components.phiseg import PHISegPrior

    prior = PHISegPrior(sampler=gauss_sampler, total_levels=7, latent_levels=5, zdim=2)
    x = torch.randn(16, 1, 256, 256)
    mus, sigmas, samples = prior(x)

    _check_z_shapes_for_phiseg_7_5(mus, sigmas, samples)


def test_phiseg_prior_with_override():
    from src.components.phiseg import PHISegPrior
    from src.components.phiseg import PHISegPosterior

    posterior = PHISegPosterior(
        sampler=gauss_sampler, total_levels=7, latent_levels=5, zdim=2, num_classes=3
    )
    prior = PHISegPrior(sampler=gauss_sampler, total_levels=7, latent_levels=5, zdim=2)

    x = torch.randn(16, 1, 256, 256)
    y = torch.randint(0, 3, size=(16, 256, 256))

    _, _, posterior_samples = posterior(x, y)
    mus, sigmas, samples = prior(x, posterior_samples=posterior_samples)

    _check_z_shapes_for_phiseg_7_5(mus, sigmas, samples)


def test_phiseg_posterior():
    from src.components.phiseg import PHISegPosterior

    posterior = PHISegPosterior(
        sampler=gauss_sampler, total_levels=7, latent_levels=5, zdim=2, num_classes=3
    )
    x = torch.randn(16, 1, 256, 256)
    y = torch.randint(0, 3, size=(16, 256, 256))
    mus, sigmas, samples = posterior(x, y)
    _check_z_shapes_for_phiseg_7_5(mus, sigmas, samples)


def test_phiseg_forward():
    from src.models import PHISeg

    x = torch.randn(16, 1, 256, 256)
    model = PHISeg(total_levels=7, latent_levels=5, zdim=2, num_classes=3)

    y_hat = model(x)
    assert [*y_hat.shape] == [16, 3, 256, 256]


def test_phiseg_training_step():
    from src.models import PHISeg

    x = torch.randn(16, 1, 256, 256)
    y = torch.randint(0, 3, size=(16, 256, 256))
    all_ys = torch.randint(0, 3, size=(16, 4, 256, 256))
    model = PHISeg(total_levels=7, latent_levels=5, zdim=2, num_classes=3)
    model.training_step((x, y, all_ys), 0)


def test_phiseg_predict_single_image():
    from src.models import PHISeg

    x = torch.randn(1, 1, 256, 256)
    model = PHISeg(total_levels=7, latent_levels=5, zdim=2, num_classes=3)
    y_hat = model.predict(x, N=4)
    assert [*y_hat.shape] == [1, 3, 256, 256]


def test_phiseg_predict_single_batch():
    from src.models import PHISeg

    x = torch.randn(4, 1, 256, 256)
    model = PHISeg(total_levels=7, latent_levels=5, zdim=2, num_classes=3)
    y_hat = model.predict(x, N=5)
    assert [*y_hat.shape] == [4, 3, 256, 256]


def test_phiseg_unusual_sizes():
    from src.models import PHISeg

    x = torch.randn(4, 1, 64, 64)
    model = PHISeg(total_levels=3, latent_levels=2, zdim=2, num_classes=3)
    y_hat = model.predict(x, N=5)
    model = PHISeg(total_levels=3, latent_levels=3, zdim=2, num_classes=3)
    y_hat = model.predict(x, N=5)
    assert [*y_hat.shape] == [4, 3, 64, 64]
