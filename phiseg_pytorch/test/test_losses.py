import torch


def test_hierarchical_reconstruction_loss():
    from src.losses import HierarchicalReconstructionLoss
    from utils import convert_to_onehot

    loss = HierarchicalReconstructionLoss(
        reconstruction_loss=torch.nn.CrossEntropyLoss(),
        weight_dict={0: 1.0, 1: 2.0, 2: 3.0},
    )
    inputs = {
        l: convert_to_onehot(
            torch.randint(0, 3, size=(32, 256, 256)), num_classes=3
        ).type(torch.FloatTensor)
        for l in range(3)
    }
    target = torch.randint(0, 3, size=(32, 256, 256)).type(torch.LongTensor)
    output = loss(inputs=inputs, target=target)
    assert output.dim() == 0


def test_hierarchical_kl_loss():
    from src.losses import HierarchicalKLLoss
    from src.losses import KL_two_gauss_with_diag_cov
    from torch.nn.functional import softplus

    loss = HierarchicalKLLoss(
        KL_divergence=KL_two_gauss_with_diag_cov, weight_dict={0: 1.0, 1: 2.0, 2: 3.0}
    )

    means = {l: torch.randn(32, 3**l) for l in range(3)}
    sigmas = {l: softplus(torch.randn(32, 3**l)) for l in range(3)}

    output = loss(means, sigmas, means, sigmas)

    assert output.dim() == 0
    assert output < 1e-5  # should be zero, but there is numerical inaccuracy
