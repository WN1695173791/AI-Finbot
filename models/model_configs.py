class SDEnet_configs():
    # structure configs
    drift_depth = 8
    diffusion_depth = 8
    in_nodes = 20
    latent_nodes = 64
    diffusion_nodes = 128

    # compilation configs
    lr_1 = 1e-6
    momentum_1 = 0.9
    lr_2 = 0.01
    momentum_2 = 0.9
    weight_decay = 5e-4

    # train configs
    epoch = 200
    batch_size = 32

    # test configs
    test_iters = 100
