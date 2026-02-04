import kornia


class Noiser:
    def __init__(self):
        self.noise = kornia.augmentation.AugmentationSequential(
            kornia.augmentation.RandomGaussianNoise(mean=0.0, std=0.05, p=1),
            # kornia.augmentation.RandomAffine(
            #     degrees=(-5, 5),
            #     translate=[0.01, 0.01],
            #     scale=[0.99, 1.01],
            #     # shear=[-1, 1],
            #     p=1,
            # ),
            # kornia.augmentation.RandomPerspective(0.6, p=1),
            # kornia.augmentation.RandomSaltAndPepperNoise((0.01, 0.01), p=1),
            same_on_batch=False,
        )

    def __call__(self, x):
        return self.noise(x)
