import kornia


class Noiser:
    def __init__(self):
        self.noise = kornia.augmentation.AugmentationSequential(
            kornia.augmentation.RandomGaussianNoise(mean=0.0, std=0.001, p=1),
            kornia.augmentation.RandomAffine(
                degrees=10,
                translate=[0.0, 0.0],
                scale=[0.9, 1.1],
                shear=[-2, 2],
                p=1,
            ),
            # kornia.augmentation.RandomPerspective(distortion_scale=0.3, p=0.9),
            # kornia.augmentation.RandomPerspective(0.6, p=1),
            # kornia.augmentation.RandomSaltAndPepperNoise((0.01, 0.01), p=1),
            same_on_batch=False,
        )

    def __call__(self, x):
        return self.noise(x)
