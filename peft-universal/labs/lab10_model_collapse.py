"""Lab 10: model collapse in miniature. Each generation is fit only to samples of the previous generation.
Tails disappear and variance shrinks. Mixing in real data each generation prevents it."""
import numpy as np

rng = np.random.default_rng(0)
real = rng.standard_normal(100_000)
n = 50  # small synthetic sample per generation, like a small synthetic dataset


def simulate(real_fraction: float, generations: int = 200):
    mu, sigma = 0.0, 1.0
    hist = []
    for g in range(generations + 1):
        if g % 50 == 0:
            hist.append((g, round(sigma, 4), round(float(np.mean(np.abs(rng.normal(mu, sigma, 20000)) > 2.5)), 4)))
        k = int(n * real_fraction)
        sample = np.concatenate([rng.normal(mu, sigma, n - k), rng.choice(real, k)])
        mu, sigma = sample.mean(), sample.std()  # "train" the next model on the mixture
    return hist


print("real data: sigma = 1.0, P(|x| > 2.5) = 0.0124")
for frac in (0.0, 0.2):
    print(f"\nreal-data fraction per generation = {frac}")
    for g, s, tail in simulate(frac):
        print(f"  gen {g:>3}: sigma = {s:<7} P(|x|>2.5) = {tail}")
