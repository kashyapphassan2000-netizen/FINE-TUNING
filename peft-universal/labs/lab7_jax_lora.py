"""Lab 7: the same LoRA math in JAX (framework-agnostic proof). Only A and B are differentiated."""
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(0)
d, r, alpha = 128, 4, 8.0
k1, k2, k3, k4 = jax.random.split(key, 4)
W0 = jax.random.normal(k1, (d, d)) / jnp.sqrt(d)                 # frozen: never passed to grad
W_target = W0 + 0.05 * jax.random.normal(k2, (d, r)) @ jax.random.normal(k3, (r, d))
# A must not be tiny: with B = 0, the gradient reaching B is proportional to A.
params = {"A": jax.random.normal(k4, (r, d)) / jnp.sqrt(d), "B": jnp.zeros((d, r))}


def forward(p, x):
    return x @ W0.T + (alpha / r) * (x @ p["A"].T @ p["B"].T)


def loss_fn(p, x):
    return jnp.mean((forward(p, x) - x @ W_target.T) ** 2)


@jax.jit
def step(p, x, lr=0.1):
    loss, g = jax.value_and_grad(loss_fn)(p, x)
    return jax.tree_util.tree_map(lambda a, b: a - lr * b, p, g), loss


for i in range(2001):
    key, sub = jax.random.split(key)
    params, loss = step(params, jax.random.normal(sub, (64, d)))
    if i % 500 == 0:
        print(f"step {i:>4}  loss {float(loss):.3e}")
merged = W0 + (alpha / r) * params["B"] @ params["A"]
x = jax.random.normal(key, (2, d))
print("merge exact:", bool(jnp.allclose(forward(params, x), x @ merged.T, atol=1e-4)))
print("frozen W0 shape", W0.shape, "| trainable:", {k: v.shape for k, v in params.items()})
