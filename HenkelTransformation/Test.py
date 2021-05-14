import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scipybessel

from pyhank import HankelTransform
def sinc(x):
    return np.sin(x) / x
def hankel_transform_of_sinc(v):
    ht = np.zeros_like(v)
    ht[v < gamma] = (v[v < gamma] ** p * np.cos(p * np.pi / 2)
                     / (2 * np.pi * gamma * np.sqrt(gamma ** 2 - v[v < gamma] ** 2)
                        * (gamma + np.sqrt(gamma ** 2 - v[v < gamma] ** 2)) ** p))
    ht[v >= gamma] = (np.sin(p * np.arcsin(gamma / v[v >= gamma]))
                      / (2 * np.pi * gamma * np.sqrt(v[v >= gamma] ** 2 - gamma ** 2)))
    return ht

for p in [1, 4]:
    transformer = HankelTransform(p, max_radius=3, n_points=256)
    gamma = 5
    func = sinc(2 * np.pi * gamma * transformer.r)
    expected_ht = hankel_transform_of_sinc(transformer.v)
    ht = transformer.qdht(func)
    dynamical_error = 20 * np.log10(np.abs(expected_ht - ht) / np.max(ht))
    not_near_gamma = np.logical_or(transformer.v > gamma * 1.25,
                                   transformer.v < gamma * 0.75)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(transformer.v, expected_ht, label='Analytical')
    plt.plot(transformer.v, ht, marker='+', linestyle='None', label='QDHT')
    plt.title(f'Hankel Transform, p={p}')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(transformer.v, dynamical_error)
    plt.title('Dynamical error')
    plt.tight_layout()


#%%
p = 4
a = 1
transformer = HankelTransform(order=p, max_radius=2, n_points=1024)
top_hat = np.zeros_like(transformer.r)
top_hat[transformer.r <= a] = 1
func = transformer.r ** p * top_hat
expected_ht = a ** (p + 1) * scipybessel.jv(p + 1, 2 * np.pi * a * transformer.v) / transformer.v
ht = transformer.qdht(func)

retrieved_func = transformer.iqdht(ht)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(transformer.v, expected_ht, label='Analytical')
plt.plot(transformer.v, ht, marker='x', linestyle='None', label='QDHT')
plt.title(f'Hankel transform $f_2(v)$, order {p}')
plt.xlabel('Frequency /$v$')
plt.xlim([0, 10])
plt.legend()

plt.subplot(2, 1, 2)
plt.title('Round-trip QDHT vs analytical function')
plt.plot(transformer.r, func, label='Analytical')
plt.plot(transformer.r, retrieved_func, marker='x', linestyle='--', label='QDHT+iQDHT')
plt.xlabel('Radius /$r$')
plt.tight_layout()
plt.show()