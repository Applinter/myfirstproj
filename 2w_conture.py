import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

g = 9.81 
l = 0.5
phi_0 = np.pi / 6

y0 = [0, 0, phi_0, 0]

def system(y, t, g, l):
    x, x_prime, phi, phi_prime = y

    phi_double_prime = (112 / 53) * (
        1.25 * g - (g * np.pi) / 12 + (np.sqrt(3) / 112) * (phi_prime**2)
    )

    x_double_prime = (
        (4 * np.sqrt(3) / 53)
        * (1.25 * g - (g * np.pi) / 12 + (np.sqrt(3) / 112) * (phi_prime**2))
        - (1 / 28) * (phi_prime**2)
    )

    return [x_prime, x_double_prime, phi_prime, phi_double_prime]

t = np.linspace(0, 10, 1000)

solution = odeint(system, y0, t, args=(g, l))

x, x_prime, phi, phi_prime = solution.T

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(t, x)
plt.title("Pos x(t)")
plt.xlabel("Time")
plt.ylabel("x")

plt.subplot(2, 2, 2)
plt.plot(t, x_prime)
plt.title("x'(t)")
plt.xlabel("Time")
plt.ylabel("x'")

plt.subplot(2, 2, 3)
plt.plot(t, phi)
plt.title("Phi")
plt.xlabel("Time")
plt.ylabel("phi")

plt.subplot(2, 2, 4)
plt.plot(t, phi_prime)
plt.title("Phi'")
plt.xlabel("Time")
plt.ylabel("phi'")

plt.tight_layout()
plt.show()
