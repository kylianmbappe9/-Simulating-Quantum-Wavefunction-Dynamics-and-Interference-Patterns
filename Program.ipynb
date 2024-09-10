import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

L = 10          # Length of the domain
N = 256         # Number of spatial points
dx = L / (N - 1)  # Spatial step size
x = np.linspace(0, L, N)
dt = 0.01       # Time step size
T = 5           # Total time
steps = int(T / dt)

V = 10 * ((x - L / 4)**2 - (L / 8)**2) * ((x - 3 * L / 4)**2 - (L / 8)**2)
V[V < 0] = 0  # Ensure potential is non-negative

sigma = 0.5
x0 = L / 2
psi0 = np.exp(-((x - x0)**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)**0.25

k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
H = -0.5 * k**2 + V

psi = psi0
for step in range(steps):
    psi_hat = np.fft.fft(psi)
    
    psi_hat_new = psi_hat * np.exp(-1j * H * dt)
    psi = np.fft.ifft(psi_hat_new)
    
    if step % 100 == 0:
        plt.plot(x, np.abs(psi)**2, label=f'Time={step*dt:.2f}')
        
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Quantum Echoes: Double-Well Potential')
plt.legend()
plt.grid(True)
plt.show()
