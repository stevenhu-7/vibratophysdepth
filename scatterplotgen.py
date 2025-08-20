import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/stevenhu/Downloads/vibratophysdepth/Cello/CelloPhysDepth.csv')

y_col = 'Average Depth of Pitch Variation (cents)'
x_col = 'Physical Center of Vibrato'

x = df[x_col].values
y = df[y_col].values

plt.figure(figsize=(12,8))
plt.scatter(x, y, alpha=0.7, label='Data', color='tab:blue')

# Smooth x for fits
x_fit = np.linspace(x.min(), x.max(), 300)

# Linear fit
slope, intercept = np.polyfit(x, y, 1)
y_lin_fit = slope * x_fit + intercept
y_lin = slope * x + intercept
r2_lin = 1 - (np.sum((y - y_lin)**2) / np.sum((y - np.mean(y))**2))
plt.plot(x_fit, y_lin_fit, color='red', label='Linear fit')

# Equation text
plt.text(0.05, 0.95, f'Linear: y={slope:.3f}x+{intercept:.3f}\n$R^2$={r2_lin:.3f}',
         color='red', fontsize=11, transform=plt.gca().transAxes, verticalalignment='top')

# Power fit: y = A * x^B
mask_pow = (x > 0) & (y > 0)
B_pow, logA_pow = np.polyfit(np.log(x[mask_pow]), np.log(y[mask_pow]), 1)
A_pow = np.exp(logA_pow)
y_pow_fit = A_pow * x_fit**B_pow
y_pow = A_pow * x[mask_pow]**B_pow
r2_pow = 1 - (np.sum((y[mask_pow] - y_pow)**2) / np.sum((y[mask_pow] - np.mean(y[mask_pow]))**2))
plt.plot(x_fit, y_pow_fit, color='green', linestyle='--', label='Power fit')
plt.text(0.05, 0.85, f'Power: y={A_pow:.3f}x^{B_pow:.3f}\n$R^2$={r2_pow:.3f}',
         color='green', fontsize=11, transform=plt.gca().transAxes, verticalalignment='top')

# Exponential fit: y = A * exp(B * x)
B_exp, logA_exp = np.polyfit(x, np.log(y), 1)
A_exp = np.exp(logA_exp)
y_exp_fit = A_exp * np.exp(B_exp * x_fit)
y_exp = A_exp * np.exp(B_exp * x)
r2_exp = 1 - (np.sum((y - y_exp)**2) / np.sum((y - np.mean(y))**2))
plt.plot(x_fit, y_exp_fit, color='blue', linestyle=':', label='Exponential fit')
plt.text(0.05, 0.75, f'Exp: y={A_exp:.3f}e^({B_exp:.3f}x)\n$R^2$={r2_exp:.3f}',
         color='blue', fontsize=11, transform=plt.gca().transAxes, verticalalignment='top')

plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title(' ')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()