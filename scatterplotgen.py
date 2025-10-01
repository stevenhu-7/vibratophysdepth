import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/stevenhu/Desktop/Vibrato Paper/Cello/CelloPhysDepth.csv')
# df = df[df['File Source'] == 'Commercial Recording']

y_col = 'Vibrato Frequency (Hz)'
x_col = 'Physical Depth of Vibrato'

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
plt.plot(x_fit, y_lin_fit, color='red',
         label=f'Linear Fit: y={slope:.4f}x+{intercept:.4f}, $R^2$={r2_lin:.4f}')

# Power fit: y = A * x^B
mask_pow = (x > 0) & (y > 0)
B_pow, logA_pow = np.polyfit(np.log(x[mask_pow]), np.log(y[mask_pow]), 1)
A_pow = np.exp(logA_pow)
y_pow_fit = A_pow * x_fit**B_pow
y_pow = A_pow * x[mask_pow]**B_pow
r2_pow = 1 - (np.sum((y[mask_pow] - y_pow)**2) / np.sum((y[mask_pow] - np.mean(y[mask_pow]))**2))
plt.plot(x_fit, y_pow_fit, color='green', linestyle='--',
         label=f'Power Fit: y={A_pow:.4f}x^{B_pow:.4f}, $R^2$={r2_pow:.4f}')

# Exponential fit: y = A * exp(B * x)
B_exp, logA_exp = np.polyfit(x, np.log(y), 1)
A_exp = np.exp(logA_exp)
y_exp_fit = A_exp * np.exp(B_exp * x_fit)
y_exp = A_exp * np.exp(B_exp * x)
r2_exp = 1 - (np.sum((y - y_exp)**2) / np.sum((y - np.mean(y))**2))
plt.plot(x_fit, y_exp_fit, color='blue', linestyle=':',
         label=f'Exp Fit: y={A_exp:.4f}e^({B_exp:.4f}x), $R^2$={r2_exp:.4f}')

# Make labels and legend larger, and add space between axis labels and ticks
label_fontsize = 20
tick_fontsize = 16
legend_fontsize = 17
labelpad_value = 16  # Increase for more space

plt.xlabel(x_col, fontsize=label_fontsize, labelpad=labelpad_value)
plt.ylabel(y_col, fontsize=label_fontsize, labelpad=labelpad_value)
plt.title(' ', fontsize=label_fontsize)
plt.legend(fontsize=legend_fontsize, loc='best', frameon=True)
plt.grid(True)
plt.tight_layout()
plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
plt.show()