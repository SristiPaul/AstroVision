import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from mpl_toolkits.mplot3d import Axes3D

# === Step 1: Load and Normalize Lunar Image ===
image = cv2.imread('lunar_surface.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image 'lunar_surface.jpg' not found.")

# Resize to 256x256
image = cv2.resize(image, (256, 256))
img_norm = image.astype(np.float32) / 255.0

# === Step 2: Compute Gradients ===
sobel_x = cv2.Sobel(img_norm, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(img_norm, cv2.CV_64F, 0, 1, ksize=5)

# === Step 3: Poisson Integration to Estimate Elevation Map ===
def poisson_integration(grad_x, grad_y):
    h, w = grad_x.shape
    f = np.zeros((h, w), dtype=np.float32)
    f[:, :-1] += grad_x[:, :-1]
    f[:, 1:] -= grad_x[:, :-1]
    f[:-1, :] += grad_y[:-1, :]
    f[1:, :] -= grad_y[:-1, :]
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    denom = (2 * np.cos(np.pi * x / w) - 2) + (2 * np.cos(np.pi * y / h) - 2)
    denom[0, 0] = 1
    z = np.real(ifft2(fft2(f) / denom))
    return z

# Run integration
dem_estimate = poisson_integration(sobel_x, sobel_y)

# === Step 4: Display as 3D Terrain ===
X, Y = np.meshgrid(np.arange(dem_estimate.shape[1]), np.arange(dem_estimate.shape[0]))
Z = dem_estimate

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='terrain', linewidth=0, antialiased=True)

ax.set_title("3D Terrain Model from Lunar Image")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Elevation")
plt.tight_layout()
plt.show()
