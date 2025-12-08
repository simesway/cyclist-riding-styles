import numpy as np
from scipy.ndimage import gaussian_filter1d


# sigma_seconds: 0.3-0.6 for distances, 0.15-0.3 for velocities
def smooth(x, sigma_seconds, fps=12.5):
  """
  Smooth a 1D array with a Gaussian kernel.
  sigma_seconds: desired smoothing window in seconds.
  fps: sampling rate of x.
  """
  sigma = sigma_seconds * fps
  return gaussian_filter1d(x, sigma=sigma)