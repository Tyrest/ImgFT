from typing import Tuple

import numpy as np
from PIL import Image


def load_image(path: str) -> Image.Image:
    """
    Load an image from the specified path.
    """
    return Image.open(path)


def to_grayscale(image: Image.Image) -> Image.Image:
    """
    Convert image to grayscale using optimal grayscale (luminance).
    """
    return image.convert("L")


def compute_fft(image_array: np.ndarray) -> np.ndarray:
    """
    Compute the 2D FFT of the image array.
    """
    return np.fft.fft2(image_array)


def normalize_fft(fft_data: np.ndarray) -> np.ndarray:
    """
    Normalize the FFT so that the image is essentially 1x1.
    This is done by dividing by the total number of pixels.
    """
    return fft_data / fft_data.size


def get_top_frequencies(
    fft_data: np.ndarray, k: int = 8192
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort frequencies by intensity (magnitude) and return the top k.
    Returns:
        top_k_values: The complex values (real, imag) of the top k frequencies.
        top_k_positions: The normalized 2D positions (y, x) of the top k frequencies.
    """
    # Flatten the FFT data
    flat_fft = fft_data.flatten()

    # Compute magnitude (intensity)
    magnitudes = np.abs(flat_fft)

    # Sort indices by magnitude in descending order
    # Note: argsort sorts in ascending order, so we reverse it
    sorted_indices = np.argsort(magnitudes)[::-1]

    # Keep top k
    # If the image is smaller than k pixels, we take all of them
    actual_k = min(k, flat_fft.size)
    top_k_indices = sorted_indices[:actual_k]
    top_k_values = flat_fft[top_k_indices]

    # Convert flat indices to 2D positions and normalize
    rows, cols = fft_data.shape
    y_indices, x_indices = np.unravel_index(top_k_indices, (rows, cols))

    # Normalize to [0, 1)
    y_norm = y_indices / rows
    x_norm = x_indices / cols

    top_k_positions = np.stack((y_norm, x_norm), axis=-1)

    # If we have fewer than k, we might want to pad?
    # But for now let's assume images are large enough (> 90x90 pixels).
    # If strictly 8192 is required, we should pad.
    if actual_k < k:
        pad_size = k - actual_k
        top_k_values = np.pad(top_k_values, (0, pad_size), "constant")

        # Pad positions with -1
        pos_padding = np.full((pad_size, 2), -1.0)
        top_k_positions = np.concatenate((top_k_positions, pos_padding), axis=0)

    # Convert to (k, 2) array of real numbers
    top_k_values_split = np.stack((top_k_values.real, top_k_values.imag), axis=-1)

    return top_k_values_split, top_k_positions


def reconstruct_image(
    values: np.ndarray, positions: np.ndarray, shape: Tuple[int, ...]
) -> np.ndarray:
    """
    Reconstruct the image from the top frequencies.
    """
    # Create empty fft spectrum
    fft_data = np.zeros(shape, dtype=np.complex128)
    flat_fft = fft_data.ravel()

    # Filter out padding (positions == -1)
    # Check first coordinate
    valid_mask = positions[:, 0] != -1
    valid_positions = positions[valid_mask]
    valid_values = values[valid_mask]

    # Denormalize positions
    rows, cols = shape
    y_indices = np.round(valid_positions[:, 0] * rows).astype(int)
    x_indices = np.round(valid_positions[:, 1] * cols).astype(int)

    # Clip to ensure valid indices
    y_indices = np.clip(y_indices, 0, rows - 1)
    x_indices = np.clip(x_indices, 0, cols - 1)

    # Convert to flat indices
    valid_indices = np.ravel_multi_index((y_indices, x_indices), shape)

    # Convert back to complex
    complex_values = valid_values[:, 0] + 1j * valid_values[:, 1]

    # Assign values
    flat_fft[valid_indices] = complex_values

    # Reshape back
    fft_data = fft_data.reshape(shape)

    # Inverse FFT
    # Since we normalized by dividing by size, we multiply by size to restore scale
    fft_data = fft_data * fft_data.size
    reconstructed = np.fft.ifft2(fft_data)

    # Return magnitude (should be real)
    return np.abs(reconstructed)
