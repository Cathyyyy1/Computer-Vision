import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def create_black_square_image(size=500, square_size=100):
    image = np.ones((size, size))
    start = (size - square_size) // 2
    end = start + square_size
    image[start:end, start:end] = 0
    return image


def normalized_laplacian_of_gaussian(x, y, sigma):
    factor = (x**2 + y**2) / (2 * sigma**2) - 1
    return (sigma**2) * (1 / (np.pi * sigma**4)) * factor * np.exp(-(x**2 + y**2) / (2 * sigma**2))

def create_normalized_log_kernel(window_size, sigma):
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
        
    kernel = np.zeros((window_size, window_size))
    center = window_size // 2
    for i in range(window_size):
        for j in range(window_size):
            x = i - center
            y = j - center
            kernel[i, j] = normalized_laplacian_of_gaussian(x, y, sigma)
            
    return kernel

def apply_normalized_log(image, sigma, window_size=None):
    
    #normalized LoG kernel
    kernel = create_normalized_log_kernel(window_size, sigma)
    
    return ndimage.convolve(image, kernel)

def find_optimal_sigma(min_sigma=1, max_sigma=100, num_samples=50, window_size=None):
    
    image = create_black_square_image(500, 100)
    
    sigma_values = np.arange(min_sigma, max_sigma)
    max_responses = []
    
    print("\nSigma Value\tMaximum Response")
    print("-" * 30)
    
    for sigma in sigma_values:
        
        # Apply normalized LoG filter
        filtered = apply_normalized_log(image, sigma, window_size)
        
        max_response = np.max(np.abs(filtered))
        max_responses.append(max_response)
        
        #for debugging purposes
        print(f"{sigma:.4f}\t\t{max_response:.6f}")
    
    #find maximum response
    optimal_idx = np.argmax(max_responses)
    optimal_sigma = sigma_values[optimal_idx]
    
    print("\n" + "=" * 60)
    print(f"Optimal sigma: {optimal_sigma:.4f}, Maximum response: {max_responses[optimal_idx]:.6f}")
    print("=" * 60)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    
    # for debugging
    plt.subplot(2, 2, 2)
    filtered = apply_normalized_log(image, optimal_sigma, window_size)
    plt.imshow(filtered, cmap='gray')
    plt.title(f'Normalized LoG Response (sigma = {optimal_sigma:.2f})')
    
    #for debugging
    plt.subplot(2, 2, 3)
    kernel = create_normalized_log_kernel(window_size, optimal_sigma)
    plt.imshow(kernel, cmap='viridis')
    plt.colorbar()
    plt.title(f'Normalized LoG Kernel (sigma = {optimal_sigma:.2f})')
    
    plt.subplot(2, 2, 4)
    plt.plot(sigma_values, max_responses) 
    plt.axvline(x=optimal_sigma, color='r', linestyle='--')
    plt.xlabel('Sigma')
    plt.ylabel('Maximum Response Magnitude')
    plt.grid(True)
    plt.title(f'Maximum Response vs Sigma (Optimal sigma = {optimal_sigma:.2f})')
    
    plt.tight_layout()
    plt.show()
    
    return optimal_sigma, sigma_values, max_responses


if __name__ == "__main__":
    window_size = 51
    
    optimal_sigma, sigma_values, max_responses = find_optimal_sigma(
        min_sigma=1, 
        max_sigma=100, 
        num_samples=50,
        window_size=window_size
    )
    
    print(f"The optimal sigma value that maximizes the normalized LoG response magnitude is: {optimal_sigma:.4f}")
    
    print("\nDetailed Results:")
    print(f"{'Sigma':<10} {'Max Response':<15}")
    print("-" * 30)
    
    peak_response = max(max_responses)
    
    for i, (sigma, response) in enumerate(zip(sigma_values, max_responses)):

        print(f"{sigma:<10.4f}{response:<15.6f}")