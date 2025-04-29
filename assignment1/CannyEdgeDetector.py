import numpy as np
import cv2
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Add Gaussian noise to an image
    """
    row, col = image.shape
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def analyze_canny_noise_sensitivity(image_path, noise_levels=[10, 50, 100]):
    """
    Analyze Canny edge detector's sensitivity to different levels of Gaussian noise.
    """
    #Read and convert image to grayscale
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise ValueError("Could not read the image")
    
    results = {}
    
    # get Canny edges for original image
    original_edges = cv2.Canny(original, 100, 200)
    results['original'] = {
        'image': original,
        'edges': original_edges
    }
    
    #process each noise level
    for sigma in noise_levels:
        noisy_image = add_gaussian_noise(original, sigma=sigma)
        noisy_edges = cv2.Canny(noisy_image, 100, 200)
        
        results[f'noise_{sigma}'] = {
            'image': noisy_image,
            'edges': noisy_edges
        }
    
    return results

def plot_results(results):
    noise_levels = len(results) - 1
    fig, axes = plt.subplots(2, noise_levels + 1, figsize=(15, 8))
    
    # pplot original image and its edges
    axes[0, 0].imshow(results['original']['image'], cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(results['original']['edges'], cmap='gray')
    axes[1, 0].set_title('Original Edges')
    axes[1, 0].axis('off')
    
    # plot noisy images and their edges
    col = 1
    for key in results:
        if key == 'original':
            continue
            
        sigma = int(key.split('_')[1])
        
        axes[0, col].imshow(results[key]['image'], cmap='gray')
        axes[0, col].set_title(f'Noise sigma={sigma}')
        axes[0, col].axis('off')
        
        axes[1, col].imshow(results[key]['edges'], cmap='gray')
        axes[1, col].set_title(f'Edges (sigma={sigma})')
        axes[1, col].axis('off')
        
        col += 1
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "image.png"
    noise_levels = [10, 50, 100]  # Low, medium, and high noise
    
    try:
        results = analyze_canny_noise_sensitivity(image_path, noise_levels)
        plot_results(results)
    except Exception as e:
        print(f"Error: {e}")