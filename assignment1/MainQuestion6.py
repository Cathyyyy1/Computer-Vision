import numpy as np
import cv2
import matplotlib.pyplot as plt
from GradientMagnitude import compute_gradient_magnitude
from ThresholdAlgo import detect_edges
from GuassianBlurring import visualize_gaussian_filters, create_gaussian_filter

def process_image(image_path):
    """
    Process image by multiple steps.
    """
    sigmas = [1.0, 2]  # Sigma values - smaller values create sharper peaks
    sizes = [7, 15]      # Filter sizes - larger sizes capture more of the Gaussian tail

    # I: Visualize Gaussian filters in 2D and 3D
    visualize_gaussian_filters(sigmas, sizes)

    # pprint numerical values
    print("\nGaussian Filter values (sigma=1.0, size=7):")
    print(create_gaussian_filter(15, 2.0))

    """
    # for finer detail
    sigmas = [0.5, 1.0, 1.5]  # More gradual progression of sigma values
    sizes = [5, 7, 9]  # Smaller filter sizes for detailed comparison
    visualize_gaussian_filters(sigmas, sizes)
    """

    """
    # a very large filter
    large_filter = create_gaussian_filter(21, 3.0)
    plt.figure(figsize=(10, 8))
    plt.imshow(large_filter, cmap='hot')
    plt.colorbar()
    plt.title('Large Gaussian Filter (sigma=3.0, size=21)')
    plt.show()
    """

    # Load image and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert to float64 and normalize to range [0, 1]
    image_array = gray.astype(np.float64) / 255.0
    
    # II: Compute gradient magnitude
    gradient_magnitude = compute_gradient_magnitude(image_array)

    # III: Detectt edges using adaptive thresholding
    edge_map, threshold = detect_edges(gradient_magnitude)
    
    # Original image
    plt.figure(figsize=(8, 6))
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    #Gradient magnitude
    plt.figure(figsize=(8, 6))
    plt.title('Gradient Magnitude')

    #Scale result to [0, 255]
    #same as below
    #but, could lose informationn if gradient_magnitude is outside 0-1 range
    #gradient_viz = (gradient_magnitude * 255).astype(np.uint8)

    # Scale gradient magnitude to [0, 1] for visualization
    gradient_viz = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())
    plt.imshow(gradient_viz, cmap='gray')
    plt.axis('off')
    plt.show()
    
    #cv2.imwrite('gradient_magnitude.png', gradient_viz)

    # Edge mapp
    plt.figure(figsize=(8, 6))
    plt.title(f'Edge Map (threshold: {threshold:.3f})')
    plt.imshow(edge_map, cmap='gray')
    plt.axis('off')
    plt.show()
    

    #cv2.imwrite('edge_map.png', edge_map)
    return gradient_magnitude, edge_map, threshold

if __name__ == "__main__":
    image_path = 'image.png'
    gradient_magnitude, edge_map, threshold = process_image(image_path)
    print(f"Final threshold value: {threshold}")