import cv2  
import matplotlib.pyplot as plt
import numpy as np


def compute_adaptive_threshold(gradient_image, epsilon=0.01):
    """
    Compute adaptive threshold for edge detection.
    """
    height, width = gradient_image.shape

    # Step 1 from worksheet:
    # Initial threshold tau0 as average intensity
    tau_prev = np.sum(gradient_image) / (height * width)
    
    while True:
        # Step 2 from worksheet:
        # Categorize pixels into lower and upper classes
        lower_mask = gradient_image < tau_prev
        upper_mask = gradient_image >= tau_prev
        
        lower_class = gradient_image[lower_mask]
        upper_class = gradient_image[upper_mask]
        
        # Step 3 from worksheet:
        # Compute average gradient magnitudes for each class
        if len(lower_class) > 0:
            m_L = np.mean(lower_class)
        else:
            m_L = 0

        if len(upper_class) > 0:
            m_H = np.mean(upper_class)
        else:
            m_H = 0
        
        # Step 4 from worksheet:
        # Update threshold
        tau_current = (m_L + m_H) / 2
        
        # Check convergence
        if abs(tau_current - tau_prev) <= epsilon:
            break
            
        tau_prev = tau_current
    
    return tau_current

def apply_threshold(gradient_image, threshold):
    """
    Apply threshold to create binary edge map.
    """
    edge_map = np.zeros_like(gradient_image, dtype=np.uint8)
    mask = gradient_image >= threshold
    edge_map[mask] = 255
    return edge_map

def detect_edges(gradient_image, epsilon=0.01):
    """
    Detect edges using adaptive thresholding.
    """
    # Compute threshold
    threshold = compute_adaptive_threshold(gradient_image, epsilon)
    
    # Apply threshold to create edge map
    edge_map = apply_threshold(gradient_image, threshold)
    
    return edge_map, threshold

'''
# for testing purposes.
if __name__ == "__main__":
    # Load gradient image 
    #assuming it's already computed from previous step
    
    img = cv2.imread('image.png')  # Load as BGR
    gradient_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edge_map, threshold = detect_edges(gradient_image)
    
    print(f"Final threshold value: {threshold}")
    
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    plt.title('Gradient Image')
    plt.imshow(gradient_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(122)
    plt.title('Edge Map')
    plt.imshow(edge_map, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
'''