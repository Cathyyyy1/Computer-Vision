import numpy as np
import cv2  
import matplotlib.pyplot as plt

def convolve2D(image, kernel):
    """
    This function computes convolution.
    """
    # Store Image dimensions
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    #Calculate Padding sizes
    pad_height = kernel_height // 2 # Integar division
    pad_width = kernel_width // 2
    
    # output array
    output = np.zeros_like(image)
    
    #Pad the input image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')
    
    # Flip kernel, but in this case they are the same.
    #because sobel operators are symmetric kernels.
    kernel = np.flipud(np.fliplr(kernel))

    # convolution computation
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            #Compute convolution at this pixel
            output[i, j] = np.sum(region * kernel)
    
    return output

def compute_gradient_magnitude(image):
    """
    Compute gradient magnitude of an image using Sobel kernels.
    """
    # Sobel kernels
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    
    # These are gradients along x and y directions
    gradient_x = convolve2D(image, sobel_x)
    gradient_y = convolve2D(image, sobel_y)
    
    # gradient magnitude
    squared_x = gradient_x**2
    squared_y = gradient_y**2
    gradient_magnitude = np.sqrt(squared_x + squared_y)
    
    return gradient_magnitude

'''
# for testing purposes
if __name__ == "__main__":
    # Load image and convert to grayscale.
    img = cv2.imread('image.png')  # Load as BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Convert to float64 and normalize to range [0, 1]
    image_array = gray.astype(np.float64) / 255.0
    
    #Compute gradient magnitude.
    result = compute_gradient_magnitude(image_array)
    
    # Scale result to [0, 255] for visualization
    result = (result * 255).astype(np.uint8)
    
    #Create figure with two subplots
    plt.figure(figsize=(12, 6))
    
    #original image
    plt.subplot(121)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    # gradient magnitude image
    plt.subplot(122)
    plt.title('Gradient Magnitude')
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    
    # Show the plots
    plt.tight_layout()
    plt.show()
    
    #cv2.imwrite('gradient_magnitude.png', result)
    '''
