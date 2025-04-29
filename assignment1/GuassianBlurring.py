import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_gaussian_filter(size, sigma):
    """
    Create a 2D Gaussian filter matrix used for image blurring and smoothing.
    The Gaussian filter creates a kernel where central pixels have higher weights
    and outer pixels have lower weights based on their distance from the center.
    """
    #Gaussian filter needs a central pixel to serve as origin.
    #and it needs to be asymmetrical around the center.
    # Thus, the size has to be odd to have a clear center pixel
    if size % 2 == 0:
        raise ValueError("Size must be odd to ensure symmetric filtering")
        
    # Create coordinate matrices for x and y distances from center
    # center is size//2 integer division to get the middle point
    center = size // 2
    #np.ogrid creates two 1D arrays that can be used for broadcasting
    y, x = np.ogrid[-center:center+1, -center:center+1]
    
    #Calculate Gaussian values using the 2D Gaussian formula
    # exp(-(x² + y²)/(2σ²)) 
    gaussian = np.exp(-(x*x + y*y) / (2.0 * sigma*sigma))
    
    # Normalize the filter so all weights sum to 1
    # This ensures the filter preserves the overall image intensity
    gaussian = gaussian / gaussian.sum()
    
    return gaussian

def visualize_gaussian_filters(sigmas, sizes):
    """
    Visualize 2D Gaussian filters as both 2D heatmaps and 3D surface plots.
    Creates a figure with multiple subplots showing different filter configurations.
    """
    # Create a figure with enough space for all visualizations
    fig = plt.figure(figsize=(20, 10))
    
    plt.subplots_adjust(
        left=0.1,   
        right=0.9,   
        bottom=0.1,  
        top=0.9,   
        wspace=0.4, 
        hspace=0.4   
    )

    # Iterate through pairs of sigma and size values
    for idx, (sigma, size) in enumerate(zip(sigmas, sizes), 1):
        #generate the Gaussian filter for current parameters
        gaussian = create_gaussian_filter(size, sigma)
        
        # 2D visualization
        plt.subplot(2, 3, idx)
        im = plt.imshow(gaussian, cmap='viridis')  # viridis colormap
        plt.colorbar(im)  # Add colorbar to show weight values
        plt.title(f'2D Gaussian (sigma={sigma}, size={size})')
        
        #3D visualization
        ax = fig.add_subplot(2, 3, idx+3, projection='3d')
        #create coordinate matrices for 3D plotting
        temp = size//2
        y, x = np.ogrid[-(temp):(temp)+1, -(temp):(temp)+1]
        x, y = np.meshgrid(x, y)  #create 2D coordinate grids
        # Create 3D surface plot
        surf = ax.plot_surface(x, y, gaussian, cmap='viridis')
        plt.colorbar(surf)  #Add colorbar for 3D plot
        ax.set_title(f'3D Gaussian (sigma={sigma}, size={size})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Value')
    
    plt.tight_layout()
    plt.show()
