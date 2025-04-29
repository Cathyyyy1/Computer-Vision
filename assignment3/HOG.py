import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

def compute_gradients(image):
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
    
    # Convert directions to range [0, 180)
    gradient_direction = gradient_direction % 180
    
    return gradient_magnitude, gradient_direction

def threshold_gradients(gradient_magnitude, threshold):
    thresholded_magnitude = gradient_magnitude.copy()
    #Threshold small gradient magnitudes to zero
    thresholded_magnitude[thresholded_magnitude < threshold] = 0
    return thresholded_magnitude

def create_cell_grid(image, cell_size):
    height, width = image.shape
    
    #calculate grid dimensions
    m = height // cell_size
    n = width // cell_size
    
    # Crop image
    crop_height = m * cell_size
    crop_width = n * cell_size
    cropped_image = image[:crop_height, :crop_width]
    
    return (m, n), cropped_image

def compute_hog(gradient_magnitude, gradient_direction, grid_shape, cell_size, num_bins=6, accumulate_magnitudes=True):

    m, n = grid_shape
    hog_features = np.zeros((m, n, num_bins))
    
    for i in range(m):
        for j in range(n):
            # Extract cell region
            cell_magnitude = np.zeros((cell_size, cell_size))
            cell_direction = np.zeros((cell_size, cell_size))
            for y in range(cell_size):
                for x in range(cell_size):
                    cell_y = i * cell_size + y
                    cell_x = j * cell_size + x
                    cell_magnitude[y, x] = gradient_magnitude[cell_y, cell_x]
                    cell_direction[y, x] = gradient_direction[cell_y, cell_x]
            
            # Flatten the cell data
            cell_magnitude_flat = cell_magnitude.flatten()
            cell_direction_flat = cell_direction.flatten()
            
            #process each pixel in the cell
            for pixel_idx in range(len(cell_magnitude_flat)):
                mag = cell_magnitude_flat[pixel_idx]
                direction = cell_direction_flat[pixel_idx]
                
                #sskip pixels with zero magnitude
                if mag == 0:
                    continue
                
                #find the bin for this orientation
                if -15 <= direction < 15 or 165 <= direction < 180:
                    bin_idx = 0  # [-15°, 15°)
                elif 15 <= direction < 45:
                    bin_idx = 1  # [15°, 45°)
                elif 45 <= direction < 75:
                    bin_idx = 2  # [45°, 75°)
                elif 75 <= direction < 105:
                    bin_idx = 3  # [75°, 105°)
                elif 105 <= direction < 135:
                    bin_idx = 4  # [105°, 135°)
                elif 135 <= direction < 165:
                    bin_idx = 5  # [135°, 165°)
                else:
                    #should not happen
                    bin_idx = 0
                
                #accumulate either magnitude or count
                if accumulate_magnitudes:
                    hog_features[i, j, bin_idx] += abs(mag)
                else:
                    hog_features[i, j, bin_idx] += 1
    
    return hog_features

def normalize_blocks(hog_features, block_size=2, epsilon=0.001):
    m, n, num_bins = hog_features.shape
    #this is the new dimensions after blocking with stride=1
    new_m = m - block_size + 1
    new_n = n - block_size + 1
    block_feature_size = block_size * block_size * num_bins
    normalized_features = np.zeros((new_m, new_n, block_feature_size))
    
    #process each block
    for i in range(new_m):
        for j in range(new_n):
            block_features = []
            for bi in range(block_size):
                for bj in range(block_size):
                    block_features.append(hog_features[i+bi, j+bj, :])
            #concatenate all histograms in the block
            block_features = np.concatenate(block_features)
            #L2 norm (formula got from assignment 3 worksheet)
            l2_norm = np.sqrt(np.sum(block_features**2) + epsilon**2)
            normalized_block = block_features / l2_norm
            normalized_features[i, j, :] = normalized_block
    
    return normalized_features

def save_hog_to_file(normalized_features, output_file):
    new_m, new_n, block_feature_size = normalized_features.shape
    #this will create new_m * new_n rows, each with block_feature_size elements
    reshaped_features = normalized_features.reshape(new_m * new_n, block_feature_size)
    
    with open(output_file, 'w') as f:
        np.savetxt(f, reshaped_features, fmt='%.8f')
    
    print(f"Saved normalized HOG features to {output_file} with {new_m * new_n} rows")

def downsample_image(image_path, scale_factor=None):

    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    original_height, original_width = img.shape[:2]
    print(f"Original image size: {original_width}x{original_height}")
    
    if scale_factor is not None:
        # Calculate new dimensions
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
    
    #Resize the image
    downsampled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    downsampled_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2GRAY)

    print(f"Downsampled to: {new_width}x{new_height}")
    
    return downsampled_img

def visualize_hog(image, hog_features, cell_size, bin_edges, scale=1.0):
    m, n, num_bins = hog_features.shape
    plt.figure(figsize=(14, 10))
    plt.imshow(image, cmap='gray', alpha=0.5)
    
    #bin centers
    bin_centers = np.zeros(len(bin_edges))
    extended_bin_edges = np.append(bin_edges, bin_edges[0] + 180)
    for i in range(len(bin_edges)):
        bin_centers[i] = (extended_bin_edges[i] + extended_bin_edges[i+1]) / 2
    
    #ccell centers
    y_centers = np.zeros(m)
    x_centers = np.zeros(n)
    
    for i in range(m):
        y_centers[i] = (i * cell_size) + (cell_size / 2)
        
    for j in range(n):
        x_centers[j] = (j * cell_size) + (cell_size / 2)
    
    X, Y = np.meshgrid(x_centers, y_centers)
    
    for bin_idx, angle in enumerate(bin_centers):
        #to radians
        rad_angle = angle * np.pi / 180
        
        #rotate 90 degrees  make lines perpendicular to gradient direction
        perpendicular_angle = rad_angle + (np.pi / 2)
        
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for i in range(m):
            for j in range(n):
                magnitude = hog_features[i, j, bin_idx]
                if magnitude > 0:
                    U[i, j] = np.cos(perpendicular_angle) * magnitude * scale
                    V[i, j] = np.sin(perpendicular_angle) * magnitude * scale
        
        #1 quiver
        plt.quiver(X, Y, U, V, color='r', scale_units='xy', scale=1, width=0.001,
                   headwidth=0, headlength=0, headaxislength=0)
        #2 quiver
        plt.quiver(X, Y, -U, -V, color='r', scale_units='xy', scale=1, width=0.001,
                   headwidth=0, headlength=0, headaxislength=0)
    
    plt.axis('off')
    plt.tight_layout()
    return plt

def compare_hog_euclidean(hog1, hog2):
    flat_hog1 = hog1.reshape(-1)
    flat_hog2 = hog2.reshape(-1)
    
    euclidean_dist = np.sqrt(np.sum((flat_hog1 - flat_hog2) ** 2))
    normalized_euclidean_dist = euclidean_dist / np.sqrt(len(flat_hog1))
    
    return normalized_euclidean_dist
def main():
    '''Section 1'''
    image_path = "image1.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    cell_size = 8  # Size of each cell tau
    threshold = 50
    num_bins = 6  # Number of orientation bins
    
    gradient_magnitude, gradient_direction = compute_gradients(image)
    thresholded_magnitude = threshold_gradients(gradient_magnitude, threshold)
    grid_shape, cropped_image = create_cell_grid(image, cell_size)
    
    #create cropped gradient maps and match the grid
    cropped_magnitude = threshold_gradients(thresholded_magnitude[:grid_shape[0]*cell_size, :grid_shape[1]*cell_size], threshold)
    cropped_direction = gradient_direction[:grid_shape[0]*cell_size, :grid_shape[1]*cell_size]
    
    #compute HOG features using both approaches
    hog_magnitude = compute_hog(cropped_magnitude, cropped_direction, grid_shape, cell_size, num_bins, accumulate_magnitudes=True)
    hog_occurrence = compute_hog(cropped_magnitude, cropped_direction, grid_shape, cell_size, num_bins, accumulate_magnitudes=False)
    
    bin_edges = np.array([-15, 15, 45, 75, 105, 135])
    
    plt_magnitude = visualize_hog(cropped_image, hog_magnitude, cell_size, bin_edges, scale=0.003)
    plt_magnitude.suptitle('HOG Features (Accumulated Magnitudes)', fontsize=18)
    plt_magnitude.savefig('hog_magnitude.png', dpi=300)
    
    plt_occurrence = visualize_hog(cropped_image, hog_occurrence, cell_size, bin_edges, scale=0.5)
    plt_occurrence.suptitle('HOG Features (Occurrence Counts)', fontsize=18)
    plt_occurrence.savefig('hog_occurrence.png', dpi=300)
    
    print("Magnitude-based HOG shape:", hog_magnitude.shape)
    print("Occurrence-based HOG shape:", hog_occurrence.shape)
    plt.show()

    '''Section 2'''
    block_size = 2
    epsilon = 0.001
    
    normalized_features = normalize_blocks(hog_magnitude, block_size, epsilon)
        
    # Output file name
    output_file = image_path.rsplit('.', 1)[0] + '.txt'
    
    # Save normalized features to file
    save_hog_to_file(normalized_features, output_file)
    
    print(f"Processed {image_path}:")
    print(f"  Original HOG shape: {hog_magnitude.shape}")
    print(f"  Normalized HOG shape: {normalized_features.shape}")
    print(f"  Expected shape: ({grid_shape[0]-1}, {grid_shape[1]-1}, {block_size*block_size*num_bins})")

    '''Section 3'''

    '''Process Image without Flash'''
    image_path_without_flash = "without_flash.jpg"
    img_without_flash = downsample_image(image_path_without_flash, scale_factor=0.25)
    gradient_magnitude_without_flash, gradient_direction_without_flash = compute_gradients(img_without_flash)
    thresholded_magnitude_without_flash = threshold_gradients(gradient_magnitude_without_flash, threshold)
    grid_shape_without_flash, cropped_image_without_flash = create_cell_grid(img_without_flash, cell_size)
    
    # Create cropped gradient maps and match the grid
    cropped_magnitude_without_flash = threshold_gradients(thresholded_magnitude_without_flash[:grid_shape_without_flash[0]*cell_size, :grid_shape_without_flash[1]*cell_size], threshold)
    cropped_direction_without_flash = gradient_direction_without_flash[:grid_shape_without_flash[0]*cell_size, :grid_shape_without_flash[1]*cell_size]
    
    #compute HOG features using both approaches
    hog_magnitude_without_flash = compute_hog(cropped_magnitude_without_flash, cropped_direction_without_flash, grid_shape_without_flash, cell_size, num_bins, accumulate_magnitudes=True)
    hog_occurrence_without_flash = compute_hog(cropped_magnitude_without_flash, cropped_direction_without_flash, grid_shape_without_flash, cell_size, num_bins, accumulate_magnitudes=False)
    
    bin_edges = np.array([-15, 15, 45, 75, 105, 135])
    
    plt_magnitude = visualize_hog(cropped_image_without_flash, hog_magnitude_without_flash, cell_size, bin_edges, scale=0.0008)
    plt_magnitude.suptitle('HOG Features Without Flash(Accumulated Magnitudes)', fontsize=18)
    plt_magnitude.savefig('hog_magnitude_without_flash.png', dpi=300)
    
    plt_occurrence = visualize_hog(cropped_image_without_flash, hog_occurrence_without_flash, cell_size, bin_edges, scale=0.2)
    plt_occurrence.suptitle('HOG Features Without Flash (Occurrence Counts)', fontsize=18)
    plt_occurrence.savefig('hog_occurrence_without_flash.png', dpi=300)
    plt.show()

    normalized_features_without_flash = normalize_blocks(hog_magnitude_without_flash, block_size, epsilon)
        
    # Output file
    output_file = image_path_without_flash.rsplit('.', 1)[0] + '.txt'
    
    # Save normalized features to file
    save_hog_to_file(normalized_features_without_flash, output_file)
    
    print(f"Processed {image_path_without_flash}:")
    print(f"  Original HOG shape: {hog_magnitude_without_flash.shape}")
    print(f"  Normalized HOG shape: {normalized_features_without_flash.shape}")
    print(f"  Expected shape: ({grid_shape_without_flash[0]-1}, {grid_shape_without_flash[1]-1}, {block_size*block_size*num_bins})")
    
    '''Process Image with Flash'''
    image_path_with_flash = "with_flash.jpg"
    img_with_flash = downsample_image(image_path_with_flash, scale_factor=0.25)
    gradient_magnitude_with_flash, gradient_direction_with_flash = compute_gradients(img_with_flash)
    thresholded_magnitude_with_flash = threshold_gradients(gradient_magnitude_with_flash, threshold)
    grid_shape_with_flash, cropped_image_with_flash = create_cell_grid(img_with_flash, cell_size)
    
    # Create cropped gradient maps and match the grid
    cropped_magnitude_with_flash = threshold_gradients(thresholded_magnitude_with_flash[:grid_shape_with_flash[0]*cell_size, :grid_shape_with_flash[1]*cell_size], threshold)
    cropped_direction_with_flash = gradient_direction_with_flash[:grid_shape_with_flash[0]*cell_size, :grid_shape_with_flash[1]*cell_size]
    
    #compute HOG features using both approaches
    hog_magnitude_with_flash = compute_hog(cropped_magnitude_with_flash, cropped_direction_with_flash, grid_shape_with_flash, cell_size, num_bins, accumulate_magnitudes=True)
    hog_occurrence_with_flash = compute_hog(cropped_magnitude_with_flash, cropped_direction_with_flash, grid_shape_with_flash, cell_size, num_bins, accumulate_magnitudes=False)
    
    plt_magnitude = visualize_hog(cropped_image_with_flash, hog_magnitude_with_flash, cell_size, bin_edges, scale=0.0008)
    plt_magnitude.suptitle('HOG Features With Flash(Accumulated Magnitudes)', fontsize=18)
    plt_magnitude.savefig('hog_magnitude_with_flash.png', dpi=300)
    
    plt_occurrence = visualize_hog(cropped_image_with_flash, hog_occurrence_with_flash, cell_size, bin_edges, scale=0.2)
    plt_occurrence.suptitle('HOG Features With Flash (Occurrence Counts)', fontsize=18)
    plt_occurrence.savefig('hog_occurrence_with_flash.png', dpi=300)
    plt.show()

    normalized_features_with_flash = normalize_blocks(hog_magnitude_with_flash, block_size, epsilon)
        
    output_file = image_path_with_flash.rsplit('.', 1)[0] + '.txt'
    
    # Save normalized features to file
    save_hog_to_file(normalized_features_with_flash, output_file)
    
    print(f"Processed {image_path_with_flash}:")
    print(f"  Original HOG shape: {hog_magnitude_with_flash.shape}")
    print(f"  Normalized HOG shape: {normalized_features_with_flash.shape}")
    #check if they match
    print(f"  Expected shape: ({grid_shape_with_flash[0]-1}, {grid_shape_with_flash[1]-1}, {block_size*block_size*num_bins})")

    '''Section 5'''
    euclidean_dist = compare_hog_euclidean(hog_magnitude_with_flash, hog_magnitude_without_flash)
    euclidean_dist_normalized = compare_hog_euclidean(normalized_features_with_flash, normalized_features_without_flash)
    print(f"Euclidean distance non-normalized HOG feature: {euclidean_dist}")
    print(f"Euclidean distance normalized HOG feature: {euclidean_dist_normalized}")
    
    print(f"Euclidean distance improvement: {euclidean_dist - euclidean_dist_normalized}")
    
if __name__ == "__main__":
    main()