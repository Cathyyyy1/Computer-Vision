import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_gray(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.uint8)

def compute_homography(src_points, dst_points):
    
    #build matrix A
    A = []
    for i in range(len(src_points)):
        x, y = src_points[i]
        u, v = dst_points[i]
        A.append([x, y, 1, 0, 0, 0, -x*u, -y*u, -u])
        A.append([0, 0, 0, x, y, 1, -x*v, -y*v, -v])
    
    A = np.array(A)
    
    #solve for h using SVD
    _, _, VT = np.linalg.svd(A)
    h = VT[-1]
    H = h.reshape(3, 3)
    
    H = H / H[2, 2]
    
    return H



def process_case(case, img1_path, img2_path, point_pairs=None):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # to grayscale
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    gray1 = rgb_to_gray(img1_rgb)
    gray2 = rgb_to_gray(img2_rgb)
    
    #point pairs
    # use switch statement stated in question 4
    if point_pairs is None:
        if case == 'A':
            src_points = np.array([
                [929, 382],
                [757, 547],
                [735, 341],
                [885, 238],
                [764, 467],
                [1097, 233],
                [1045, 414]
            ], dtype=np.float32)

            dst_points = np.array([
                [793, 697],
                [629, 869],
                [599, 663],
                [739, 554],
                [634, 790],
                [949, 537],
                [910, 722]
            ], dtype=np.float32)
        elif case == 'B':
            src_points = np.array([
                [1071, 242],
                [995, 203],
                [931, 381],
                [842, 644],
                [755, 546],
                [768, 467],
                [860, 353],
                [883, 235],
                [995, 772]
            ], dtype=np.float32)

            dst_points = np.array([
                [949, 428],
                [901, 395],
                [872, 572],
                [829, 840],
                [775, 739],
                [779, 657],
                [830, 545],
                [846, 424],
                [919, 963]
            ], dtype=np.float32)
        elif case == 'C':
            src_points = np.array([
                [702, 735],
                [496, 742],
                [838, 664],
                [536, 646],
                [681, 649],
                [791, 614],
                [720, 542],
                [657, 546],
                [570, 559],
                [596, 497],
                [993, 804]
            ], dtype=np.float32)

            dst_points = np.array([
                [656, 948],
                [443, 943],
                [828, 857],
                [523, 851],
                [669, 843],
                [796, 807],
                [752, 739],
                [689, 743],
                [595, 754],
                [651, 693],
                [919, 997]
            ], dtype=np.float32)
            

    else:
        src_points = np.array([(p[0], p[1]) for p in point_pairs], dtype=np.float32)
        dst_points = np.array([(p[2], p[3]) for p in point_pairs], dtype=np.float32)
    
    #find h
    H = compute_homography(src_points, dst_points)
    
    #question 1
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(gray1, cmap='gray')
    plt.title(f"Image I ({img1_path})")
    for i, (x, y) in enumerate(src_points):
        plt.plot(x, y, 'rs', markersize=10)  # Red squares

    plt.subplot(1, 2, 2)
    plt.imshow(gray2, cmap='gray')
    plt.title(f"Image I~ ({img2_path})")
    
    for i, (x, y) in enumerate(dst_points):
        plt.plot(x, y, 'rs', markersize=10)  # Red squares
    
    plt.suptitle(f"Case {case}: Selected Point Pairs in Grayscale Images")
    plt.tight_layout()
    plt.savefig(f"case_{case}_grayscale_selected_points.png")
    
    #Question 3
    plt.figure(figsize=(8, 8))
    plt.imshow(gray2, cmap='gray')
    plt.title(f"Case {case}: Original points (red) and mapped points (green)")
    
    #original points
    for i, (x, y) in enumerate(dst_points):
        plt.plot(x, y, 'rs', markersize=10)
    
    # Map source points to destination using homography
    src_homogeneous = np.hstack([src_points, np.ones((len(src_points), 1))])
    mapped_homogeneous = np.dot(H, src_homogeneous.T).T
    mapped_points = mapped_homogeneous[:, :2] / mapped_homogeneous[:, 2:3]
    
    for i, (x, y) in enumerate(mapped_points):
        plt.plot(x, y, 'gs', markersize=10)
        #plt.plot([dst_points[i][0], x], [dst_points[i][1], y], 'g--', alpha=0.5)
    
    plt.savefig(f"case_{case}_mapped_points.png")
    
    return H

def create_combined_image(H, img1_path, img2_path, padding=100, case ='F'):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    gray1 = rgb_to_gray(img1_rgb)
    gray2 = rgb_to_gray(img2_rgb)
    
    h1, w1 = gray1.shape
    h2, w2 = gray2.shape
    
    #H_inv to map from dst to src
    H_inv = np.linalg.inv(H)
    
    #I make the new image dimensions match img2 with some padding to ensure all content is visible
    new_width = w2 + 2 * padding
    new_height = h2 + 2 * padding
    
    #RGB values are initialized to zero,
    new_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    offset_x = padding
    offset_y = padding
    #i used meshgrid for the new image coordinates
    y_coords, x_coords = np.mgrid[0:new_height, 0:new_width]
    
    #copy the second image to the blue and green channels
    for y in range(new_height):
        for x in range(new_width):
            img2_x = x - offset_x
            img2_y = y - offset_y
            
            if 0 <= img2_x < w2 and 0 <= img2_y < h2:
                new_img[y, x, 1] = gray2[img2_y, img2_x]
                new_img[y, x, 2] = gray2[img2_y, img2_x]
    
    #Now, I apply the inverse homography to find corresponding points in img1
    ones = np.ones_like(x_coords)
    coords = np.stack((x_coords - offset_x, y_coords - offset_y, ones), axis=-1)
    points = coords.reshape(-1, 3)
    #apply the inverse homography
    mapped_points = np.dot(H_inv, points.T).T
    
    x_mapped = mapped_points[:, 0] / mapped_points[:, 2]
    y_mapped = mapped_points[:, 1] / mapped_points[:, 2]
    x_mapped = x_mapped.reshape(new_height, new_width)
    y_mapped = y_mapped.reshape(new_height, new_width)
    
    for y in range(new_height):
        for x in range(new_width):
            #coorresponding coordinate in image 1
            img1_x = int(round(x_mapped[y, x]))
            img1_y = int(round(y_mapped[y, x]))
            
            if 0 <= img1_x < w1 and 0 <= img1_y < h1:
                #copy red channel
                new_img[y, x, 0] = gray1[img1_y, img1_x]
    
    # Create visualization to show the results
    plt.figure(figsize=(12, 10))
    plt.imshow(new_img)
    plt.title(f"Case {case}: Combined Visualization: Red = I, Green/Blue = I~")
    plt.axis('off')
    plt.savefig(f"case_{case}_combined_homography_visualization.png", bbox_inches='tight')
    plt.show()
    
    #For debug
    # plt.figure(figsize=(12, 10))
    # plt.imshow(gray2, cmap='gray')
    # plt.title("Visualization of mapped points from Image 1 to Image 2")
    # sample_points = []
    # for _ in range(1000):
    #     x = np.random.randint(0, w1)
    #     y = np.random.randint(0, h1)
    #     sample_points.append([x, y, 1])
    # sample_points = np.array(sample_points)
    # mapped_samples = np.dot(H, sample_points.T).T
    # mapped_samples[:, 0] = mapped_samples[:, 0] / mapped_samples[:, 2]
    # mapped_samples[:, 1] = mapped_samples[:, 1] / mapped_samples[:, 2]
    
    # plt.scatter(mapped_samples[:, 0], mapped_samples[:, 1], c='r', s=1, alpha=0.5)
    # plt.savefig("point_mapping_visualization.png", bbox_inches='tight')
    
    return new_img


if __name__ == "__main__":
    hallway1 = "hallway1.jpg"
    hallway2 = "hallway2.jpg"
    hallway3 = "hallway3.jpg"
    
    results = {}
    
    print("\nSelect a case to process:")
    print("A - The right wall of hallway1.jpg to the right wall of hallway2.jpg")
    print("B - The right wall of hallway1.jpg to the right wall of hallway3.jpg")
    print("C - The floor of hallway1.jpg to the floor of hallway3.jpg")
    
    case_choice = input("Enter case (A/B/C): ").strip().upper()

    if case_choice == 'A':
        print("\nProcessing Case A: The right wall of hallway1.jpg to the right wall of hallway2.jpg")
        H_A = process_case('A', hallway1, hallway2)
        results['A'] = (H_A)
        
        print(f"\nCase A:")
        print("Estimated Homography Matrix H:")
        print(H_A)
        
        print("\nCreating combined visualization for Case A")
        combined_img_A = create_combined_image(H_A, hallway1, hallway2, 100, 'A')
        
    elif case_choice == 'B':
        print("\nProcessing Case B: The right wall of hallway1.jpg to the right wall of hallway3.jpg")
        H_B = process_case('B', hallway1, hallway3)
        results['B'] = (H_B)
        
        print(f"\nCase B:")
        print("Estimated Homography Matrix H:")
        print(H_B)
        
        print("\nCreating combined visualization for Case B")
        combined_img_B = create_combined_image(H_B, hallway1, hallway3, 100, 'B')
        
    elif case_choice == 'C':
        print("\nProcessing Case C: The floor of hallway1.jpg to the floor of hallway3.jpg")
        H_C = process_case('C', hallway1, hallway3)
        results['C'] = (H_C)
        
        print(f"\nCase C:")
        print("Estimated Homography Matrix H:")
        print(H_C)
        
        print("\nCreating combined visualization for Case C")
        combined_img_C = create_combined_image(H_C, hallway1, hallway3, 100, 'C')
        
    else:
        print("Invalid case selection. Please choose A, B, or C.")
    
