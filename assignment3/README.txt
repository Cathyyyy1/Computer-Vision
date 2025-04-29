Question 1 is implemented in NLoG.py file.
I implemented the formula shown in question 1 of this assignment.
I used a window for the Normalized Laplacian of Gaussian and convolve with the entire image.
I used existing library skimage for convolution.
I printed the black square on a white background, the image after convolution and the window of NLoG for verifying my results.
The responses of changing sigma is also plotted.

Question 3 is implemented in HOG.py file.
Each section in question 3 is labeled in def main() function.
Each section corresponse to each sub part of question 3. 
I also grouped the part for computing HOG array for the image with and without flash light and save normalized vector.
This file takes the input from image1.png, image2.png, with_flash.jpg, and without_flash.jpg.

Question 4 s implemented in CornerDetector.py file
I determined threshold for corner detection using 97 percentiles.
It takes the input from University_College,_University_of_Toronto.jpg and University_College_Lawn,_University_of_Toronto,_Canada.jpg
You can change the value for sigma in the main function to test various sigma value responses.
I implemented this task based on what we learned in class.
Each step discussed in class is labeled in the code as well.