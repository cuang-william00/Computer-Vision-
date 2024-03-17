
### Supporting code for Computer Vision Assignment 1
### See "Assignment 1.ipynb" for instructions

import math

import numpy as np
from skimage import io

def load(img_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.
    HINT: Converting all pixel values to a range between 0.0 and 1.0
    (i.e. divide by 255) will make your life easier later on!

    Inputs:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    
    # YOUR CODE HERE
    
    #Storing numpy array which is divided by 255 constant value in out
    out = np.divide(io.imread(img_path), 255)
    
    return out

def print_stats(image):
    """ Prints the height, width and number of channels in an image.
        
    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels).
        
    Returns: none
                
    """
    
    # YOUR CODE HERE
    
    #Storing the height, width and channels of an image in relevant variable
    height, width, channels = image.shape
    
    #Printing the output
    print(height, width, channels)
    
    return None

def crop(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds. Use array slicing.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index 
        start_col (int): The starting column index 
        num_rows (int): Number of rows in our cropped image.
        num_cols (int): Number of columns in our cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """


    ### YOUR CODE HERE
    out = image[start_col:num_cols , start_row:num_rows]

    return out


def change_contrast(image, factor):
    """Change the value of every pixel by following

                        x_n = factor * (x_p - 0.5) + 0.5

    where x_n is the new value and x_p is the original value.
    Assumes pixel values between 0.0 and 1.0 
    If you are using values 0-255, change 0.5 to 128.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        factor (float): contrast adjustment

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """


    ### YOUR CODE HERE
    #Utilising code logic to generate a constrast image form the original image
    out = np.clip( (factor * (image - 0.5) + 0.5) * 255, 0, 255).astype("uint8")

    return out


def resize(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.
    i.e. for each output pixel, use the value of the nearest input pixel after scaling

    Inputs:
        input_image: RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """

    #Declaring parameters
    rows, cols, nchannels = input_image.shape
    newr = int(rows * output_rows)
    newc = int(cols * output_cols)
    canvas = np.zeros((newr, newc, 3))

    #Defining new row and column proportions as 'x' and 'y'
    x = rows/newr
    y = cols/newc
    
    #Traversing through the canvas and painting the image
    for i in range(newr):
        for j in range(newc):
            #Calculating the array to be resized
            temp = int(y*j)
            temp1 = int(x*i)
            #Put the new resized pixel array into canvas array's blank page
            canvas[ i, j, : ] = input_image[ temp1, temp, : ]
    
    #Returning the output
    out = canvas
    return out

def greyscale(input_image):
    """Convert a RGB image to greyscale. 
    A simple method is to take the average of R, G, B at each pixel.
    Or you can look up more sophisticated methods online.
    
    Inputs:
        input_image: RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.

    Returns:
        np.ndarray: Greyscale image, with shape `(output_rows, output_cols)`.
    """

    #First declaring variables
    #Extracting colours and making a blank image canvas
    red = input_image[:, :, 0]
    green = input_image[:, :, 1]
    blue = input_image[:, :, 2]
    out = np.zeros_like(input_image)
    
    #Calculating average of the pixel values
    avg = (red + green + blue) / 3

    #Painting the image
    out[:, :, 0] = avg
    out[:, :, 1] = avg
    out[:, :, 2] = avg

    return out

def binary(image, num):
    """ This function converts a greyscale image to a binary mask.
    If the pixel value > num then the pizel is 1, else 0.
    We return an image as an array with the applied steps
    """

    #Getting binary images from input image

    #First create a blank canvas
    out = np.zeros_like(image)
    
    #Now paint  the canvas with 1 or 0 binary mask
    out [num < image] = 5/5

    return out

def conv2D(image, kernel):
    """ Convolution of a 2D image with a 2D kernel. 
    Convolution is applied to each pixel in the image.
    Assume values outside image bounds are 0.
    
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """

    ### YOUR CODE HERE
    # Prepare the output array with the same dimensions as the input image
    output_image = np.zeros_like(image)
    img_height, img_width = image.shape
    kern_height, kern_width = kernel.shape
    
    # Pad the input image to handle edge cases
    vertical_pad = kern_height // 2
    horizontal_pad = kern_width // 2
    padded_img = np.pad(image, ((vertical_pad, vertical_pad), (horizontal_pad, horizontal_pad)), mode='constant')

    # Flip the kernel both vertically and horizontally
    flipped_kernel = np.flipud(np.fliplr(kernel))

    # Perform convolution by sliding the kernel over the image
    for row in range(img_height):
        for col in range(img_width):
            #Extract the current region of interest from the padded image
            current_region = padded_img[row:row+kern_height, col:col+kern_width]
            #Element-wise multiplication and summation to apply the filter
            output_image[row, col] = np.sum(current_region * flipped_kernel)

    return output_image

def test_conv2D():
    """ A simple test for your 2D convolution function.
        You can modify it as you like to debug your function.
    
    Returns:
        None
    """

    # Test code written by 
    # Simple convolution kernel.
    kernel = np.array(
    [
        [1,0,1],
        [0,0,0],
        [1,0,0]
    ])

    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1

    # Run your conv_nested function on the test image
    test_output = conv2D(test_img, kernel)

    # Build the expected output
    expected_output = np.zeros((9, 9))
    expected_output[2:7, 2:7] = 1
    expected_output[5:, 5:] = 0
    expected_output[4, 2:5] = 2
    expected_output[2:5, 4] = 2
    expected_output[4, 4] = 3

    # Test if the output matches expected output
    assert np.max(test_output - expected_output) < 1e-10, "Your solution is not correct."


def conv(image, kernel):
    """Convolution of a RGB or grayscale image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    out = None
    ### YOUR CODE HERE
    out=np.zeros_like(image)
    height, width, depth = image.shape
    height_kernel, width_kernel = kernel.shape
    
    #Handle boundary cases by padding the image
    temp=(height_kernel//2, height_kernel//2)
    temp1=(width_kernel//2, width_kernel//2)
    temp2=(0,0)
    canvas=np.pad(image, (temp,temp1,temp2), mode='constant')

    #Flip the kernel
    kernel=np.flipud(kernel) #upside down
    kernel=np.fliplr(kernel) #left right

    #Traverse the image and perform the convolution
    for dep in range(depth):
      for i in range(height):
        for j in range(width):

          #Make a smaller image from the padded image (canvas)
          smaller_image = canvas[i:height_kernel+i, j:width_kernel+j, dep]
          
          #Apply filtering effect on the smaller image
          mult=smaller_image * kernel
          
          #Place the effect on the original image
          out[i,j,dep]=np.sum(mult)
    
    #Clipping the image to valid range
    np.clip( out*255, 0, 255).astype("uint8")

    return out

    
def gauss2D(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function.
       You should not need to edit it.
       
    Args:
        size: filter height and width
        sigma: std deviation of Gaussian
        
    Returns:
        numpy array of shape (size, size) representing Gaussian filter
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def corr(image, kernel):
    """Cross correlation of a RGB image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    out = None
    ### YOUR CODE HERE

    return out


