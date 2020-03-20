import math

from PIL import Image as PILImage
from abc import ABC, abstractmethod

class Carver:

    IMAGE_TYPES = set(['.png', '.jpeg', '.jpg'])

    """
    Rep invariant:
    - filename 
        - String type
        - ends with one of the strings in IMAGE_TYPES
        - length > length of corresponging image type (ex. ".png")
    
    - image
        - Dictionary type
        - image['height'] 
            - int type
            - > 0
        - image['width'] 
            - int type 
            - > 0
        - image['pixels'] 
            - Dictionary type
            - keys of 'r', 'g', and 'b' map to equal length lists
            - 2D List type of ints
            - length of list = height
            - length of every nested list = width
            - 0 <= every value in list <= 255

    Safety from rep exposure:
    - TODO 
    """

    def __check_rep(self):
        pass

    def __init__(self, filename):
        self.__filename = filename
        self.__image = self.__load_image(filename)

    def __load_image(self, filename):
        with open(filename, 'rb') as img_handle:
            img = Image.open(img_handle)
            img_data = img.getdata()

            pixels = list(img_data)

            w, h = img.size

            if img.mode.startswith('RGB'):
                return ColorImage(w, h, pixels)
                
            elif img.mode.startswith('L'):
                return GreyscaleImage(w, h, pixels)
            
            else:
                raise ValueError('Unsupported image mode: %r' % img.mode)

            # this part was pulled from old lab, not sure why it's needed
            w, h = img.size
            return {'height': h, 'width': w, 'pixels': pixels}

    

class Image():

    def __check_rep(self):
        # Note: technically I should make sure every pixel value is an int, but that would be v slow
        #       Do you have ideas?

        if self.__img_og == None or self.__width == None or self.__width == None or self.__pixels == None:
            raise Exception("Can't have None values in rep.")

        if type(self.__img_og) != dict:
            raise Exception("Invalid original image.")

        keys_og = self.__img_og.keys()
        if (keys_og) != 3 or 'height' not in keys_og or 'width' not in keys_og or 'pixels' not in keys_og:
            raise Exception("Incorrect keys in original image.")

        if type(self.__height) != int or self.__height <= 0:
            raise Exception("Invalid height.")
            
        if type(self.__width) != int or self.__width <= 0:
            raise Exception("Invalid width.")

        if type(self.__pixels) != list or len(self.__pixels) <= 0:
            raise Exception("Invalid pixels")


    def __init__(self, w, h, pixels, grayscale):
        # for check_rep()
        self.__img_og = None
        self.__height = None
        self.__width = None
        self.__pixels = None

        self.__img_og = {
            'height': h,
            'width' : w,
            'pixels': pixels
        }

        pixels_gray_1D = grayscale

        # keep track of modified dimensions
        self.__width = w
        self.__height = h

        # convert to 2D grid of pixels
        self.__pixels = list()
        self.__pixels_gray_2D = list()
        for i in range(h):
            low = i * w
            high = low + w
            self.__pixels.append(list(pixels[low: high])) # list() prevents mutation of input (i think lol)
            self.__pixels_gray_2D.append(list(pixels_gray_1D[low: high]))

        self.__check_rep()

    def __get_pixel(self, row, col):
        #return self.__energy_map(row, col)
        pass

    def __get_energy(self, pixels):
        # # this is from the lab too
        # # Kx and Ky kernels
        # kernelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        # kernelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

        # # Ox and Oy kerneled images (without rounding)
        # resultX = correlate(image, kernelX)
        # resultY = correlate(image, kernelY)


        # # calculates final pixels for results based on Ox and Oy
        # for i in range(len(image['pixels'])):
        #     result['pixels'].append(math.sqrt(resultX['pixels'][i]**2 + resultY['pixels'][i]**2))

        # # round pixel values before returning
        # round_and_clip_image(result)
        # return result
        pass

    def __get_new_pixel(self, kernel, a, b):
        """
        Returns the appropriate value for a pixel based off of the kernel        
        """
        value = 0 

        # sets kernel around pixel in question
        a -= len(kernel)//2 
        b -= len(kernel)//2

        for x in range(len(kernel)):
            #returns the color based on if the kernel bounds exceed the image bounds or not
            for y in range(len(kernel)):
                value += kernel[x][y] * self.__get_pixel(a+x, b+y) 
        return value

    def __set_pixel(self, x, y, c):
        """
        Takes in 2D pixel location and sets the corresponding pixel in the image to c.
        """
        # makes sure input pixel is within range of image pixels
        # checks x
        if x < 0: 
            x = 0
        elif x >= self.__width:
            x = self.__width-1
        # checks y
        if y < 0:
            y = 0
        elif y >= self.__height:
            y = self.__height-1

        #self.__energy_map[x][y] = c

    def __correlate(self, pixels, kernel):
        """
        Compute the result of correlating the given image with the given kernel.

        Kernel Representation: List of Lists (each nested list contains one row of values, 
        total number of lists represents total number of rows in kernel)
        """

        # goes through each pixel and applies kernel to each one
        for x in range(self.__width):
            for y in range(self.__height):
                new_pixel = self.__get_new_pixel(kernel, x, y)   # calculates new pixel from original image
                self.__set_pixel(x, y, new_pixel)        # sets pixel in result image

    


class ColorImage(Image):

    def __init__(self, w, h, pixels):
        pixels_gray_1D = self.__get_grayscale_pixels()
        super().__init__(w, h, pixels, pixels_gray_1D)

    def __get_grayscale_pixels(self):
        # this is a formula we used for the lab; it turns rgb tuple into single int
        return list(map(lambda px: px[0]*.299+px[1]*.587+px[2]*.114, self.__pixels))

    
class GreyscaleImage(Image):

    def __init__(self, w, h, pixels):
        super().__init__(w, h, pixels, pixels)

    