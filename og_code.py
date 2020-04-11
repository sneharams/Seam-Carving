#!/usr/bin/env python3

import math
import time
from PIL import Image

# FROM LAB0

def get_pixel(image, x, y):
    """
    Takes in 2D pixel location and returns the corresponding pixel in the image.
    """
    # makes sure input pixel is within range of image pixels
    # checks x
    if x < 0: 
        x = 0
    elif x >= image['width']:
        x = image['width']-1
    # checks y
    if y < 0:
        y = 0
    elif y >= image['height']:
        y = image['height']-1

    # calculates pixel position in 1D list & returns
    return image['pixels'][image['width'] * y + x]


def set_pixel(image, x, y, c):
    """
    Takes in 2D pixel location and sets the corresponding pixel in the image to c.
    """
    # makes sure input pixel is within range of image pixels
    # checks x
    if x < 0: 
        x = 0
    elif x >= image['width']:
        x = image['width']-1
    # checks y
    if y < 0:
        y = 0
    elif y >= image['height']:
        y = image['height']-1

    # calculates pixel position in 1D list & sets pixel
    image['pixels'][image['width'] * y + x] = c


def apply_per_pixel(image, func):
    """
    Takes in lambda function and returns image with function applied to each pixel.
    """
    # initialize result image
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': list(image['pixels']) # copy of image rather than reference
    }

    # goes through each pixel in image
    for y in range(image['height']):
        for x in range(image['width']):
            pixel = get_pixel(image, x,y)       # pixel in original image
            new_pixel = func(pixel)             # pixel with function applied
            set_pixel(result, x, y, new_pixel)  # sets corresponding pixel in image to new pixel

    return result

def inverted(image):
    """
    Returns inverted lambda image by calling apply_per_pixel with inversion lambda function.
    """
    # pixel is inverted if you subtract original pixel from 255 (max pixel value)
    return apply_per_pixel(image, lambda c: 255-c)

def correlate(image, kernel):
    """
    Compute the result of correlating the given image with the given kernel.

    The output of this function should have the same form as a 6.009 image (a
    dictionary with 'height', 'width', and 'pixels' keys), but its pixel values
    do not necessarily need to be in the range [0,255], nor do they need to be
    integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    Kernel Representation: List of Lists (each nested list contains one row of values, 
    total number of lists represents total number of rows in kernel)
    """
    width = image['width']      # width of image
    height = image['height']    # height of image

    # initialize result image
    result = {
        'height': height,
        'width': width,
        'pixels': list(image['pixels'])
    }

    # goes through each pixel and applies kernel to each one
    for x in range(width):
        for y in range(height):
            new_pixel = get_new_pixel(kernel, x, y, height, width, image)   # calculates new pixel from original image
            set_pixel(result, x, y, new_pixel)                              # sets pixel in result image

    return result

def get_new_pixel(kernel, a, b, height, width, image):
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
                value += kernel[x][y] * get_pixel(image, a+x, b+y) 
        return value


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the 'pixels' list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """

    # goes through each pixel and makes sure its an integer between [0,255]
    for i in range(len(image['pixels'])):
        p = round(image['pixels'][i]) # rounds to integer value
        if p < 0:
            p = 0
        elif p > 255:
            p = 255
        image['pixels'][i] = p


# FILTERS

def blurred(image, n):
    """
    Return a new image representing the result of applying a box blur (with
    kernel size n) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """

    # blur kernel based on n
    kernel = [[1/(n*n)]*n for j in range(n)]

    # unrounded kerneled image
    result = correlate(image, kernel)

    # round pixel values before returning
    round_and_clip_image(result)
    return result

def sharpened(image, n):
    """
    Return a new image representing the result of applying a sharpening (with
    kernel size n) to the given input image.

    This process blurs the image based on a box blur with kernel size n, and
    then subtracts the blurred value from 2*original value
    """
    # find blur kernel based on n and computed blurred image (unrounded)
    kernel = [[1/(n*n)]*n for j in range(n)]
    blurred = correlate(image, kernel)

    # initialize result image
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': list(image['pixels'])
    }

    # unrounded kerneled image by subtracting blurred pixels from 2*og pixels
    for i in range(len(image['pixels'])):
        result['pixels'][i] = 2*image['pixels'][i]-blurred['pixels'][i]
    
    # round pixel values before returning
    round_and_clip_image(result)
    return result

def edges(image):
    """
    Return a new image representing the result of applying a Sobel operator filter
    to the given input image.

    This process kernels 2 images based on Kx and Ky kernels and then finds the square
    root of each resulting image pixels' squared and added together.
    """
    # Kx and Ky kernels
    kernelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    kernelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    # Ox and Oy kerneled images (without rounding)
    resultX = correlate(image, kernelX)
    resultY = correlate(image, kernelY)

    # initialize result image
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': []
    }

    # calculates final pixels for results based on Ox and Oy
    for i in range(len(image['pixels'])):
        result['pixels'].append(math.sqrt(resultX['pixels'][i]**2 + resultY['pixels'][i]**2))

    # round pixel values before returning
    round_and_clip_image(result)
    return result


# VARIOUS FILTERS

def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """
    def color_filter(image):
        height = image['height']
        width = image['width']

        # initialize arrays to hold red, green, and blue pixels seperately
        r_pixels = []
        g_pixels = []
        b_pixels = []

        # populate arrays
        for pixel_group in image['pixels']:
            r_pixels.append(pixel_group[0])
            g_pixels.append(pixel_group[1])
            b_pixels.append(pixel_group[2])

        # create image with each rgb pixel list and run filters on them
        r_result = filt({'height': height, 'width': width, 'pixels': r_pixels})
        g_result = filt({'height': height, 'width': width, 'pixels': g_pixels})
        b_result = filt({'height': height, 'width': width, 'pixels': b_pixels})

        # combine sepereate rgb pixel lists into one 
        rgb_pixels = []
        for i in range(len(r_result['pixels'])):
            rgb_pixels.append((r_result['pixels'][i], g_result['pixels'][i], b_result['pixels'][i]))

        # create final result image and return
        result = {
            'height': height,
            'width': width,
            'pixels': rgb_pixels
        }
        return result

    return color_filter



def make_blur_filter(n):
    def blurred_single_param(image):
        # calls existing blurred filter with n given
        return blurred(image, n)
    return blurred_single_param


def make_sharpen_filter(n):
    def sharpened_single_param(image):
        # calls existing sharpened filter with n given
        return sharpened(image, n)
    return sharpened_single_param


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """
    def combined_filters(image):
        # initialize result image to apply filters to
        result = {
            'height': image['height'],
            'width': image['width'],
            'pixels': list(image['pixels'])
        }

        # iterate through each filter and apply it to modified image
        for filter in filters:
            result = filter(result) # use filtered image for next loop
        return result

    # return filters as one function
    return combined_filters


# SEAM CARVING

# Main Seam Carving Implementation

def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image.
    """
    greyscale = greyscale_image_from_color_image(image) # greyscale version of image

    # initialize return colored image to remove pixels
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': list(image['pixels'])
    }

    # finds minimum energy seam and removes it from greyscale and colored pictures
    for i in range(1, ncols+1):
        energy_map = compute_energy(greyscale)                      # calculates energy map
        cum_energy_map = cumulative_energy_map(energy_map)          # calculates cumulative energy map
        min_energy_seam = minimum_energy_seam(cum_energy_map)       # calculates minimum energy seam
        greyscale = image_without_seam(greyscale, min_energy_seam)  # removes seam from greyscale image
        result = image_without_seam(result, min_energy_seam)        # removes seam from color image

    # return final color image
    return result




# Optional Helper Functions for Seam Carving

def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    greyscale_pixels = list(map(lambda px: px[0]*.299+px[1]*.587+px[2]*.114, image['pixels']))

    # go through the rgb values of each picture, and creates a list of greyscale pixels
    

    # greyscale_pixels = []
    # for pixel_group in image['pixels']:
    #     greyscale_pixels.append(round(.299 * pixel_group[0] + .587 * pixel_group[1] +.114 * pixel_group[2]))

    # creates final greyscale picture and returns
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': greyscale_pixels
    }
    return result


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    # calls implemented edges function
    return edges(grey)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy function),
    computes a "cumulative energy map" as described in the lab 1 writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    width = energy['width']

    cum_energy_map = []
    # go through each pixel and add the value of lowest adjacent pixel (row above)
    for i in range(len(energy['pixels'])):
        cum_energy_map.append(energy['pixels'][i])

        #don't need to check row above if pixel in first row
        if i >= width:
            # find possible range for adjacents and column of current pixel
            adjacent = range(i-width-1, i-width+2)
            column = i % width

            # adjust adjacent range based on edges of picture
            if column == 0:
                adjacent = adjacent[1:]
            if column == width-1:
                adjacent = adjacent[:-1]
        
            # choose the lowest val adjacent pixel and add to current pixel in cum_energy_map
            low = cum_energy_map[adjacent[0]]
            for j in adjacent[1:]:
                if cum_energy_map[j] < low:
                    low = cum_energy_map[j]
            cum_energy_map[-1] += low

    # create and return cumulative energy map
    result = {
        'height': energy['height'],
        'width': width,
        'pixels': cum_energy_map
    }
    return result


def minimum_energy_seam(c):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 1 writeup).
    """
    width = c['width']
    seam = []

    # list of indexes in current row and possible adjacent options
    curr_row = range(len(c['pixels'])-width, len(c['pixels']))
    adjacent = range(len(c['pixels'])-width, len(c['pixels']))

    # while the possible indexes are valid (>=0), find lowest possible val in adjacent range
    while adjacent[0] >= 0:
        low = c['pixels'][adjacent[0]]
        low_i = adjacent[0]
        for i in adjacent[1:]:
            if c['pixels'][i] < low:
                low = c['pixels'][i]
                low_i = i
        # add lowest value index to seam list
        seam.append(low_i)

        # update curr_row to be one lower (each index - width)
        # update adjacent to have adjacent pixels on row above
        curr_row = range(curr_row[0]-width, curr_row[-1]-width+1)
        adjacent = range(low_i-width-1, low_i-width+2)

        # adjust for edges of image
        if adjacent[0] < curr_row[0]:
            adjacent = adjacent[1:]
        if adjacent[-1] > curr_row[-1]:
            adjacent = adjacent[:-1]

    # return final list of indexes
    return seam
        


def image_without_seam(im, s):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    # start with original pixels
    shrunk_pixels = list(im['pixels'])

    # remove indexes from largest to smallest to prevent shrinking
    for i in sorted(s, reverse=True):
        del shrunk_pixels[i]

    # create and return final image
    result = {
        'height': im['height'],
        'width': im['width']-1,
        'pixels': shrunk_pixels
    }
    return result

def enhanced_smoothing(image, level):
    """
    Given an image and smoothing level [0-100], returns a smoothed image that
    avoids edges based on imputed level. Keeps edges prominent.

    ~25 level seems ideal for faces.
    """
    # set up result image
    result = {
        'height': image['height'],
        'width': image['width'],
        'pixels': []
    }

    # make blurred image
    blur_filter = color_filter_from_greyscale_filter(make_blur_filter(5))
    image_blurred = blur_filter(image)
    # make energy map
    image_map = edges(greyscale_image_from_color_image(image)) # energy map

    # calculate threshold for level of smoothing
    threshold = (level/100)
    #go through each pixel and choose original or blurred pixel based on threshold
    # energy map pixel value/255 has to be greater than threshold to use original
    for i in range(len(image_map['pixels'])):

        if image_map['pixels'][i]/255 > threshold:      # original image
            r_pixel = image['pixels'][i][0]
            g_pixel = image['pixels'][i][1]
            b_pixel = image['pixels'][i][2]
        else:                                           # blurred image
            r_pixel = image_blurred['pixels'][i][0]
            g_pixel = image_blurred['pixels'][i][1]
            b_pixel = image_blurred['pixels'][i][2]

        # adds chosen pixel to result
        result['pixels'].append((r_pixel, g_pixel, b_pixel))

    return result


# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES

def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img = img.convert('RGB')  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_color_image(image, filename, mode='PNG'):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode='RGB', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == '__main__':
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.

    # colored_invert = color_filter_from_greyscale_filter(make_blur_filter(9))
    # colored_inverted_cat = colored_invert(load_color_image('test_images/python.png'))
    # save_color_image(colored_inverted_cat, 'filtered_images/python.png')

    # colored_invert = color_filter_from_greyscale_filter(make_sharpen_filter(7))
    # colored_inverted_cat = colored_invert(load_color_image('test_images/sparrowchick.png'))
    # save_color_image(colored_inverted_cat, 'filtered_images/sparrowchick.png')

    # filter1 = color_filter_from_greyscale_filter(edges)
    # filter2 = color_filter_from_greyscale_filter(make_blur_filter(5))
    # filt = filter_cascade([filter1, filter1, filter2, filter1])
    # save_color_image(filt(load_color_image('test_images/frog.png')), 'filtered_images/frog.png')

    # seems like around 25 is best for faces
    
    # shows various levels of smoothing (25, 50, 75) and a completely blurred image (blur level 5)
    save_color_image(seam_carving((load_color_image('test_images/cats.png')), 60), 'result_images/correct60.png')


    pass
    
