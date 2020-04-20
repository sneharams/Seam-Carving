import random
import time
import argparse
import math

#################################################################################################################################
# METHOD HELPERS                                                                                                                #
#################################################################################################################################

def get_kerneled_pixel(pixel_map, kernel, a, b, w, h):
    """
    Returns the appropriate value for a pixel based off of the kernel        
    """
    value = 0 
    # sets kernel around pixel in question
    a -= len(kernel)//2 
    b -= len(kernel)//2
    for i in range(len(kernel)):
        #returns the color based on if the kernel bounds exceed the image bounds or not
        for j in range(len(kernel)):
            value += kernel[i][j] * get_pixel(pixel_map, a+i, b+j, w, h) 
    return value

def get_pixel(pixel_map, row, col, w, h):
    """
    Returns the pixel value with row & col adjusted for bounds
    """
    # adjust row
    if row >= h:
        row = h-1
    if row < 0:
        row = 0
    # adjust col
    if col >= w:
        col = w-1
    if col < 0:
        col = 0
    return pixel_map[row][col]

def correlate(pixel_map, kernel, w, h):
    """
    Compute the result of correlating the given image with the given kernel.

    Kernel Representation: List of Lists (each nested list contains one row of values, 
    total number of lists represents total number of rows in kernel)
    """
    new_pixels = list()
    # goes through each pixel and applies kernel to each one
    for x in range(h):
        pixel_row = list()
        for y in range(w):
            new_pixel = get_kerneled_pixel(pixel_map, kernel, x, y, w, h)   # calculates new pixel from original image
            pixel_row.append(new_pixel)        # sets pixel in result image
        new_pixels.append(pixel_row)
    return new_pixels

def get_energy_calc(resultX, resultY, r, c):
    oX = resultX[r][c]**2
    oY = resultY[r][c]**2
    return math.sqrt(oX + oY)

def get_energy(pixel_map, kX, kY, a, b, w, h):
    """
    Returns the appropriate value for a pixel based off of the kernel        
    """
    valueX = 0 
    valueY = 0
    # sets kernel around pixel in question
    a -= len(kX)//2 
    b -= len(kX)//2
    for i in range(len(kX)):
        #returns the color based on if the kernel bounds exceed the image bounds or not
        for j in range(len(kX)):
            valueX += kX[i][j] * get_pixel(pixel_map, a+i, b+j, w, h)
            valueY += kY[i][j] * get_pixel(pixel_map, a+i, b+j, w, h)
    energy = math.sqrt(valueX**2 + valueY**2)
    return energy

#################################################################################################################################
# METHODS                                                                                                                       #
#################################################################################################################################

# ORIGINAL (1)###################################################################################################################

def original(gray_map, w, h):
    cum_map = calculate_one_direction(gray_map, w, h)
    seam_indices = list()
    low = 0
    high = w
    for r in range(h-1, -1, -1):
        lowest_index = low
        lowest_energy = cum_map[r][low]
        for c in range(low, high):
            if cum_map[r][c] < lowest_energy:
                lowest_index = c
                lowest_energy = cum_map[r][c]
        seam_indices.append(lowest_index)
        low = max(lowest_index-1, 0)
        high = min(lowest_index+2, w)
    return seam_indices

def calculate_one_direction(gray_map, w, h):
    cum_energy_map = list()

    # Kx and Ky kernels
    kernelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    kernelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    # Ox and Oy kerneled images (without rounding)
    resultX = correlate(gray_map, kernelX, w, h)
    resultY = correlate(gray_map, kernelY, w, h)

    def get_min_energy(low, high):
        min_energy = cum_energy_map[r-1][low]
        for i in range(low+1, high):
            if cum_energy_map[r-1][i] < min_energy:
                min_energy = cum_energy_map[r-1][i]
        return min_energy

    cum_energy_row = list()
    for c in range(w):     
        cum_energy_row.append(get_energy_calc(resultX, resultY, 0, c))
    cum_energy_map.append(cum_energy_row)

    # for rows > 0, sums pixel with 3 adjacent pixels above
    for r in range(1, h):
        cum_energy_row = list()
        for c in range(w):
            # find adj pixel range
            low = max(0, c-1)
            high = min(w, c+2)
            # calc cum energy for pixel
            energy = get_energy_calc(resultX, resultY, r, c)
            adj_energy = get_min_energy(low, high)
            cum_energy_row.append(energy + adj_energy)
        # update maps
        cum_energy_map.append(cum_energy_row)
    return cum_energy_map

# RECURSIVE (2) #################################################################################################################

def recursive(gray_map, w, h):
    kernelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    kernelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    # (energy, cum_energy, min_adj_col)
    cum_energy_map = [[(None, None) for _ in range(w)] for _ in range(h)]
    min_energy = [math.inf]

    def recursive_helper(r, c, cum_energy, seam):
        # print(r, c)
        # for row in cum_energy_map:
        #     print(row)
        # print() 
        current_cum_energy = 0
        energy = 0
        # pixel already visited
        if cum_energy_map[r][c][0] != None:
            energy = cum_energy_map[r][c][0]
            current_cum_energy = energy + cum_energy
            # already found smaller energy seam up to this point
            if current_cum_energy >= cum_energy_map[r][c][1]:
                return (False, False)
        # uncharted territory
        else:
            energy = get_energy(gray_map, kernelX, kernelY, r, c, w, h)
            current_cum_energy = energy + cum_energy
        if current_cum_energy >= min_energy[0]:
            return (False, False)
        # update map & seam
        cum_energy_map[r][c] = (energy, current_cum_energy)
        # cum_energy_map[r][c] = (int(energy), int(current_cum_energy))
        # check if last row
        if r+1 == h:
            min_energy[0] = current_cum_energy
            return current_cum_energy, seam + [c]
        low = max(0, c-1)
        high = min(w, c+2)
        ret = (False, False)
        for c2 in range(low, high):
            c2_seam = recursive_helper(r+1, c2, current_cum_energy, seam+[c])
            if c2_seam[0] != False:
                if not ret[0] or c2_seam[0] < ret[0]:
                    ret = c2_seam
        return ret

    seam = (False, False)
    for i in range(w):
        i_seam = recursive_helper(0, i, 0, [])
        if i_seam[0] != False:
            if not seam[0] or i_seam[0] < seam[0]:
                seam = i_seam
    seam[1].reverse()
    return seam[1]

# DJIKSTRAS (IN PROGRESS) #######################################################################################################

# def djikstras(gray_map, w, h):
#     kernelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
#     kernelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

#     seams = 
#     # (energy, cum_energy, min_adj_col)
#     cum_energy_map = [[(None, None) for _ in range(w)] for _ in range(h)]
#     min_energy = [math.inf]

#     def recursive_helper(r, c, cum_energy, seam):
       


#################################################################################################################################
# TESTING                                                                                                                       #
#################################################################################################################################

def make_grayscale_map(w, h):
    gray_map = list()
    for _ in range(h):
        row = list()
        for _ in range(w):
            row.append(random.randint(0, 256))
        gray_map.append(row)
    return gray_map

def check_method(num_trials, method_correct, method_test, w, h):
    print("testing on", w, "by", h, "image")
    for i in range(num_trials):
        # make map
        gray_map = make_grayscale_map(w, h)
        # compare seams
        seam_expected = method_correct(gray_map, w, h)
        seam_result = method_test(gray_map, w, h)
        if seam_expected != seam_result:
            print("method failed on trial:", i)
            print("expected:", seam_expected)
            print("result:", seam_result)
            return
    print("method passed all trials:", num_trials)

def time_trials(num_trials, method, w, h):
    print("testing on", w, "by", h, "image")
    elapsed = 0
    for _ in range(num_trials):
        # make map
        gray_map = make_grayscale_map(w, h)
        # time seam calculation
        start = time.time()
        method(gray_map, w, h)
        elapsed += time.time() - start
    print("time elapsed for", num_trials, "seams is:", round(elapsed, 2), "seconds")

def incremental_time_trials(num_trials, method, n):
    print("testing up to", n, "by", n, "image")
    for d in range(2,n):
        elapsed = 0
        for _ in range(num_trials):
            # make map
            gray_map = make_grayscale_map(d, d)
            # time seam calculation
            start = time.time()
            method(gray_map, d, d)
            elapsed += time.time() - start
        print("time elapsed for", num_trials, "seams of size", d, "is:", round(elapsed, 2), "seconds")
    print()

#################################################################################################################################
# TEST                                                                                                                          #
#################################################################################################################################

if __name__ == '__main__':
    # time_trials(1, recursive_sorted, 10, 10)
    # incremental_time_trials(1, original, 200)
    # incremental_time_trials(1, recursive, 200)
        