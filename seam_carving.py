import math

from PIL import Image as PILImage

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
        self.__type, self.__image = self.__load_image()

    def __load_image(self):
        with open(self.__filename, 'rb') as img_handle:
            img = PILImage.open(img_handle)
            img_data = img.getdata()
            pixels = list(img_data)
            w, h = img.size

            if img.mode.startswith('RGB'):
                pixels_gray = list(map(lambda px: px[0]*.299+px[1]*.587+px[2]*.114, pixels))
                return 'RGB', Image(w, h, pixels, pixels_gray)
                
            elif img.mode.startswith('L'):
                return 'L', Image(w, h, pixels, pixels)
            
            else:
                raise ValueError('Unsupported image mode: %r' % img.mode)

    def remove_vertical_seams(self, num):
        self.__image.remove_vertical_seams(num)

    def remove_horizontal_seams(self, num):
        pass

    def save_image(self, filename):
        image = self.__image.get_image()
        mode = 'RGB'
        if self.__type == 'RGB':
            out = PILImage.new(mode='RGB', size=(image['width'], image['height']))
        else:
            out = PILImage.new(mode='L', size=(image['width'], image['height']))
            mode = 'L'
        out.putdata(image['pixels'])
        if isinstance(filename, str):
            out.save(filename)
        else:
            out.save(filename, mode)
        out.close()

    def revert_image(self):
        pass

    def display_image(self):
        pass

    

class Image():

    def __check_rep(self):
        # # Note: technically I should make sure every pixel value is an int, but that would be v slow
        # #       Do you have ideas?

        # if self.__img_og == None or self.__width == None or self.__width == None or self.__pixels == None:
        #     raise Exception("Can't have None values in rep.")

        # if type(self.__img_og) != dict:
        #     raise Exception("Invalid original image.")

        # keys_og = self.__img_og.keys()
        # if (keys_og) != 3 or 'height' not in keys_og or 'width' not in keys_og or 'pixels' not in keys_og:
        #     raise Exception("Incorrect keys in original image.")

        # if type(self.__height) != int or self.__height <= 0:
        #     raise Exception("Invalid height.")
            
        # if type(self.__width) != int or self.__width <= 0:
        #     raise Exception("Invalid width.")

        # if type(self.__pixels) != list or len(self.__pixels) <= 0:
        #     raise Exception("Invalid pixels")
        pass


    def __init__(self, w, h, pixels, pixels_gray_1D):
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
        self.__energy_map, self.__cum_energy_map = self.__get_cum_energy()

        self.__check_rep()


    def __get_pixel(self, row, col):
        
        if row >= self.__height:
            row = self.__height-1
        if row < 0:
            row = 0

        if col >= self.__width:
            col = self.__width-1
        if col < 0:
            col = 0

        return self.__pixels_gray_2D[row][col]


    def __get_cum_energy(self):
        cum_energy_map = list()
        energy_map = list()

        # Kx and Ky kernels
        kernelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

        # Ox and Oy kerneled images (without rounding)
        resultX = self.__correlate(self.__pixels_gray_2D, kernelX)
        resultY = self.__correlate(self.__pixels_gray_2D, kernelY)

        energy_row = list()
        cum_energy_row = list()

        for c in range(len(self.__pixels_gray_2D[0])):     
            energy = math.sqrt(resultX[0][c]**2 + resultY[0][c]**2)
            cum_energy_row.append(energy)
            energy_row.append(energy)

        cum_energy_map.append(cum_energy_row)
        energy_map.append(energy_row)

        # for rows > 0, sums pixel with 3 adjacent pixels above
        for r in range(1, self.__height):
            energy_row = list()
            cum_energy_row = list()
            for c in range(self.__width):
                # find adj pixel range
                low = max(0, c-1)
                high = min(len(self.__pixels_gray_2D), c+2)
                # calc energy for pixel
                energy = math.sqrt(resultX[r][c]**2 + resultY[r][c]**2)
                energy_row.append(energy)
                # calc cummulative energy for pixel
                min_energy = cum_energy_map[r-1][low]
                for i in range(low+1, high):
                    if cum_energy_map[r-1][i] < min_energy:
                        min_energy = cum_energy_map[r-1][i]
                cum_energy_row.append(energy + min_energy)
            # update maps
            energy_map.append(energy_row)
            cum_energy_map.append(cum_energy_row)

        return energy_map, cum_energy_map

    def __remove_vertical_seam_and_update(self):
        seam_indices = list()
        low = 0
        high = len(self.__pixels[0])
        for r in range(len(self.__pixels)-1, -1, -1):
            lowest_index = low
            lowest_energy = self.__cum_energy_map[r][low]
            for c in range(low, high):
                if self.__cum_energy_map[r][c] < lowest_energy:
                    lowest_index = c
                    lowest_energy = self.__cum_energy_map[r][c]
            seam_indices.append(lowest_index)
            low = max(lowest_index-1, 0)
            high = min(lowest_index+2, self.__width)
            del self.__cum_energy_map[r][lowest_index]
            del self.__energy_map[r][lowest_index]
            del self.__pixels_gray_2D[r][lowest_index]
            del self.__pixels[r][lowest_index]
        self.__width -= 1
        self.__update_maps(seam_indices)

    def __update_maps(self, seam_indices):
        # Kx and Ky kernels
        kernelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

        seam_indices.reverse() # go from top to bottom of image

        # first row update
        c = seam_indices[0]
        # makes sure removed pixel wasn't furthest right column
        if c < len(self.__pixels[0]):
            oX = self.__get_new_pixel(kernelX, 0, c)
            oY = self.__get_new_pixel(kernelY, 0, c)
            energy = math.sqrt(oX**2 + oY**2)
            self.__cum_energy_map[0][c] = energy
            self.__energy_map[0][c] = energy

        # makes sure removed pixel wasn't furthest left column
        if c-1 >= 0:
            oX = self.__get_new_pixel(kernelX, 0, c-1)
            oY = self.__get_new_pixel(kernelY, 0, c-1)
            energy = math.sqrt(oX**2 + oY**2)
            self.__cum_energy_map[0][c-1] = energy
            self.__energy_map[0][c-1] = energy

        # update rest of rows
        for r in range(1, len(self.__pixels)):
            c = seam_indices[r]

            # makes sure removed pixel wasn't furthest right column
            if c < len(self.__pixels[0]):
                # get Ox and Oy values for pixels next to removed seam
                oX = self.__get_new_pixel(kernelX, r, c)
                oY = self.__get_new_pixel(kernelY, r, c)
                energy = math.sqrt(oX**2 + oY**2)

                # find adj pixel range
                low = max(0, c-1)
                high = min(len(self.__pixels_gray_2D[0]), c+2)

                # calc energy for pixel
                self.__cum_energy_map[r][c] = energy
                self.__energy_map[r][c] = energy

                min_energy = self.__cum_energy_map[r-1][low]
                # calc cummulative energy for pixel
                for i in range(low+1, high):
                    if self.__cum_energy_map[r-1][i] < min_energy:
                        min_energy = self.__cum_energy_map[r-1][i]
                self.__cum_energy_map[r][c] += min_energy

            # makes sure removed pixel wasn't furthest left column
            if c-1 >= 0:
                # get Ox and Oy values for pixels next to removed seam
                oX = self.__get_new_pixel(kernelX, r, c-1)
                oY = self.__get_new_pixel(kernelY, r, c-1)
                energy = math.sqrt(oX**2 + oY**2)

                # find adj pixel range
                low = max(0, c-2)
                high = min(len(self.__pixels_gray_2D[0]), c+1)

                # calc energy for pixel
                self.__cum_energy_map[r][c-1] = energy
                self.__energy_map[r][c-1] = energy

                min_energy = self.__cum_energy_map[r-1][low]
                # calc cummulative energy for pixel
                for i in range(low+1, high):
                    if self.__cum_energy_map[r-1][i] < min_energy:
                        min_energy = self.__cum_energy_map[r-1][i]
                self.__cum_energy_map[r][c-1] += min_energy

    def remove_vertical_seams(self, num):
        for _ in range(num):
            self.__remove_vertical_seam_and_update()

    def remove_horizontal_seams(self, num):
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

    # def __set_pixel(self, x, y, c):
    #     """
    #     Takes in 2D pixel location and sets the corresponding pixel in the image to c.
    #     """
    #     # makes sure input pixel is within range of image pixels
    #     # checks x
    #     if x < 0: 
    #         x = 0
    #     elif x >= self.__width:
    #         x = self.__width-1
    #     # checks y
    #     if y < 0:
    #         y = 0
    #     elif y >= self.__height:
    #         y = self.__height-1

    #     #self.__energy_map[x][y] = c

    def __correlate(self, pixels, kernel):
        """
        Compute the result of correlating the given image with the given kernel.

        Kernel Representation: List of Lists (each nested list contains one row of values, 
        total number of lists represents total number of rows in kernel)
        """
        new_pixels = list()
        # goes through each pixel and applies kernel to each one
        for x in range(self.__height):
            pixel_row = list()
            for y in range(self.__width):
                new_pixel = self.__get_new_pixel(kernel, x, y)   # calculates new pixel from original image
                pixel_row.append(new_pixel)        # sets pixel in result image
            new_pixels.append(pixel_row)

        return new_pixels

    def get_image(self):
        pixels_1D = list()
        for row in self.__pixels:
            pixels_1D += row
        image = {
            'height': self.__height,
            'width' : self.__width,
            'pixels': pixels_1D
        }
        return image


if __name__ == '__main__':
    carv = Carver('test_images/cats.png')
    carv.remove_vertical_seams(60)
    carv.save_image('result_images/cats60.png')

    