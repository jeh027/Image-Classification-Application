import numpy as np
from PIL import Image


# RGB Image #
class RGBImage:
    """
    The class RGBImage contains a constructor and many class
    functions as a template for image objects in RGB color spaces
    """

    def __init__(self, pixels):
        """
        This constructor takes in two arguments: self and pixels.
        The function checks if there are any exceptions in the input
        and initializes pixels to a 3D list, num_row to the amount of 
        rows in the list and num_cols to the number of columns in the 
        list
        """

        if type(pixels) != list or len(pixels) == 0:
            raise TypeError()

        for row in pixels:
            for col in row:
                if type(row) != list or len(row) == 0:
                    raise TypeError()
                if len(row) != len(pixels[0]):
                    raise TypeError()
                if type(col) != list or len(col) != 3:
                    raise TypeError()

        for row in pixels:
            for col in row:
                for num in col:
                    if num < 0 or num > 255 or type(num) != int:
                        raise ValueError()

        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        """
        This function is a getter method, taking in the argument: self.
        The function uses self to return a tuple containing the number 
        of rows and number of columns.
        """
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        This function takes in one argument: self and uses self to
        get the corresponding pixels list. The function then returns a 
        deep copy of the list.
        """
        return [[list(inlist) for inlist in lst] for lst in self.pixels]


    def copy(self):
        """
        This function takes in one argument: self and returns a deep
        copy using the helper function get_pixels above.
        """
        return RGBImage.get_pixels(self)

    def get_pixel(self, row, col):
        """
        This function takes in three arguments: self, row and col. The 
        function returns a tuple containing three color values representing
        a pixel at the position row, col. If the input is invalid, the 
        function will raise an error.
        """
        
        if type(row) != int or type(col) != int:
            raise TypeError()

        try:
            return tuple(self.pixels[row][col])
        except IndexError:
            raise ValueError()


    def set_pixel(self, row, col, new_color):
        """
        This function takes in four arguments: self, row, col, and new
        color. The function updates the color of the pixel at the postion
        row, col. If there is an invalid input, the function will raise
        a TypeError() or ValueError(). This function returns None.
        """

        if type(row) != int or type(col) != int:
            raise TypeError()
        if new_color[0] > 255 or new_color[1] > 255 or new_color[2] > 255:
            raise ValueError()
        if type(new_color) != tuple or len(new_color) != 3:
            raise TypeError()
        if all(map(lambda num: type(num) == int, new_color)) == False:
            raise TypeError()

        for i in range(len(new_color)):
            if new_color[i] >= 0:
                self.pixels[row][col][i] = new_color[i]


# Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    The class RGBImage contains a constructor and many class
    functions as a template for altering pixel matrices and 
    calculating incurred costs by customers. 
    """

    def __init__(self):
        """
        This constructor initializes the instance cost to zero by 
        default. The variable cost tracks the total incurred cost 
        of self. 
        """
        self.cost = 0

    def get_cost(self):
        """
        This function is a getter method that returns the current
        total incurred cost of self. 
        """
        return self.cost

    def negate(self, image):
        """
        This function takes in two arguments: self and image. The function
        inverts the image given by the argument by subtracting each pixel
        value by 255.
        """
        return RGBImage([[[255 - value for value in lst2] for lst2 in lst1] \
            for lst1 in image.pixels])


    def grayscale(self, image):
        """
        This function takes in two arguments: self and image. The function
        converts the image given in the argument into grayscale by taking the
        average of the pixels matrix.
        """
        return RGBImage([[[sum(lst2)//3 for value in lst2] for lst2 in lst1] \
            for lst1 in image.pixels])


    def rotate_180(self, image):
        """
        This function takes in two arguments: self and image. The function
        rotates the image given in the argument by 180 degrees. 
        """
        r1 = image.get_pixels()[::-1]
        return RGBImage([col[::-1] for col in r1])


# Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    The class creates a money version of the template class, the class
    utilizes variables such as cost and coupouns to keep track of cost
    or use coupouns. Everytime a method has been called or used it 
    increments the cost variable.
    """

    def __init__(self):
        """
        This constructor initializes a cost instance to 0 by default 
        which will track the total cost incurred by the user.
        """
        self.cost = 0
        self.coupons = 0
        self.rotations = 0

    def negate(self, image):
        """
        This function takes in two arguments: self and image. Whenever 
        this function is called, the function will add $5 dollars to the 
        instance cost and return the inverted image using the class:
        ImageProcessingTemplate.
        """
        self.cost += 5
        return ImageProcessingTemplate.negate(self, image)

    def grayscale(self, image):
        """
        This function takes in two arguments: self and image. Whenever 
        this function is called, the function will add 6 dollars to the 
        instance cost and return the grayscaled image using the class:
        ImageProcessingTemplate.
        """
        self.cost += 6
        return ImageProcessingTemplate.grayscale(self, image)


    def rotate_180(self, image):
        """
        This function takes in an input of the image and rotates the image to 
        user's liking and increments the cost by 10 everytime ran. This 
        function uses inheritance.
        """
        self.rotations += 1

        if self.rotations % 2 != 0:
            if self.coupons > 0:
                self.coupons -= 1
                return super().rotate_180(image)
            else:
                self.cost += 10
                return super().rotate_180(image)
        elif self.rotations % 2 == 0:
            if self.coupons > 0:
                self.coupons -= 1
            else:
                self.cost -= 10


    def redeem_coupon(self, amount):
        """
        The function takes in one input and reduces the cost
        by the amount of times the method is called (tracks how many
        times the method is called).
        """
        if amount <= 0:
            raise ValueError()
        if type(amount) != int:
            raise TypeError()

        self.coupons += amount


# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    This class has two methods of chroma_ley and sticker
    and is the premium version of the app, where it automatically
    increses the cost of the app to be $50 dollars.
    """

    def __init__(self):
        """
        Constructor of the class that intializes the variable
        cost to be $50.
        """
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        This function takes in 3 inputs and changes the chroma background
        of the image. It checks if the types and instances of the inputs
        are valid or not by raising errors, if they are invalid. 
        """

        if not isinstance(chroma_image, RGBImage) or not \
        isinstance(background_image, RGBImage):
            raise TypeError()

        if len(chroma_image.pixels) != len(background_image.pixels):
            raise ValueError()

        chroma = list(chroma_image.pixels)

        for lst in chroma_image.pixels:
            row = chroma_image.pixels.index(lst)
            for pix in lst:
                if pix == list(color):
                    col = lst.index(pix)
                    chroma[row][col] = background_image.pixels[row][col]

        return RGBImage(chroma)


    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        The function takes in 4 inputs and checks if each input is valid, by
        raising errors if the input is invalid, it then places a sticker onto
        the image by creating a new image.
        """
        sticker_rows = len(sticker_image.pixels)
        sticker_cols = len(sticker_image.pixels[0])
        background_rows = len(background_image.pixels)
        background_cols = len(background_image.pixels[0])

        if not isinstance(sticker_image, RGBImage) or not \
        isinstance(background_image, RGBImage):
            raise TypeError()

        if type(x_pos) != int or type(y_pos) != int:
            raise ValueError()

        if sticker_rows >= background_rows or sticker_cols >= \
        background_cols:
            raise ValueError()

        if background_rows < y_pos + sticker_rows or background_cols < \
        x_pos + sticker_cols:
            raise ValueError()

        background = list(background_image.pixels)

        for row in range(y_pos, y_pos + sticker_rows):
            for col in range(x_pos, x_pos + sticker_cols):
                background[row][col] = sticker_image.pixels \
                [row-y_pos][col-x_pos]

        return RGBImage(background)


# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    The class implements machine learning type features
    by checking image data to see how freqeuent and revelant it is
    """

    def __init__(self, n_neighbors):
        """
        The function intializes variable n_neigbhors
        """
        self.n_neighbors = n_neighbors

    def fit(self, data):
        """
        The function takes in an input of data and sets 
        the value of data into self.data
        """
        if len(data) <= self.n_neighbors:
            raise ValueError()
        if self.data:
            raise ValueError()

        self.data = data

    @staticmethod
    def distance(image1, image2):
        """
        The function takes in two inputs and caculates
        the Euclidean distance of both inputs, and checks
        if the both inputs are RBGImage by raising errors if they are
        not
        """ 
        if not isinstance(image1, RGBImage) or not isinstance(image2, \
        RGBImage):
            raise TypeError()

        if len(image1.pixels) != len(image2.pixels):
            raise ValueError()

        p1 = image1.get_pixels()
        p2 = image2.get_pixels()

        return sum([(p1[row][col][chan] - p2[row][col][chan])**2 \
            for row in range(len(p1)) \
            for col in range(len(p1[row])) \
            for chan in range(len(p1[row][col]))]) ** (1/2)


    @staticmethod
    def vote(candidates):
        """
        The function finds the most viewed or popular label and 
        returns it. In case of a tie, any one of them is returned.
        """
        counter = 0
        popular = candidates[0]

        for candidate in candidates:
            frequency = candidates.count(candidate)
            if frequency > counter:
                counter = frequency
                popular = candidate

        return popular

    def predict(self, image):
        """
        The function gives a guess using the vote method 
        for the neighbors
        """
        if not self.data:
            raise ValueError

        distance = [(ImageKNNClassifier.distance \
        (image, tup[0]), tup[1]) for tup in self.data]

        sort = sorted(distance, key=lambda x: x[0], \
        reverse=False)[:self.n_neighbors]

        cand_list = [tup[1] for tup in sort]

        return ImageKNNClassifier.vote(candidate_list)


def img_read_helper(path):
    img = Image.open(path).convert("RGB")
    matrix = np.array(img).tolist()
    return RGBImage(matrix)


def img_save_helper(path, image):
    img_array = np.array(image.get_pixels())
    img = Image.fromarray(img_array.astype(np.uint8))
    img.save(path)


def create_random_pixels(low, high, nrows, ncols):
    return np.random.randint(low, high + 1, (nrows, ncols, 3)).tolist()