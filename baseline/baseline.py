import numpy as np
import csv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.misc import imresize
from PIL import Image
import PIL.ImageOps    

# On a high level, for any user input user_drawing, this program can search for 
# the K images of each class (number) that best match user_drawing. Currently, 
# the K best images within the actual class of user_drawing are averaged. 
# user_drawing and the average are displayed on screen.

CONST_SEARCH_SIZE = 10000 #number of mnist images to search 
K = 10 #number of best images within each class to average

# Preprocessing #
def print_pic(pixels):
    pixels = pixels.reshape((28, 28)) #reshape into 28x28 array
    plt.imshow(pixels, cmap='gray') #plot
    plt.show()

# Array of K best images for each class 0...9. Tuple contains image info, 
# consisting of (min_dist, min_index, best_pic) i.e. image's
# (euclidean distance from user input, index in MNIST, actual pixel info)  
bests = [[(float('inf'),-1,[]) for numKs in range(K)] for label in range(10)] 

num_in = str(input('Enter num: '))
# Import drawing and convert to grayscale
# Input must be in format ("0.png", "1.png", etc, due to line 87)
user_drawing = Image.open(num_in + '.png').convert('L') 
user_drawing = np.array(user_drawing, dtype='uint8')
# Scale image, then reshape to 1D array (consistent with MNIST images)
user_drawing = imresize(user_drawing, (28, 28)).reshape((1, 784)) 

# Invert image (black->white, white->black)
user_drawing = np.invert(user_drawing) 

with open('mnist_train.csv', 'r') as csv_file:
    count = 0
    for data in csv.reader(csv_file):
        # First column is the label
        label = int(data[0])
        # Other columns are pixels
        pixels = data[1:]

        # Make those columns into an array of 8-bits pixels
        # This array will be 1D with length 784
        # Pixel intensity values are integers from 0 to 255
        pixels = np.array(pixels, dtype='uint8')
        dist = np.linalg.norm(user_drawing - pixels)
        # Update bests array if necessary
        if(dist < bests[label][-1][0]):
            # Tuple containing relevant image info
            tup = (dist, count, pixels)
            for i in range(K):
                if dist < bests[label][i][0]: 
                    # Insert tuple into bests array
                    bests[label].insert(i, tup)
                    # Delete last (Kth smallest distance) tuple
                    del bests[label][-1]
                    break

        count += 1
        if(count % 2000 == 0):
            print("Searched %d of %d") %(count, CONST_SEARCH_SIZE) 
        if(count == CONST_SEARCH_SIZE):
            break

# Output user input
print_pic(user_drawing)

# Output average euclidean distance for each class
for i in range(10):
    print "Avg euclidean distance for %d is %f" \
    %(i, sum(a for a,b,c, in bests[i]) / K)

# Compute average picture for each class
avgs = [[] for i in range(10)]
for i in range(10):
    data = []
    for k in range(K):
        data.append(bests[i][k][2])
    avgs[i] = np.average(data, axis=0)

# Output best picture in correct class
print_pic(avgs[int(num_in)])

'''
# Print average pictures in all classes
for i in range(10):
    print_pic(avgs[i])
'''