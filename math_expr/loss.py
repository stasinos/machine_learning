import numpy as np
from PIL import Image


def measure( array1, array2 ):
    # most of the figure is white, so simply counting
    # matches always gives too high a score.
    # Count the percentage of points of each image
    # "found" by the other image, and combine in an
    # f score-like manner.
    points_in_both = np.logical_and( array1, array2 )
    a = np.count_nonzero( np.logical_and(array1,points_in_both) )
    b = np.count_nonzero( array1 )
    c = np.count_nonzero( array2 )
    q1 = float(a) / float(b)
    q2 = float(a) / float(c)
    return 2*q1*q2/(q1+q2)

images = ["poly","natlog","exp","sine","sinc","cosh","sinconlinear","sineonln"]
bwimages = []

test_images = ["natlog","sineonln"]

for filename in images:
    pic = Image.open("{}.png".format(filename)).convert('1')
    # white is 255, so becomes True
    # apply NOT to make points in graph True bg False
    pic_array = np.logical_not( np.array(pic) )
    bwimages.append( pic_array )

for filename in test_images:
    pic = Image.open("{}.png".format(filename)).convert('1')
    pic_array = np.logical_not( np.array(pic) )
    for i,a in enumerate(bwimages):
        q = int( 100*measure(pic_array,a) )
        print("{} and {} have {}% match".format(filename,images[i],q))
    print("")
