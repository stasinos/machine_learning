import numpy as np
from PIL import Image


def measure( array1, array2 ):
    # most of the figure is white, so simply counting matches
    # would give too high a score. Instead, count the fraction
    # points of each image "found" by the other image, and
    # combine the two recall-like  metrics in an f score-like
    # manner in order to penalize painting everything black.
    points_in_both = np.logical_and( array1, array2 )
    a = np.count_nonzero( np.logical_and(array1,points_in_both) )
    q1 = float(a) / float( np.count_nonzero(array1) )
    q2 = float(a) / float( np.count_nonzero(array2) )
    return 2*q1*q2/(q1+q2)

def evaluate( pic1, pic2 ):
    # White is 255, so becomes True when converting to B/W.
    # Apply logical-not to make the points of the graph True
    # and the backgrounf False.
    array1 = np.logical_not( np.array(pic1) )
    array2 = np.logical_not( np.array(pic2) )
    return measure( array1, array2 )

images = ["poly","natlog","exp","sine","sinc","cosh","sinconlinear","sineonln"]
bwimages = []

test_images = ["natlog","sineonln"]

for filename in images:
    pic = Image.open( "{}-bw.png".format(filename) )
    bwimages.append( pic )

for filename in test_images:
    pic = Image.open("{}.png".format(filename)).convert('1')
    for i,a in enumerate(bwimages):
        q = int( 100*evaluate(pic,a) )
        print("{} and {} have {}% match".format(filename,images[i],q))
    print("")
