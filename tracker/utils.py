import os
import shutil
import sys
import time
import math
import cv2
import numpy as np



# ==================================================================================================
# input / output functions and classes
# ==================================================================================================


def show( msg ):
    sys.stderr.write( "%s" %msg )
    sys.stderr.flush()

def error( msg, fatal=True ):
    if fatal:
        sys.stderr.write( "\n!! ERROR: %s\n" %msg )
        #raise Exception()
        quit()
    else:
        sys.stderr.write( "\n!! WARNING: %s\n" %msg )
"""
class TimeCount(object):
    def __init__( self, total ):
        self._total = total
        self._start = time.time()

    def show( self, count ):
        if count == 0:
            count += 1

        deltat = time.time() - self._start
        stept = deltat / float(count)
        finalt = float(self._total-count) * stept

        back = "\b" * 75

        deltats = time.strftime( "%H:%M:%S", time.gmtime(deltat) )
        finalts = time.strftime( "%H:%M:%S", time.gmtime(finalt) )

        show( "%sFrame: %5d/%5d (%5.2f%%), Ellap: %s, Expec: %s" %(back, count, self._total, (float(count) / float(self._total)) * 100.0, deltats, finalts) )
"""


# ==================================================================================================
# sanity check functions
# ==================================================================================================

def check_dir( io, dname, txt ):
    if not os.path.isdir( dname ):
        io.error( "%s not found: '%s'" %(txt, dname) )

def check_file( io, fname, txt ):
    if not os.path.isfile( fname ):
        io.error( "%s not found: '%s'" %(txt, fname) )

# ==================================================================================================
# file input / output functions
# ==================================================================================================

def write_data( data, fname, sep="\t", backup=True ):
    if backup:
        backup_file( fname )

    fo = open( fname, "w" )
    fo.write( "\n".join( map( lambda d: sep.join( map( str, d ) ), data ) ) )
    fo.close()

def read_data( fname, sep="\t" ):
    return map( lambda s: map( float, s.strip().split( sep ) ), open( fname ).read().strip().split( "\n" ) )

def backup_file( fname ):
    if os.path.isfile( fname ):
        i = 1
        while os.path.isfile( "%s.%d.bak" %(fname, i) ):
            i += 1
        shutil.copyfile( fname, "%s.%d.bak" %(fname, i) )



# ==================================================================================================
# linear algebra methods
# ==================================================================================================

def lin_dist( p1, p2 ):
    return np.linalg.norm( (p1[0]-p2[0], p1[1]-p2[1]) )

def norm_vector( v1 ):
    nv1 = np.linalg.norm( v1 )
    if( nv1 == 0 ):
        return np.array( (0,0) )
    else:
        return v1 / nv1

def nice_dot( u, v ):
    return max( min( np.dot( u, v ), 1.0 ), -1.0 ) 



# ==================================================================================================
# image utils
# ==================================================================================================

def make_mask( poly, height, width ):
    mask = np.zeros( (height, width), dtype=np.uint8 )

    cv2.fillPoly( mask, poly, 255 )

    return mask

def make_mask_not( poly, height, width ):
    mask = np.ones( (height, width), dtype=np.uint8 ) * 255

    cv2.fillPoly( mask, poly, 0 )

    return mask



# ==================================================================================================
# computatinal geometry methods
# ==================================================================================================

"""
Determine if a point is inside a given polygon or not Polygon is a list of (x,y) pairs. This
function returns True or False. The algorithm is called "Ray Casting Method".
"""
def point_in_poly( poly, point ):
    (x, y) = (point[0], point[1])
    
    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside








def getTranslationMatrix2d(dx, dy):
    """
    Returns a numpy affine transformation matrix for a 2D translation of
    (dx, dy)
    """
    return np.matrix([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

def rotate_image(image, angle, centroid=None):
    """
    Rotates the given image about it's centre
    """

    image_size = (image.shape[1], image.shape[0])
    if centroid==None: image_center = tuple(np.array(image_size) / 2)
    else: image_center = centroid

    rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])
    trans_mat = np.identity(3)


    w2 = image_size[0] * 0.5
    h2 = image_size[1] * 0.5

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    tl = (np.array([-w2, h2]) * rot_mat_notranslate).A[0]
    tr = (np.array([w2, h2]) * rot_mat_notranslate).A[0]
    bl = (np.array([-w2, -h2]) * rot_mat_notranslate).A[0]
    br = (np.array([w2, -h2]) * rot_mat_notranslate).A[0]

    x_coords = [pt[0] for pt in [tl, tr, bl, br]]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in [tl, tr, bl, br]]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))
    new_image_size = (new_w, new_h)

    new_midx = new_w * 0.5
    new_midy = new_h * 0.5

    dx = int(new_midx - w2)
    dy = int(new_midy - h2)

    trans_mat = getTranslationMatrix2d(dx, dy)
    affine_mat = (np.matrix(trans_mat) *np.matrix(rot_mat))[0:2, :]
    result = cv2.warpAffine(image, affine_mat, new_image_size, flags=cv2.INTER_LINEAR)

    return result


def biggestContour(contours, howmany=1):
    biggest = []
    for blob in contours:
        area = cv2.contourArea(blob)
        biggest.append( (area, blob) )
    if len(biggest)==0: return None
    biggest = sorted( biggest, key=lambda x: -x[0])
    if howmany==1: return biggest[0][1]
    return [x[1] for x in biggest[:howmany] ]

def getBiggestContour(image, howmany=1):
    (blobs, dummy) = cv2.findContours( image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
    return biggestContour(blobs, howmany)

def calcCountourCentroid(contour):
    moments = cv2.moments( contour )
    return int(math.ceil(moments['m10']/moments['m00'])), int(math.ceil(moments['m01']/moments['m00'])) 


def fit2BiggestContour(image):
    contour = getBiggestContour(image)
    x,y, w, h = cv2.boundingRect(contour)
    mask = np.zeros( (h,w), dtype=np.uint8 )
    contour = [ [[p[0][0]-x,p[0][1]-y]] for p in contour ]
    cv2.fillPoly( mask, np.array([contour]), 255 )
    return mask, (x,y)