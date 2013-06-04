import os
import shutil
import sys
import time

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