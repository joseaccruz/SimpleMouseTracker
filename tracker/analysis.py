import cv2
import numpy as np
import math

from data import *
from utils import *


#======================================================================================================================
# Mouse detection functions
#======================================================================================================================
def detect( io, video_file, thresh, laser, poly, show_video, delay, jump, fade_margin=False ):

    # split the indivitual thresholds
    (thresh1, thresh2) = thresh

    # start video capture
    capture = cv2.VideoCapture( video_file )

    frame_count = capture.get( cv2.cv.CV_CAP_PROP_FRAME_COUNT )
    frame_height = int(capture.get( cv2.cv.CV_CAP_PROP_FRAME_HEIGHT ))
    frame_width  = int(capture.get( cv2.cv.CV_CAP_PROP_FRAME_WIDTH ))

    # start the  counter
    io.start_progress( frame_count )

    # create the mask
    poly = [np.array( poly, dtype=np.int32 )]
    mask = make_mask( poly, frame_height, frame_width )
    mask_not = make_mask_not( poly, frame_height, frame_width )

    if fade_margin: 
        margin = None
        p1_left_x_cut = min( poly[0][0][0], poly[0][3][0])
        p2_left_x_cut = max( poly[0][0][0], poly[0][3][0])
        p1_left_y_cut = min( poly[0][0][1], poly[0][3][1])
        p2_left_y_cut = max( poly[0][0][1], poly[0][3][1])

        p1_right_x_cut = min( poly[0][1][0], poly[0][2][0])
        p2_right_x_cut = max( poly[0][1][0], poly[0][2][0])
        p1_right_y_cut = min( poly[0][1][1], poly[0][2][1])
        p2_right_y_cut = max( poly[0][1][1], poly[0][2][1])

        p1_top_x_cut = min( poly[0][0][0], poly[0][1][0])
        p2_top_x_cut = max( poly[0][0][0], poly[0][1][0])
        p1_top_y_cut = min( poly[0][0][1], poly[0][1][1])
        p2_top_y_cut = max( poly[0][0][1], poly[0][1][1])

        p1_bottom_x_cut = min( poly[0][2][0], poly[0][3][0])
        p2_bottom_x_cut = max( poly[0][2][0], poly[0][3][0])
        p1_bottom_y_cut = min( poly[0][2][1], poly[0][3][1])
        p2_bottom_y_cut = max( poly[0][2][1], poly[0][3][1])
    
    # data list
    frame_data = FrameData()

    if show_video: io.show( "\nPress 'q' to terminate...\n" )
    

    # start the frame pointer
    f = jump
    if jump > 0:
        capture.set( cv2.cv.CV_CAP_PROP_POS_FRAMES, jump )

    # main loop
    while True:
        # get the frame from the video
        (ret, frame) = capture.read()
        if not ret:  break

        if fade_margin: 
            left_avg = np.average( frame[p1_left_y_cut+5:p2_left_y_cut-5, p1_left_x_cut+5:p2_left_x_cut+5] )
            cv2.line( frame, (poly[0][0][0]-10,poly[0][0][1]-10), (poly[0][3][0]-10,poly[0][3][1]+10), (left_avg,left_avg,left_avg), 20 )
            right_avg = np.average( frame[p1_right_y_cut+5:p2_right_y_cut-5, p1_right_x_cut-5:p2_right_x_cut-5] )
            cv2.line( frame, (poly[0][1][0]+10,poly[0][1][1]-10), (poly[0][2][0]+10,poly[0][2][1]+10), (right_avg,right_avg,right_avg), 20 )
            top_avg = np.average( frame[p1_top_y_cut+5:p2_top_y_cut+5, p1_top_x_cut+5:p2_top_x_cut-5] )
            cv2.line( frame, (poly[0][0][0]-10,poly[0][0][1]-10), (poly[0][1][0]+10,poly[0][1][1]-10), (top_avg,top_avg,top_avg), 20 )
            bottom_avg = np.average( frame[p1_bottom_y_cut-5:p2_bottom_y_cut-5, p1_bottom_x_cut+5:p2_bottom_x_cut-5] )
            cv2.line( frame, (poly[0][3][0]-10,poly[0][3][1]+10), (poly[0][2][0]+10,poly[0][2][1]+10), (bottom_avg,bottom_avg,bottom_avg), 20 )
        

        if (f % 25) == 0:
            io.show_progress( f )

        # extract the color information
        hsv = cv2.cvtColor( frame, cv2.COLOR_RGB2HSV )
        [hue, sat, img] = cv2.split( hsv )

        # play with the luminosity to get the black mouse
        data = _detect_aux( img, mask, mask_not, thresh1, thresh2 )
        (laser_on, laser_value) = detect_laser( img, laser )

        data.laser_on = laser_on

        frame_data.add( f, data )

        if show_video:
            if( not data.empty ):
                cv2.circle( frame, data.center, 2, (0, 0, 255), -1 )
                cv2.circle( frame, data.head,   2, (0, 255, 0), -1 )
                cv2.circle( frame, data.tail,   2, (0, 255, 0), -1)

            if laser_on:
                cv2.circle( frame, data.center,   10, (0, 0, 255), -1)

            cv2.polylines( frame, poly, True, (255, 0, 0), 1 )
            cv2.putText( frame, "%d" %f, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255) );
            cv2.imshow("Mouse Tracker", frame)

            key = cv2.waitKey( delay+33)

            if key == 113:
                break

          # increment frame pointer
        f += 1

    print "end"
    return frame_data

#
#
#
def _detect_aux( img, mask, mask_not, thresh1, thresh2 ):

    data = None
    #
    #  get a rough region around the mouse using the first threshold
    #
    (ret, img_bin) = cv2.threshold( img, thresh1, 255, cv2.THRESH_BINARY_INV );
    #img_bin = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 351, 50)            
        

    cv2.imshow("bin", img_bin)
    
    # get only the arena
    img_bin = np.bitwise_and( img_bin, mask )

    # clean up noise and enlarge the interest region
    img_bin = cv2.erode ( img_bin, kernel=cv2.getStructuringElement( cv2.MORPH_RECT, (3,3) ), iterations=2 )
    img_bin = cv2.dilate( img_bin, kernel=cv2.getStructuringElement( cv2.MORPH_RECT, (4,4) ), iterations=10 )

    # get the blobs
    (blobs, dummy) = cv2.findContours( np.array( img_bin ), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )

    # if found
    if len(blobs) > 0:
        # get only the biggest blob
        blob = sorted( blobs, key=lambda x: -len(x) )[0]

        # get only the square around the blob to speed up the process
        minx = np.min(blob[:,0,0])
        maxx = np.max(blob[:,0,0])
        miny = np.min(blob[:,0,1])
        maxy = np.max(blob[:,0,1])

        small_img = img[miny:maxy, minx:maxx]

        # use a stricter threshold
        (ret, small_img_bin) = cv2.threshold( small_img, thresh2, 255, cv2.THRESH_BINARY_INV )
        #small_img_bin = cv2.adaptiveThreshold(small_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 351, 50)            
     

        small_mask = mask[miny:maxy, minx:maxx]
        small_img_bin = np.bitwise_and( small_img_bin, small_mask )

        # get all the blobs bigger than N pixels (should be at most 2 because of the wire)
        (blobs, dummy) = cv2.findContours( np.array( small_img_bin ), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
        blobs = filter( lambda x: cv2.contourArea(x) > 10, blobs )

        if len(blobs) > 0:
            blob = np.concatenate( blobs )

            moments = cv2.moments( small_img_bin )
            area = cv2.contourArea(blob)

            # TODO: COMMENT THIS CONSTANTS
            # Only process objects with an area bigger than this
            if area > 500:
                centroid = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))

                # gets the most distant point to the centroid (can be the nose or the tail, who knows? we assume the nose)
                dists = map( lambda p: lin_dist( p[0], centroid ), blob )
                ndx = dists.index(max(dists))
                head = tuple(blob[ndx][0]+[minx, miny])

                # gets the point of the blob most distant to the last point detected
                dists = map( lambda p: lin_dist( p[0], blob[ndx][0] ), blob )
                ndx = dists.index(max(dists))
                tail = tuple(blob[ndx][0]+[minx, miny])

                centroid = (centroid[0] + minx, centroid[1] + miny)

                data = FrameDataItem( centroid, head, tail )

    if data is None:
        data = FrameDataItem()

    return data

def detect_laser( img, laser ):
    (x, y, side, thresh, greater) = laser

    value = np.mean( img[y-side:y+side, x-side:x+side] )
    return (greater and value > thresh) or (not greater and value < thresh), value


#======================================================================================================================
# Fixing the data
#======================================================================================================================

def fix_data( io, frame_data, poly=None ):
    io.show( "\nFixing the data...\n" )
    _fix_aux( io, frame_data.get_list(), poly )

def _fix_aux_new( io, data_list, poly ):
    (head_last, tail_last) = (None, None)

    f = 0
    while f < len(data_list):
        dt = data_list[f]

        if not dt.empty:
            head = np.array(dt.head)
            tail = np.array(dt.tail)

            if (not head_last is None):
                print min(lin_dist(head, head_last), lin_dist(head, tail_last))
                if lin_dist(head, head_last) > lin_dist(head, tail_last):
                    (dt.head, dt.tail) = (dt.tail, dt.head)

            head_last = np.array(dt.head)
            tail_last = np.array(dt.tail)

        f += 1



def _fix_aux( io, data_list, poly ):
    # start the  counter
    io.start_progress( len(data_list) )

    windows = []
    curr_window = None

    # go through all frames and swaps heads/tails in order to make them compatible
    f = 0
    while f < len(data_list):
        if (f % 500) == 0:
            io.show_progress( f )

        dt = data_list[f]

        # skip empty frames
        if not dt.empty:
            head = np.array(dt.head)
            tail = np.array(dt.tail)

            # get the two possible vector (head->tail) or (tail->head)
            v1 = norm_vector( head - tail )
            v2 = norm_vector( tail - head )

            # if the current window was not initialized
            if curr_window is None:
                # arbitrarly assume 'v1' as the reference vector
                curr_window = FixWindow( v1, head, tail, f )
                windows.append( curr_window )

            # if we already have a window
            else:
                # computes the alternative angles for each of the possible directions
                theta1 = np.degrees( np.arccos( nice_dot( v1, curr_window.ref_vector ) ) )
                theta2 = np.degrees( np.arccos( nice_dot( v2, curr_window.ref_vector ) ) )

                # if we have a small angle we just need to check the best one and add it to the window
                if min(theta1, theta2) < 40:
                    # updates the reference vector
                    curr_window.end = f
                    if theta1 < theta2:
                        curr_window.ref_vector = v1
                        curr_window.ref_head = head
                        curr_window.ref_tail = tail
                    else:
                        # swap heads/tails
                        curr_window.ref_vector = v2
                        curr_window.ref_head = tail
                        (dt.head, dt.tail) = (tail.tolist(), head.tolist())
                        (v1, v2) = (v2, v1)

                else:
                    # let's gess the new head position
                    dist1 = lin_dist( np.array( curr_window.ref_head ), np.array( dt.head ) ) + lin_dist( np.array( curr_window.ref_tail ), np.array( dt.tail ) )
                    dist2 = lin_dist( np.array( curr_window.ref_head ), np.array( dt.tail ) ) + lin_dist( np.array( curr_window.ref_tail ), np.array( dt.head ) )

                    if dist1 > dist2:
                        (dt.head, dt.tail) = (tail.tolist(), head.tolist())
                        (v1, v2) = (v2, v1)

                    # go back to this frame in another window (we dont' increment 'f' here!)
                    curr_window = None
                    continue

            # checks if the head-tail points to the speed vector direction
            speed_ok = 0
            if (f > 10) and (not data_list[f-10].empty):
                speed_vector = norm_vector( np.array( dt.center ) - np.array( data_list[f-10].center ) )
                speed_vector_len = lin_dist( np.array( dt.center ), np.array( data_list[f-10].center ) )

                #speed_theta1 = np.degrees( np.arccos( nice_dot( v1, speed_vector ) ) )
                #speed_theta2 = np.degrees( np.arccos( nice_dot( v2, speed_vector ) ) )

                speed_proj_1 = nice_dot( v1, speed_vector ) * speed_vector_len
                #speed_proj_2 = nice_dot( v2, speed_vector ) * speed_vector_len
                speed_proj_2 = -speed_proj_1

                if speed_proj_1 > 0:
                    curr_window.speed_pos += speed_proj_1
                    curr_window.speed_ok += speed_proj_1
                else:
                    curr_window.speed_neg += speed_proj_1
                    curr_window.speed_ok += speed_proj_1

        # next frame
        f += 1

    print "=== WINDOWS BEFORE MERGING ==="
    for w in windows:
        print w

    # swap full windows based on speed
    for w in filter( lambda w: w.speed_ok < 0, windows ):
        w.swap( data_list )

    # merge the small noisy windows
    wbig = None
    for (i, wnow) in enumerate(windows):
        # this is a big window keep it (don't touch it)
        if wnow.is_big():
            wbig = wnow

        # this is a contiguous small window
        elif (not wbig is None) and ((wbig.end+1) == wnow.start):
            dt1 = data_list[wbig.end]
            dt2 = data_list[wnow.start]

            dist1 = lin_dist( np.array( dt1.head ), np.array( dt2.head ) ) + lin_dist( np.array( dt1.tail ), np.array( dt2.tail ) )
            dist2 = lin_dist( np.array( dt1.head ), np.array( dt2.tail ) ) + lin_dist( np.array( dt1.tail ), np.array( dt2.head ) )

            if dist1 > dist2:
                wnow.swap( data_list )

            # update the big window
            wbig.end = wnow.end
            wnow.empty = True

    print "=== WINDOWS AFTER MERGING ==="
    for w in windows:
        if not w.empty:
            print w

    # warn about problematic windows
    for i in xrange(len(windows)-1):
        if (not windows[i].empty) and (windows[i].end + 1) == windows[i+1].start:
            dt1 = data_list[windows[i].end]
            dt2 = data_list[windows[i+1].start]

            dist1 = lin_dist( np.array( dt1.head ), np.array( dt2.head ) ) + lin_dist( np.array( dt1.tail ), np.array( dt2.tail ) )
            dist2 = lin_dist( np.array( dt1.head ), np.array( dt2.tail ) ) + lin_dist( np.array( dt1.tail ), np.array( dt2.head ) )

            if( dist1 > dist2 ):
                print "WARNING: %d > %d: %s\t%s" %(dist1, dist2, windows[i], windows[i+1])

class FixWindow(object):
    def __init__( self, ref_vector, ref_head, ref_tail, frame ):
        self.ref_vector = ref_vector
        self.ref_head = ref_head
        self.ref_tail = ref_tail
        self.start = frame
        self.end = frame
        self.speed_ok = 0
        self.speed_pos = 0
        self.speed_neg = 0
        self.empty = False

    def __str__( self ):
        if self.empty:
            return "%6d - %6d (MERGED)" %(self.start, self.end)
        else:
            return "%6d - %6d (len == %d, ok == %5.2f, pos == %5.2f, neg == %5.2f)\t%s\t%s" %(self.start, self.end, self.end-self.start, self.speed_ok, self.speed_pos, self.speed_neg, str(self.ref_vector), str(self.ref_head))

    def length( self ):
        return self.end-self.start

    def is_big(self):
        w_ave = abs(self.speed_ok / (self.length()+1))

        return (self.length() > 100) and (w_ave > 1.0)

    def swap( self, data_list ):
        for f in xrange(self.start, self.end+1):
            dt = data_list[f]

            if not dt.empty:
                (dt.head, dt.tail) = (dt.tail, dt.head)



#======================================================================================================================
# Calculating backbone
#======================================================================================================================

def __draw_dots( img, data ):
    if (not data is None) and (not data.empty):
        cv2.circle( img, data.head, 3, (0, 0, 255), -1 )
        cv2.circle( img, data.tail,   3, (255, 0, 0), -1 )

        if data.laser_on:
            cv2.circle( img, data.center, 10, (0, 0, 255), -1 )
        else:
            cv2.circle( img, data.center,   3, (0, 255, 0), -1 )

def __cut_roi(img, data):
    if (not data is None) and (not data.empty):
        p1 = max(data.center[0]-150, 0), max(data.center[1]-150, 0)
        p2 = data.center[0]+150, data.center[1]+150
        rat = img[p1[1]:p2[1], p1[0]:p2[0]]
        return rat, p1
    else:
        return None, None


def __rotate_image(img, data, center=None):
    if (not data is None) and (not data.empty):
        rad = math.atan2(data.head[1]-data.tail[1], data.head[0]-data.tail[0])
        deg = math.degrees(rad)
        return rotate_image(img, deg+90, center)
    else:
        return None


def __calcbackbone(mask, n_points):
    mask_height = mask.shape[0]
    step =  int(math.ceil(float(mask_height)/float(n_points)))
    backbone = []
    for i in range(0, mask_height, step):
        y_cut = i
        if (y_cut+step)>=mask_height: y_cut = mask_height-step
        conts, dummy = cv2.findContours(mask[y_cut:y_cut+step].copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
        big = biggestContour(conts)
        if big!=None and len(big)>=5:
            big = big + np.array([0, y_cut])
            center, axes, angle = cv2.fitEllipse(big)
            center =  int(center[0]) , int(center[1])
            backbone.append( center )
    return backbone

def calculate_backbone( io, frame_data, video_file, poly, thresh, show_video, delay, jump, backbone_n_points):

    # split the indivitual thresholds
    (thresh1, thresh2) = thresh

    # start video capture
    capture = cv2.VideoCapture( video_file )

    frame_count = capture.get( cv2.cv.CV_CAP_PROP_FRAME_COUNT )
    frame_height = int(capture.get( cv2.cv.CV_CAP_PROP_FRAME_HEIGHT ))
    frame_width  = int(capture.get( cv2.cv.CV_CAP_PROP_FRAME_WIDTH ))

    # start the  counter
    io.start_progress( frame_count )

    # create the mask
    poly = [np.array( poly, dtype=np.int32 )]
    mask = make_mask( poly, frame_height, frame_width )
    mask_not = make_mask_not( poly, frame_height, frame_width )

    if show_video: io.show( "\nPress 'q' to terminate...\n" )
    
    data_list = frame_data.get_list()

    # start the frame pointer
    f = jump
    if jump > 0: capture.set( cv2.cv.CV_CAP_PROP_POS_FRAMES, jump )


    # main loop
    while True:
        f = int(capture.get( cv2.cv.CV_CAP_PROP_POS_FRAMES ))
        # get the frame from the video
        ret, frame = capture.read()
        if not ret: break
        if (f % 25) == 0: io.show_progress( f )
        data = frame_data.get( f )
        data_list[f]._n_backbone_points = backbone_n_points
        data_list[f].backbone = None

        # extract the color information
        hsv = cv2.cvtColor( frame, cv2.COLOR_RGB2HSV )
        [hue, sat, img] = cv2.split( hsv )
        img = cv2.bitwise_and(mask, img)
        img += mask_not
        img, first_cut_top_point = __cut_roi( img, data )

        if img!=None:

            (ret, img_bin) = cv2.threshold( img, thresh2, 255, cv2.THRESH_BINARY_INV )
            (blobs, dummy) = cv2.findContours( img_bin.copy() , cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
            blob = np.concatenate( blobs )
            rect = cv2.boundingRect(blob)
            second_cut_top_point = rect[0], rect[1]
            second_cut_bottom_point = rect[0]+rect[2], rect[1]+rect[3]
            small_img_bin = img_bin[ second_cut_top_point[1]:second_cut_bottom_point[1], second_cut_top_point[0]:second_cut_bottom_point[0] ]
            
            center =    data.center[0] - first_cut_top_point[0]  - second_cut_top_point[0],   data.center[1]    - first_cut_top_point[1]     - second_cut_top_point[1]
            head =      data.head[0]   - first_cut_top_point[0]  - second_cut_top_point[0],   data.head[1]      - first_cut_top_point[1]     - second_cut_top_point[1]
            tail =      data.tail[0]   - first_cut_top_point[0]  - second_cut_top_point[0],   data.tail[1]      - first_cut_top_point[1]     - second_cut_top_point[1]
            
            center_store = np.zeros_like( small_img_bin )
            head_store = np.zeros_like( small_img_bin )
            tail_store = np.zeros_like( small_img_bin )
            cv2.circle( center_store, center, 5, 255, -1 )
            cv2.circle( head_store, head, 5, 255, -1 )
            cv2.circle( tail_store, tail, 5, 255, -1 )

            small_img_bin_rot = __rotate_image( small_img_bin, data )
            center_store_rot = __rotate_image( center_store, data )
            head_store_rot = __rotate_image( head_store, data )
            tail_store_rot = __rotate_image( tail_store, data )

            center_contour = getBiggestContour( center_store_rot )
            head_contour = getBiggestContour( head_store_rot )
            tail_contour = getBiggestContour( tail_store_rot )

            center = calcCountourCentroid(center_contour)
            head = calcCountourCentroid(head_contour)
            tail = calcCountourCentroid(tail_contour)

            (blobs, dummy) = cv2.findContours( small_img_bin_rot.copy() , cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
            blob = np.concatenate( blobs )
            rect = cv2.boundingRect(blob)
            cutP1 = rect[0], rect[1]
            cutP2 = rect[0]+rect[2], rect[1]+rect[3]

            small_img_bin_rot = small_img_bin_rot[head[1]:tail[1],cutP1[0]:cutP2[0]]
            backbone = __calcbackbone(small_img_bin_rot, backbone_n_points)

            center = center[0]-cutP1[0], center[1]-head[1]
            tail = tail[0]-cutP1[0], tail[1]-head[1]
            head = head[0]-cutP1[0], head[1]-head[1]

            backbone_tmp = [head] + backbone + [tail]
            if len(backbone_tmp)<(backbone_n_points+2): 
                for i in range( (backbone_n_points+2)-len(backbone_tmp) ): backbone_tmp += [ (0,0) ]
            data_list[f].backbone = backbone_tmp
            
            if show_video:               

                rat_img = cv2.merge( (small_img_bin_rot,small_img_bin_rot,small_img_bin_rot) )
                for p in backbone: cv2.circle( rat_img, p, 2, (0, 0, 255), -1 )
                cv2.circle( rat_img, head, 4, (0, 255, 0), -1 )
                cv2.circle( rat_img, center, 4, (0, 255, 0), -1 )
                cv2.circle( rat_img, tail, 4, (0, 255, 0), -1 )
            
                cv2.imshow("Rat", rat_img)
        

        if show_video:    
            __draw_dots( frame, data)
            cv2.imshow("Mouse Tracker", frame)
            key = cv2.waitKey(delay)
            if key == 113: break

        