#!/usr/bin/env python

import argparse
import os
import sys
import time

import cv2
import numpy as np

from tracker.command import *
from tracker.data import *
from tracker.utils import *

#======================================================================================================================
# Util functions
#======================================================================================================================

def change_frame( f_new ):
    global capture

    capture.set( cv2.cv.CV_CAP_PROP_POS_FRAMES, f_new )

#======================================================================================================================
# Call back function
#======================================================================================================================

def on_mouse_main( e, mx, my, lb, rb ):
    global fcount
    global fwidth

    # if LEFT BUTTON IS PRESSED CHANGE THE FRAME POSITION
    if lb == 1:
        xs = 10
        xe = fwidth - (2 * xs)
        f_new = (mx-xs) / float(xe-xs) * fcount

        if f_new >= 0 and f_new < fcount:
            change_frame( f_new )


#======================================================================================================================
# User interaction functions
#======================================================================================================================

CIRCLE_RADIUS = 100
TIME_BEFORE_WINDOW = 100

KEY_ESC = 27
KEY_C = 99
KEY_D = 100
KEY_F = 102
KEY_G = 103
KEY_H = 104
KEY_I = 105
KEY_N = 110
KEY_Q = 113
KEY_S = 115
KEY_T = 116
KEY_SPC = 32

# special keys
if os.name == 'posix':
    KEY_LEFT = 65361
    KEY_UP = 65362
    KEY_RIGHT = 65363
    KEY_DOWN = 65364
    KEY_PGDOWN = 65434
    KEY_PGUP = 65435

    KEY_DEL = {65597:'0', 65569:'1', 65570:'2', 65571:'3', 65572:'4', 65573:'5', 65574:'6', 65583:'7', 65576:'8', 65577:'9'}

elif os.name == 'nt':
    KEY_LEFT = 2424832
    KEY_UP = 2490368
    KEY_RIGHT = 2555904
    KEY_DOWN = 2621440
    KEY_PGDOWN = 65365
    KEY_PGUP = 65366

    KEY_DEL = {65597:'0', 65569:'1', 65570:'2', 65571:'3', 65572:'4', 65573:'5', 65574:'6', 65583:'7', 65576:'8', 65577:'9'}
else:
    print "Unknown system: %s !!!" %os.name
    quit()


def key_process( key, capture, frame, f, t, flags ):
    global delay

    if key < 0:
        return True

    ret = True

    if key == KEY_C:
        flags['polygon'] = not flags['polygon']

    if key == KEY_F:
        flags['frame'] = not flags['frame']

    if key == KEY_G:
        flags['data'] = not flags['data']

    if key == KEY_H:
        flags['help'] = not flags['help']

    if key == KEY_S:
        cv2.imwrite( "snap_%d.png" %f, frame )

    #if key == KEY_P:
    #    flags['prev'] = not flags['prev']

    if key == KEY_RIGHT:
        change_frame( f + 50 )

    if key == KEY_LEFT:
        change_frame( f - 50 )

    if key == KEY_UP:
        delay = max( delay / 2, 1 )

    if key == KEY_DOWN:
        delay = min( delay * 2, 100000 )
        
    if key == KEY_SPC:
        flags['pause'] = True

    if key == KEY_Q:
        ret = False

    return ret

#======================================================================================================================
# Drawing utility functions
#======================================================================================================================

def draw_help( img ):
    txts = (
        "f: toogle frame info",
        "c: toogle polygon contour",
        "g: toogle graphic data",
        "p: toogle previous frame data",
        "s: take the current snapshot",
        "",
        "RIGHT: forward 2 secs",
        "LEFT : backward 2 secs",
        "",
        "UP  : speed up",
        "DOWN: slow down",
        "Pg DOWN: next window",
        "Pg UP  : prev. window",
        "",
        "q: exit" )

    draw_help_aux( img, txts, 10, 15, 20 )

def draw_help_aux( img, txts, x_left, y_start, y_inc ):
    y = y_start
    for txt in txts:
        cv2.putText( img, txt, (x_left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) )
        y += y_inc

def draw_data( frame, frame_data, f, flags ):
    # draw lines
    draw_line( frame, frame_data.get( f ), 1 )

    if flags['prev']:
        draw_line( frame, frame_data.get( f-1 ), 1 )

    draw_speed( frame, frame_data.get( f-10 ), frame_data.get( f ) )

def draw_dots( img, data ):
    if (not data is None) and (not data.empty):
        cv2.circle( img, data.head, 3, (0, 0, 255), -1 )
        cv2.circle( img, data.tail,   3, (255, 0, 0), -1 )

        if data.laser_on:
            cv2.circle( img, data.center, 10, (0, 0, 255), -1 )
        else:
            cv2.circle( img, data.center,   3, (0, 255, 0), -1 )

def draw_line( img, data, thick=1 ):
    if (not data is None) and (not data.empty):
        cv2.line( img, data.tail, data.head, (255, 255, 255), thick )

def draw_speed( img, dt1, dt2 ):
    if (not dt1 is None) and (not dt2 is None) and (not dt1.empty) and (not dt2.empty):
        mouse_vector = norm_vector( np.array( dt2.head ) - np.array( dt2.tail ) )
        speed_vector = norm_vector( np.array( dt2.center ) - np.array( dt1.center ) )

        ld = abs(lin_dist( np.array(dt2.center), np.array(dt1.center) ) * nice_dot( mouse_vector, speed_vector ))
        speed_point = np.int32( np.array( dt2.center ) + speed_vector * ld )

        if ld > 5:
            color = (0, 0, 255)
        else:
            color = (255, 0, 255)

        #cv2.line( img, dt2.center, dt1.center, color, 2 )
        cv2.line( img, dt2.center, tuple(speed_point), color, 2 )
        cv2.circle( img, dt1.center, 3, (0, 255, 0), 1 )
        cv2.putText( img, "%5.2f" %ld, dt2.tail, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255) );

def draw_poly( img, poly_points ):
    cv2.polylines( img, poly_points, True, (0, 0, 255), 1 )

    for (i,(x, y)) in enumerate(poly_points[0]):
        cv2.circle( img, (x, y), 4, (0,255,0), -1 )


#======================================================================================================================
# Main function
#======================================================================================================================

def main( io, prj ):
    global capture
    global fcount
    global fwidth
    global delay

    cv2.namedWindow( "Mouse Tracker" )

    # gets the polygon points
    poly_points = [np.array( prj.poly, dtype=np.int32 )]

    # flags
    flags = {'dcircle': False, 'data': True, 'frame': True, 'help': False, 'pause': False, 'polygon':False, 'prev': True}

    # start video capture
    capture = cv2.VideoCapture( prj.video_file )
    fcount = int(capture.get( cv2.cv.CV_CAP_PROP_FRAME_COUNT ))
    fps = int(capture.get( cv2.cv.CV_CAP_PROP_FPS ))

    # create the mask
    poly = [np.array( prj.poly, dtype=np.int32 )]

    # read the data list
    io.show( "\nReading data..." )
    frame_data = FrameData( prj.data_file )
    io.show( "\b\b\b [Done]\n" )

    # main loop
    while True:
        f = int(capture.get( cv2.cv.CV_CAP_PROP_POS_FRAMES ))
        t = int(capture.get( cv2.cv.CV_CAP_PROP_POS_MSEC ))
        ts = (t / 1000)
        tm = (t % 1000)

        # get the frame from the video
        (ret, frame) = capture.read()

        if not ret:
            break

        fwidth = np.size(frame, 1)
        fheight = np.size(frame, 0)

        # FLAG based drawing
        draw_dots( frame, frame_data.get( f ) )

        if flags['help']:
            draw_help( frame )

        if flags['data']:
            draw_data( frame, frame_data, f, flags )

        if flags['frame']:
            cv2.putText( frame, "frame: %7d" %f, (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255) )
            cv2.putText( frame, "time : %s.%04d" %(time.strftime( "%H:%M:%S", time.gmtime(ts) ), tm), (0, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255) )
            cv2.putText( frame, "speed: %5.2f fps" %(1000.0/(delay*fps)), (0, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255) )

        if flags['polygon']:
            draw_poly( frame, poly_points )

        cv2.imshow("Mouse Tracker", frame)
        cv2.setMouseCallback( "Mouse Tracker", on_mouse_main, 0 )

        # PAUSED
        if flags['pause']:
            cv2.putText( frame, "Press any key...", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) );
            cv2.imshow("Mouse Tracker", frame)
            flags['pause'] = False
            cv2.waitKey( 0 )

        cv2.putText( frame, "press 'h' for help", (10, fheight-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) )

        ok = key_process( cv2.waitKey( delay ), capture, frame, f, t, flags )

        if not ok:
            break

# === Returns a fully parametrized 'Command' object.
# === Mandatory function to work with 'generic_gui' module.
def cfg():
    command = Command( "View" )
    command.add_arg( Argument( name="Project directory", atype="dir", cmd_name="project",  desc="project directory name" ) )
    command.add_arg( Argument( name="Frame delay",       atype="int", cmd_name="--delay",  cmd_option="-d", default=1, minimum=0, maximum=100000, desc="set the delay between frames (ms)." ) )

    return command

# === Receives the arguments as they were entered on the gui or in the command line
# === Returns a fully parametrized 'Command' object.
def run( args, io ):
    global delay

    # open the project
    check_dir( io, args.project, "Project File" )
    prj = Project( args.project )

    delay = args.delay

    main( io, prj )

    cv2.destroyAllWindows()

#
# If the script is called directly from the command line
#
if __name__ == "__main__":
    # get the arguments from the command line
    args = cfg().get_parser().parse_args()
    run( args, InOutTerminal() )

