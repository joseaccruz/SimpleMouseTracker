import json
import os

import numpy as np

from tracker.utils import *

# ======================================================================================================================
# Projects class
# ======================================================================================================================
class Project(object):
    def __init__( self, dir_name=None ):
        self._data_keys = ["_video_file", "_poly", "_thresh", "_laser", "_frame_rate", "_frame_width", "_frame_height"]

        # loads the project file from the directory
        if Project.is_project( dir_name ):
            self._dir_name = dir_name

            data = json.load( open( dir_name + os.sep + "project.cfg" ) )

            for key in self._data_keys:        
                self.__dict__[key] = data.get( key, None )

            self._validate()

    #
    # Public Methods
    #
    def save( self ):
        data = {}
        fname = self._dir_name + os.sep + "project.cfg"

        for key in self._data_keys:        
            data[key] = self.__dict__.get( key, None )

        backup_file( fname )
        json.dump( data, open( fname, "w" ), indent=4 )

    @staticmethod
    def new( dir_name, video_file, poly, thresh, laser ):
        prj = Project()

        prj._dir_name = dir_name
        prj._video_file = video_file
        prj._poly = poly
        prj._thresh = thresh
        prj._laser = laser

        # get data from the video file
        capture = cv2.VideoCapture( prj.video_file )
        prj._frame_rate = capture.get( cv2.cv.CV_CAP_PROP_FPS )
        prj._frame_width  = capture.get( cv2.cv.CV_CAP_PROP_FRAME_WIDTH )
        prj._frame_height = capture.get( cv2.cv.CV_CAP_PROP_FRAME_HEIGHT )

        return prj

    @staticmethod
    def is_project( dir_name ):
        return (not dir_name is None) and os.path.isdir( dir_name ) and os.path.isfile( dir_name + os.sep + "project.cfg" )

    #
    # Properties
    #
    def _get_video_file( self ):
        return self._video_file
    
    def _set_video_file( self, value ):
        self._video_file = value
    
    def _get_data_file( self ):
        return self._dir_name + os.sep + "coords.dat"

    def _get_frame_rate( self ):
        return self._frame_rate

    def _get_frame_width( self ):
        return self._frame_width
    
    def _get_frame_height( self ):
        return self._frame_height
    
    def _get_poly( self ):
        return self._poly
    
    def _set_poly( self, value ):
        self._poly = value

    def _get_thresh( self ):
        return self._thresh
    
    def _set_thresh( self, value ):
        self._thresh = value

    def _get_laser( self ):
        return self._laser
    
    def _set_laser( self, value ):
        self._laser = value

    # properties
    video_file  = property( _get_video_file, _set_video_file )
    data_file   = property( _get_data_file )
    frame_rate = property( _get_frame_rate )
    frame_width = property( _get_frame_width )
    frame_height = property( _get_frame_height )
    poly = property( _get_poly, _set_poly )
    thresh = property( _get_thresh, _set_thresh )
    laser = property( _get_laser, _set_laser )

    # private methods
    def _validate(self):
        keys = ["_dir_name", "_video_file", "_poly", "_thresh", "_laser", "_frame_rate", "_frame_width", "_frame_height"]
        keys_not_found = filter( lambda k: not self.__dict__.has_key( k ), keys )

        if len(keys_not_found) > 0:
            error( "Missing data in project file: (%s). This is probably a wrong version of the project file..." %", ".join( keys_not_found ) )

# ======================================================================================================================
# Data collected from videos
# ======================================================================================================================
class FrameData(object):
    PROX_NONE        = 0
    PROX_WIN         = 1
    PROX_CLOSE       = 2

    def __init__( self, fname=None ):
        self._data = []

        if not fname is None:
            self.load( fname )
        
    def add( self, frame, data_item ):
        self._data.append( data_item )

    def get( self, frame ):
        if (frame > 0) and (frame < len(self._data)):
            ret = self._data[frame]
        else:
            ret = None

        return ret

    def get_list( self ):
        return self._data

    def save( self, fname ):
        backup_file( fname )

        fo = open( fname, "w" )
        for (frame, dt) in enumerate( self._data ):
            fo.write( "%d\t%s\n" %(frame, str(dt)) )
        fo.close()

    def load( self, fname ):
        self._data = []

        data = map( lambda s: s.strip().split( '\t' ), open( fname ).read().strip().split( '\n' ) )

        for (frame, dt) in enumerate(data):
            if len(dt) != 9:
                error( "Wrong number of fields in data file '%s' frame '%d. Expected 10 found %d." %(fname, frame, len(dt[1:])) )

            if int(dt[0]) != frame:
                error( "Wrong frame number in data file '%s'. Expected %d found %d." %(fname, frame, int(dt[0])) )

            self._data.append( FrameDataItem.from_array( map( int, dt[1:] ) ) )

# ======================================================================================================================
# Frame Data Entry
# ======================================================================================================================
class FrameDataItem(object):
    def __init__( self, center=None, head=None, tail=None, laser_on=0 ):
        self._empty = (center is None) or (head is None) or (tail is None)
        self._center = center
        self._head = head
        self._tail = tail
        self._laser_on = laser_on

    def __str__( self ):
        if self._empty:
            txt = "0\t0\t0\t0\t0\t0\t0\t0"
        else:
            txt = "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d" %(not self._empty, self._center[0], self._center[1], self._head[0], self._head[1], self._tail[0], self._tail[1], self._laser_on)

        return txt

    @staticmethod
    def from_array( data ):
        # check if it's empty
        if( not bool(data[0]) ):
            data_item = FrameDataItem()
        else:           
            data_item = FrameDataItem( (data[1], data[2]), (data[3], data[4]), (data[5], data[6]), data[7] )

        return data_item

    #
    # Properties
    #  
    def _get_empty( self ):
        return self._empty

    def _set_empty( self, value ):
        self._empty = value

    def _get_center( self ):
        return self._center

    def _get_head( self ):
        return self._head

    def _set_head( self, value ):
        self._head = value

    def _get_tail( self ):
        return self._tail

    def _set_tail( self, value ):
        self._tail = value

    def _get_laser_on( self ):
        return self._laser_on

    def _set_laser_on( self, value ):
        self._laser_on = value

    # properties
    empty  = property( _get_empty, _set_empty )
    center = property( _get_center )
    head = property( _get_head, _set_head )
    tail = property( _get_tail, _set_tail )
    laser_on = property( _get_laser_on, _set_laser_on )
