#!/usr/bin/env python

import argparse
import os
import sys
import time

import cv2
import numpy as np

from tracker.analysis import *
from tracker.command import *
from tracker.data import *
from tracker.utils import *

# Mandatory function to work with 'generic_gui' module. Should return a fully parametrized 'Command' object.
def cfg():
    command = Command( "Fix" )
    command.add_arg( Argument( name="Project File",  atype="dir",  cmd_name="project",  desc="project directory name" ) )

    return command

# Mandatory function to work with 'generic_gui' module. Receives the arguments as they were entered on the gui or in the command line
def run( args, io ):
    # open the project
    check_dir( io, args.project, "Project File" )
    prj = Project( args.project )

    # read the data list
    io.show( "\nReading data..." )
    frame_data = FrameData( prj.data_file )
    io.show( "[Data ok]\n" )

    # fix the data
    fix_data( io, frame_data, prj.poly )
    
    frame_data.save( prj.data_file )
    io.show( "[DONE]\n" )

    cv2.destroyAllWindows()

#
# If the script is called directly from the command line
#
if __name__ == "__main__":
    # get the arguments from the command line
    args = cfg().get_parser().parse_args()
    run( args, InOutTerminal() )