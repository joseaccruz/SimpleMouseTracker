#!/usr/bin/env python

import argparse

from tracker.analysis import *
from tracker.command import *
from tracker.data import *
from tracker.utils import *


# Mandatory function to work with 'generic_gui' module. Should return a fully parametrized 'Command' object.
def cfg():
    command = Command( "Calc backbone" )
    command.add_arg( Argument( name="Project File",  atype="dir",  cmd_name="project",  desc="project directory name" ) )
    command.add_arg( Argument( name="Show Video",    atype="bool", cmd_name="--show_video", cmd_option="-s", default=True, desc="show the video while processing." ) )
    command.add_arg( Argument( name="Frame delay",   atype="int",  cmd_name="--delay",      cmd_option="-d", default=1, minimum=0, maximum=100000,    desc="set the delay between frames (ms)." ) )
    command.add_arg( Argument( name="Jump to frame", atype="int",  cmd_name="--jump",       cmd_option="-j", default=0, minimum=0, maximum=100000,    desc="jump to a specific frame." ) )
    command.add_arg( Argument( name="Number of backbone points", atype="int",  cmd_name="--backbone", cmd_option="-b", default=10, minimum=2, maximum=100000,    desc="set the number of backbone points." ) )
    #command.add_arg( Argument( name="Path to export rat images", atype="str",  cmd_name="--exportimages", cmd_option="-x", default=None, desc="Path to where the rat images should be saved" ) )
    
    return command

# Mandatory function to work with 'generic_gui' module. Receives the arguments as they were entered on the gui or in the command line
def run( args, io ):
    # open the project
    check_dir( io, args.project, "Project File" )
    prj = Project( args.project )

    io.show( "\nReading data..." )
    frame_data = FrameData( prj.data_file )
    io.show( "[Data ok]\n" )

    calculate_backbone( io, frame_data, prj.video_file, prj.poly, prj.thresh, args.show_video, args.delay, args.jump, args.backbone)

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