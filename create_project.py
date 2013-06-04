#!/usr/bin/env python

#import argparse
import os

from tracker.command import *
from tracker.data import *
from tracker.utils import *

# Mandatory function to work with 'generic_gui' module. Should return a fully parametrized 'Command' object.
def cfg():
	command = Command( "New" )
	command.add_arg( Argument( name="Project File", atype="dir",  cmd_name="project",  desc="project directory name" ) )
	command.add_arg( Argument( name="Video File",   atype="file", cmd_name="video",    desc="input video full name" ) )
	command.add_arg( Argument( name="Polygon",      atype="str",  cmd_name="--poly",  cmd_option="-p", default="[[136, 28], [573, 31], [566,464], [132, 460]]", desc='set the polygon data (e.g. "[[0,0],[0,10],[10,10],[10,0]]").' ) )
        
	return command

# Mandatory function to work with 'generic_gui' module. Receives the arguments as they were entered on the gui or in the command line
def run( args, io ):
        #filename = "e:\\Videos_temp_to_Jose_Analisis\\Dec2012_ArchTDLS\\Dec2012BOpen_field_Lside.avi"
        filename = args.video
        check_file( io, filename, "Video file" )

        #projectpath = "C:\\Users\\User\\Documents\\tracker\\tecuapetla\\fatuel10"
        projectpath = args.project
        if not os.path.isdir( projectpath ):
                os.mkdir( projectpath )

        exec( "poly=" + args.poly )

        prj = Project.new( args.project, os.path.realpath( args.video ), poly, [70, 70], (20, 20, 5, 255, False) )
        prj.save()

        io.show( "DONE!\n" )

#
# If the script is called directly from the command line
#
if __name__ == "__main__":
	# get the arguments from the command line
	args = cfg().get_parser().parse_args()
	run( args, InOutTerminal() )
