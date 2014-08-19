# Simple Mouse Tracker

Simple Mouse Tracker (SMT) is a video tracker software based on Open CV. It intends to track body, head and tail cordinates (X,Y) of a single color mouse over a homogeneous color background.

## 1.Software Requirements

The SMT runs on any platform that supports Python. Software requirements are:

1. Python >= 2.7.*
2. NumPy  >= 1.7.*  (see: http://www.numpy.org/)
3. OpenCV 2         (see: http://opencv.willowgarage.com/wiki/)

## 2. Installation

### 2.1. Required software

#### 2.1.1. Windows

_Note: This steps were only tested with 32bits Python version._

1. Download and install python 2.7 32bits

    `http://www.python.org/download/`
    
2. Download and install numpy 32bits for python 2.7

    `http://sourceforge.net/projects/numpy/files/NumPy/`

3. Download OpenCV 2.4.2 and unzip it to `C:\`

    `http://sourceforge.net/projects/opencvlibrary/files/opencv-win/`

4. Add Python to the Windows PATH.
    _(check here how to: http://www.computerhope.com/issues/ch000549.htm)_

5. Add `c:\opencv\build\x86\vc10\bin` to the Windows PATH

6. Copy the file `c:\opencv\build\python\2.7\cv2.pyd` to `c:\Python27\Lib\site-packages\` directory

7. Open the command console:
    * Start
    * Run
    * Command
    * Type `cmd`
    * At the command window type `python`

8. To check if it is correctly installed run the following python command:

    `>>> import cv2`

If nothing happens everything should be OK.

#### 2.1.1. Linux

Run the following installation commands:
   
    $ sudo apt-get install python2.7
    $ sudo apt-get install python-numpy
    $ sudo apt-get install python-opencv

### 2.2. Simple Mouse Tracker installation

1. Unzip the zip file to any folder in your computer (e.g. `/home/user/mt`).

2. From the console run the following command:

    `$ python /home/user/mt/create_project.py`

If you get the following message, the installation should be fine:
    
    usage: create_project.py [-h] [-p POLY] project video
    create_project.py: error: too few arguments

_Note: In the above command replace `/home/user/mt` by you own settings._

## 3. Usage

_Note:_ Before start remember that the examples in this section assume the following:

* The mouse tracker was installed in `/home/user/mt`.
* We have a video file named 'video.avi' in the directory '/home/user/myvideos';
* We call the project 'test' and will save it in the directory '/home/user/myprjs'

### STEP 1 - Create a project for the video we want to analyze:

In the command line run:

    $ python /home/user/mt/create_project.py /home/user/myprjs/test /home/user/myvideos/video.avi

This will create the subdirectory `test` under the directory `/home/user/myprjs`. This directory will contain all the data files relative to the project.

### STEP 2 - Define the mask region for the analysis:

In the command line run:

    $ python /home/user/mt/set_mask.py /home/user/myprjs/test

You should see the video running in two windows 'Original' and 'Mouse'. The 'Original' one is your video as you acquired it, the 'Mouse' is the binary version accoring to the threshold options defined. The goal here is to have the mouse showing up as a white blob on a black background in the 'Mouse' window. Nothing else should appear white in that window. To acchieve that you can:

1. Drag around the blue dots in order to include as much as you can from the arena without adding noise.
2. Click the mouse right button to add more dot's to the polygon and fine tune the area.
3. Play around with the `_thresh` parameters in the `/home/user/myprjs/test/project.cfg` file:

    `_thresh: [ 70, 35 ]`

When done press the 's' key to save the new mask into the project or 'q' to quit without saving.

### STEP 3 - Run the analysis:

In the command line run:

    $ python /home/user/mt/get_data.py /home/user/myprjs/test

This will create the file `/home/user/myprjs/test/coords.dat` with all the coordinates from the tracking. If you want to watch the video during the analysis (this will be a lot slower!) do:

    $ python /home/user/mt/get_data.py -s /home/user/myprjs/test

Notice the '-s' option.

Each time you run `get_data.py` a new version of the `coords.dat` file wil be created and the previous one will be renamed `coords.dat.NNN.bak`.

### 4 - Explore the data:

In the command line run:

    $ python /home/user/mt/view_data.py /home/user/myprjs/test

This will show you the result of the tracking. Remember: The red dot should always match the nose of the mouse!

Useful options:

* You can jump for a given frame using the '-j' parameter.
* You can slowdown the video with the '-d' parameter.

The following command will show the results from frame 1050 on, with an interval of 300 ms between frames:

    $ python /home/user/mt/view_data.py -j 1050 -d 300 /home/user/myprjs/test

### 5 - Fix the data:

Often the `get_data.py` script makes a lot of mistakes when classifying the nose and the tail. If this is your case you can try to correct the mistakes after processing all the file using the following command:

    $ python /home/user/mt/fix_data.py /home/user/myprjs/test

This will analyse the 'coords.dat' and try to correct the mistakes by the relative movement of the head, tail and body during the whole video.

When done a new `/home/user/myprjs/test/coords.dat` will be saved.


### 6 - Estimate the position of the mouse backbone:

    $ python /home/user/mt/calc_backbone.py /home/user/myprjs/test -b 5

This will calculate the line that divide the mouse blob in half, basing in N number of points (option -b).

# Have Fun
    
