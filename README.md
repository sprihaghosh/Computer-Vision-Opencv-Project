# Spriha-Ghosh-21JE0938-Computer-Vision-Opencv-
Atulya Project
Task at hand:

I work in a firm where they have a specific department, to autonomously put aruco markers on square boxes. My task is to write a python code, for finding square boxes in an image and then place aruco markers exactly overlapping on it with the following rules.

Box Colour  Marker ID
Green       1
Orange      2
Black       3
Pink-Peach  4

The aruco markers should exactly be placed on the squares only, even the orientation of the square should be kept in mind.

To accomplish this task, the Opencv module of python is used. 

First, all the ids of the aruco markers given is identified by importing cv2.aruco

Then, the markers are rotated to vertical positions by rotating them by angle of inclination found. This is done by using cv2.getRotationMatrix2D and cv2.warpAffine.

Angle is found by getting the slope between the centre and the mid point of one of the edges.

Specific square boxes with diffrernt colours are identified through masking and contour operations.

At a time, one coloured square box is masked and chosen as the roi. Its coordinates are taken as offsets and the relevant aruco marker is resized according to it. 

The resized marker is then overlayed on the roi.

The above task is carried out for each of colours and the final image is obtained as the combination of all the overlayed outputs.
