##### CODE TO DETECTOR ACTIVE SPEAKER IN A MULTI SPEAKER SETUP #####

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import datetime
from collections import OrderedDict

# for inner mouth coordinates 

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (60, 68)),
])


def mouth_aspect_ratio(mouth):
    # compute the euclidean distances 
    
    A = dist.euclidean(mouth[2], mouth[6])
    
    E = dist.euclidean(mouth[1], mouth[5])
    F = dist.euclidean(mouth[3], mouth[7])
    
    # compute the euclidean distances between the horizontal
    # mouth landmark (x, y)-coordinates

    C = dist.euclidean(mouth[0], mouth[4])
    D = dist.euclidean(mouth[1], mouth[3])
    G = dist.euclidean(mouth[5], mouth[7])

    # compute the mouth aspect ratio
    mouth_ar = (A  + E + F) / ( C + D + G)

    # return the mouth aspect ratio
    return mouth_ar


# construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
                help="path to input video file")
args = vars(ap.parse_args())

# define constants, one for mouth aspect ratio to indicate
# movement and one constant for the number of consecutive frames
# the mouth must be above set threshold

mouth_threshold = 0.19
mouth_consecutive_frames = 3

# initialize the frame counters and the total number of blinks
count = 0

# dlib's face Dtector (Histogram OG-based) and then create
# the facial landmark predictor

print("[Details] active/passive speaker detector active...")
Dtector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
# grab the indexes of the facial landmarks for the inner mouth

(lStart, lEnd) = FACIAL_LANDMARKS_IDXS["mouth"]
# start video stream 
video_frame = FileVideoStream(args["video"]).start()
fileStream = True
time.sleep(1.0)

# loop over frames from the video stream
while True:
    #check if buffer empty else proceed
    if fileStream and not video_frame.more():
        break

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    
    frame = video_frame.read()
    frame = imutils.resize(frame, width=750)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # grayscale frame
    
    rects = Dtector(gray, 0)

    # loop over the face detections
    
    for i, rect in enumerate(rects):
    
    
        # determine the facial landmarks for the face region.
        # convert the facial landmark coordinates to a NumPy array.
        
        shape = predictor(gray, rect) 
        shape = face_utils.shape_to_np(shape)

        # extract the inner coordinates, then use the
        # coordinates to compute the mouth aspect ratio 
        
        innermouth = shape[lStart:lEnd]
        innermouth_ar = mouth_aspect_ratio(innermouth)

        mouth_ar = innermouth_ar 
        mouth_consecutive_frames = 3
        innermouthHull = cv2.convexHull(innermouth)
        cv2.drawContours(frame, [innermouthHull], -1, (0, 255, 0), 1)

        # check to see if the mouth aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        
        if mouth_ar > 0.55:
            
            if count<mouth_consecutive_frames: 
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.putText(frame, "Face is Speaking", (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                

            count += 1

        # otherwise, the mouth aspect ratio is not below the blink
        # threshold
        
        else:
            count =0
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.putText(frame, "Face is not speaking", (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # if the mouths were open for a thresholded number of
            # then increment the total number of blinks
        
        
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        # draw the total number of mouth movements on the frame along with
        # the computed mouth aspect ratio for the frame
        
        cv2.putText(frame, "Threshold Ratio: {:.2f}".format(mouth_ar), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
               

    # show frames
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    
    if key == ord("z"):
        break
cv2.destroyAllWindows()
video_frame.stop()
