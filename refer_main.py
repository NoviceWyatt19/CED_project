from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# -------------------------------------------------------------------------
# loop over frames from the video stream
while (cap.isOpened()):
    # -------------------------eyes------------------------------
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale channels
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # loop over the face detections
    for (x, y, w, h) in rects:
        # construct a dlib rectangle object from the Haar cascade bounding box
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            # if the eyes were closed for a sufficient number of frames, then sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True
                    #(replace)pin9.write(1) 눈 탐지x경우 보낼 신호로 대체

                    # check to see if the TrafficHat buzzer should be sounded
                    if args["alarm"] > 0:
                        #(replace)
                        th.buzzer.blink(0.1, 0.1, 10, background=True)

                # draw an alarm on the frame
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # otherwise, the eye aspect ratio is not below the blink threshold, so reset the counter and alarm
        else:
            COUNTER = 0
            ALARM_ON = False
            #(replace)pin9.write(0)

        # draw the computed eye aspect ratio on the frame to help
        # with debugging and setting the correct eye aspect ratio thresholds and frame counters
        cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # -------------------------------car--------------------------------
    ret, frame = cap.read()

    if (type(frame) == type(None)):
        break

    frame = process(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
cap.release()