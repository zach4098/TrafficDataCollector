import cv2
import depthai as dai
import numpy as np
import time as time1
from time import time, ctime
import blobconverter

t = time()
timeOfCreation = ctime(t)
timeOfCreation = str(timeOfCreation)

data = open("Data/Data collected @ {}.txt".format(timeOfCreation), "x")
data = open("Data/Data collected @ {}.txt".format(timeOfCreation), "w")

labelMap = ["No Vehicle", "Vehicle"]

pipeline = dai.Pipeline()

camRgb = pipeline.createColorCamera()
detectionNetwork = pipeline.createMobileNetDetectionNetwork()
objectTracker = pipeline.createObjectTracker()

xlinkOut = pipeline.createXLinkOut()
trackerOut = pipeline.createXLinkOut()

xlinkOut.setStreamName("preview")
trackerOut.setStreamName("tracklets")

camRgb.setPreviewSize(672, 384)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(30)

detectionNetwork.setBlobPath(str(blobconverter.from_zoo(name = "vehicle-detection-adas-0002", shaves = 6)))
detectionNetwork.setConfidenceThreshold(0.8)
detectionNetwork.input.setBlocking(False)

objectTracker.setDetectionLabelsToTrack([1])
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

camRgb.preview.link(detectionNetwork.input)
objectTracker.passthroughTrackerFrame.link(xlinkOut.input)

detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

#camRgb.video.link(objectTracker.inputTrackerFrame)

detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
detectionNetwork.out.link(objectTracker.inputDetections)
objectTracker.out.link(trackerOut.input)

with dai.Device(pipeline) as device:
    preview = device.getOutputQueue("preview", 4, False)
    tracklets = device.getOutputQueue("tracklets", 4, False)

    startTime = time1.monotonic()
    counter = 0
    fps = 0
    frame = None
    countOff = 0

    sensorBox = [316, 356]
    sensorBoxY = [172, 212]

    leftBox = [0, 336]
    bottomBox = [0, 192]
    rightBox = [336, 672]
    upperBox = [192, 384]

    vehicles = []
    totalVehicles = 0
    vehicleDict = []

    leftRight = False

    vehicleDir = ""

    while True:
        imgFrame = preview.get()
        track = tracklets.get()
        counter += 1
        currentTime = time1.monotonic()
        countOff += 1
        if (currentTime - startTime) > 1:
            fps = counter/(currentTime - startTime)
            counter = 0
            startTime = currentTime
        color = (255, 0, 0)
        frame = imgFrame.getCvFrame()
        trackletsData = track.tracklets
        
        for t in trackletsData:
            roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            bbox = [x1, y1, x2, y2]

            midpoint = (bbox[0]+bbox[2])/2
            midpoint = int(round(midpoint))

            midpointY = (bbox[1] + bbox[3])/2
            midpointY = int(round(midpointY))

            area = (y1 - y2) * (x1 - x2)

            vehicleID = t.id
            label = labelMap[t.label]

            vehicleStatus = str(t.status)

            if leftRight:
                if vehicleStatus == "TrackingStatus.NEW" and bbox[2] <= leftBox[1]:
                    vehicleDir = "left"
                elif vehicleStatus == "TrackingStatus.NEW" and bbox[0] >= rightBox[0]:
                    vehicleDir = "right"

                if vehicleID not in vehicles and sensorBox[0] <= midpoint <= sensorBox[1]:
                    t = time()
                    currentTime = ctime(t)
                    vehicles.append(vehicleID)
                    totalVehicles += 1
                    vehicleDict.append(("Vehicle #{}".format(totalVehicles), str(currentTime), vehicleDir))

                if vehicleID in vehicles and vehicleStatus == "TrackingStatus.REMOVED" or sensorBox[0] > midpoint < sensorBox[1]:
                    if vehicleID in vehicles and len(vehicles) != 0:
                        vehicles.remove(vehicleID)
            elif not leftRight:
                if vehicleStatus == "TrackingStatus.NEW" and bbox[3] <= bottomBox[1]:
                    vehicleDir = "left"
                elif vehicleStatus == "TrackingStatus.NEW" and bbox[1] >= upperBox[0]:
                    vehicleDir = "right"

                if vehicleID not in vehicles and sensorBoxY[0] <= midpointY <= sensorBoxY[1]:
                    t = time()
                    currentTime = ctime(t)
                    vehicles.append(vehicleID)
                    totalVehicles += 1
                    vehicleDict.append(("Vehicle #{}".format(totalVehicles), str(currentTime), vehicleDir))
                    data.write("{}-{}-{}".format("Vehicle #{}".format(totalVehicles), str(currentTime), vehicleDir) + "\n")

                if vehicleID in vehicles and vehicleStatus == "TrackingStatus.REMOVED" or sensorBoxY[0] > midpointY < sensorBoxY[1]:
                    if vehicleID in vehicles and len(vehicles) != 0:
                        vehicles.remove(vehicleID)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        if leftRight:
            cv2.rectangle(frame, (sensorBox[0], 0), (sensorBox[1], 384), color, 1)
            cv2.rectangle(frame, (leftBox[0], 0), (leftBox[1], 384), (0, 255, 0), 1)
            cv2.rectangle(frame, (rightBox[0], 0), (rightBox[1], 384), (0, 0, 255), 1)
        elif not leftRight:
            cv2.rectangle(frame, (0, sensorBoxY[0]), (672, sensorBoxY[1]), color, 1)
            cv2.rectangle(frame, (0, bottomBox[0]), (672, bottomBox[1]), (0, 255, 0), 1)
            cv2.rectangle(frame, (0, upperBox[0]), (672, upperBox[1]), (0, 0, 255), 1)
        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_PLAIN, 4, color)
        #print(vehicles)
        print(totalVehicles)
        cv2.imshow("tracker", frame)
        if cv2.waitKey(1) == ord('t'):
            leftRight = not leftRight

        if cv2.waitKey(1) == ord('q'):
            #data = open("Data/Data collected @ {}.txt".format(timeOfCreation), "x")
            #data = open("Data/Data collected @ {}.txt".format(timeOfCreation), "w")
            #for item in vehicleDict:
                #data.write("{}-{}-{}".format(item[0], item[1], item[2]) + "\n")
            data.close()
            break