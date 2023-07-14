import cv2
import depthai as dai
import numpy as np
import time
import blobconverter

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

camRgb.video.link(objectTracker.inputTrackerFrame)

detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
detectionNetwork.out.link(objectTracker.inputDetections)
objectTracker.out.link(trackerOut.input)

with dai.Device(pipeline) as device:
    preview = device.getOutputQueue("preview", 4, False)
    tracklets = device.getOutputQueue("tracklets", 4, False)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    frame = None
    countOff = 0


    while True:
        imgFrame = preview.get()
        track = tracklets.get()
        counter += 1
        currentTime = time.monotonic()
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

            area = (y1 - y2) * (x1 - x2)

            ID = t.id
            label = labelMap[t.label]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_PLAIN, 4, color)
        cv2.imshow("tracker", frame)

        if cv2.waitKey(1) == ord('q'):
            break