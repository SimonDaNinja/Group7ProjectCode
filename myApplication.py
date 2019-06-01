#!/usr/bin/env python3

# Copyright (C) 2018 Christian Berger
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

# sysv_ipc is needed to access the shared memory where the camera image is present.
import sysv_ipc
# numpy and cv2 are needed to access, modify, or display the pixels
import numpy
import cv2
# OD4Session is needed to send and receive messages
import OD4Session_for_Python3
# Import the OpenDLV Standard Message Set.
import opendlv_standard_message_set_v0_9_6_pb2

#BEFORE DEPLOYMENT, REMOVE THE FOLLOWING LINE!
import os
import time

import math
import scipy.spatial

MAXIMUM_PEDAL_POSITION = .25
MINIMUM_PEDAL_POSITION = -1
MAXIMUM_STEERING_ANGLE_AMPLITUDE = .2

# All behavior currently relies on the mode choice "colormap"
modeChoice = input("Do you want to detect edges or raw or colormap? [edges/raw/colormap] ").lower()

################################################################################
# This dictionary contains all distance values to be filled by function onDistance(...).
distances = { "front": 0.0, "left": 0.0, "right": 0.0, "rear": 0.0 };

################################################################################
# This callback is triggered whenever there is a new distance reading coming in.

def IJFromIndex(index, numberOfPoints):
    i = 0
    n = numberOfPoints-1
    numberOfPointsOnInterval = n
    #print("index shape ",index.shape)
    #print("number of points ",numberOfPoints.shape)
    #exit()
    while index>(n-1):
        numberOfPointsOnInterval -= 1
        n += numberOfPointsOnInterval
        i += 1
    jFromLast =  (n-1) - index
    j = numberOfPoints-jFromLast-1
    if i > numberOfPoints:
        print("i = ",i)
    if j > numberOfPoints:
        print("j from last= ",jFromLast)
        print("j = ",j)
    return int(i),(j)

def IndexRangeFromI(i,numberOfPoints):
    lowerIndex = 0
    ii = 0
    upperIndex = numberOfPoints - 1
    numberOfPointsOnInterval = numberOfPoints-1
    while ii<i:
        ii += 1
        lowerIndex += numberOfPointsOnInterval
        numberOfPointsOnInterval -= 1
        upperIndex += numberOfPointsOnInterval
    return range(lowerIndex, upperIndex)

def GetClusters(adjacencyVector):
    n = len(adjacencyVector)
    numberOfPoints = int((1/2)*(1+math.sqrt(8*n+1)))
    indeces = numpy.where(adjacencyVector != 0)
    visited = set()
    clusterList = list()
    for index in indeces[0]:
        i,j = IJFromIndex(index, numberOfPoints)
        toBeInvestigated = []
        if not i in visited:
            if not j in visited:
                clusterIndeces = set()
                clusterSize = 0
                toBeInvestigated.append(j)
                visited.add(j)
                clusterIndeces.add(j)
                clusterSize += 1

                toBeInvestigated.append(i)
                visited.add(i)
                clusterIndeces.add(i)
                clusterSize += 1
            else:
                for cluster in clusterList:
                    if j in cluster[0]:
                        clusterIndeces = cluster[0]
                        clusterSize = cluster[1]
                        clusterList.remove(cluster)
                        break
                toBeInvestigated.append(i)
                visited.add(i)
                clusterIndeces.add(i)
                clusterSize += 1
        else:
            if not j in visited:
                for cluster in clusterList:
                    if i in cluster[0]:
                        clusterIndeces = cluster[0]
                        clusterSize = cluster[1]
                        clusterList.remove(cluster)
                        break
                toBeInvestigated.append(j)
                visited.add(j)
                clusterIndeces.add(j)
                clusterSize += 1
        while len(toBeInvestigated)>0:
            i = toBeInvestigated.pop()
            indexRange = IndexRangeFromI(i, numberOfPoints)
            for index in indexRange:
                _,j = IJFromIndex(index, numberOfPoints)
                if j not in visited:
                    if adjacencyVector[index] != 0:
                        visited.add(j)
                        toBeInvestigated.append(j)
                        clusterIndeces.add(j)
                        clusterSize += 1
        cluster = (clusterIndeces,clusterSize)
        clusterList.append(cluster)
    return clusterList

def DetectCones(img, pixelsPerWidthUnit, pixelsPerDepthUnit):
    widthDepthRatio = .7*pixelsPerWidthUnit/pixelsPerDepthUnit
    gapTolDepth = 20
    gapTolWidth = 20
    distanceTol = 220
    numberOfRows, numberOfColumns = img.shape
    cones = []
    numberOfWidthGapsOnPreviousLine = 10000000
    depthCount = 0
    for i in reversed(range(numberOfRows)):
        #ignore the first few lines
        if depthCount>gapTolDepth:
            row = img[i,:] 
            if (row!=0).any():
                conePositions = (numpy.where(row!=0))[0]
                if len(cones)>0:
                    previousCone = cones[0]
                    if math.sqrt(depthCount**2+(widthDepthRatio*(previousCone[1]-conePositions[0]))**2)>distanceTol:
                        cone = (i,conePositions[0])
                        cones.append(cone)
                        return cones
                    elif math.sqrt(depthCount**2+(widthDepthRatio*(previousCone[1]-conePositions[-1]))**2)>distanceTol:
                        cone = (i,conePositions[-1])
                        cones.append(cone)
                        return cones
                else:
                    cone = (i,conePositions[0])
                    cones.append(cone)
        depthCount += 1
    return cones

def onDistance(msg, senderStamp, timeStamps):
    # print ("Received distance; senderStamp= %s" % (str(senderStamp)))
    # print ("sent: %s, received: %s, sample time stamps: %s" % (str(timeStamps[0]), str(timeStamps[1]), str(timeStamps[2])))
    # print ("%s" % (msg))
    if senderStamp == 0:
        distances["front"] = msg.distance
    if senderStamp == 1:
        distances["left"] = msg.distance
    if senderStamp == 2:
        distances["rear"] = msg.distance
    if senderStamp == 3:
        distances["right"] = msg.distance

COLS = 480
ROWS = 640

# Create a session to send and receive messages from a running OD4Session;
# Replay mode: CID = 253
# Live mode: CID = 112
# TODO: Change to CID 112 when this program is used on Kiwi.
session = OD4Session_for_Python3.OD4Session(cid=112)
# Register a handler for a message; the following example is listening
# for messageID 1039 which represents opendlv.proxy.DistanceReading.
# Cf. here: https://github.com/chalmers-revere/opendlv.standard-message-set/blob/master/opendlv.odvd#L113-L115
messageIDDistanceReading = 1039
session.registerMessageCallback(messageIDDistanceReading, onDistance, opendlv_standard_message_set_v0_9_6_pb2.opendlv_proxy_DistanceReading)
# Connect to the network session.
session.connect()

################################################################################
# The following lines connect to the camera frame that resides in shared memory.
# This name must match with the name used in the h264-decoder-viewer.yml file.
name = "/tmp/img.argb"
# Obtain the keys for the shared memory and semaphores.
keySharedMemory = sysv_ipc.ftok(name, 1, True)
keySemMutex = sysv_ipc.ftok(name, 2, True)
keySemCondition = sysv_ipc.ftok(name, 3, True)
# Instantiate the SharedMemory and Semaphore objects.
shm = sysv_ipc.SharedMemory(keySharedMemory)
mutex = sysv_ipc.Semaphore(keySemCondition)
cond = sysv_ipc.Semaphore(keySemCondition)






################################################################################
#############################       FUNCTIONS       ############################
################################################################################


#This function takes as input an hsv image and outputs a matrix where blue pixels are 1 and other pixels are 0
def ColorFilterBlue(img):
    hsvBlue = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #hsvLowBlue = numpy.array([105,80,80])
    #hsvHiBlue = numpy.array([130,255,100])
    hsvLowBlue = numpy.array([100,50,50])
    hsvHiBlue = numpy.array([140,255,100])
    blueCones = cv2.inRange(hsvBlue,hsvLowBlue,hsvHiBlue)
    return blueCones

#This function takes as input an hsv image and outputs a matrix where blue pixels are 1 and other pixels are 0
def ColorFilterOrange(img):
    hsvOrange = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsvLowOrange = numpy.array([-10,100,80])
    hsvHiOrange = numpy.array([5,255,255])
    orangeCones = cv2.inRange(hsvOrange,hsvLowOrange,hsvHiOrange)
    return orangeCones

#This function takes as input an hsv image and outputs a matrix where yellow pixels are 1 and other pixels are 0
def ColorFilterYellow(img):
    hsvYellow = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsvLowYellow = numpy.array([25,100,100])
    hsvHiYellow = numpy.array([30,255,255])
    yellowCones = cv2.inRange(hsvYellow,hsvLowYellow,hsvHiYellow)
    return yellowCones

################################################################################
# This callback is triggered whenever there is a new distance reading coming in.
def DetectCones(img, pixelsPerWidthUnit, pixelsPerDepthUnit):
    widthDepthRatio = .7*pixelsPerWidthUnit/pixelsPerDepthUnit
    gapTolDepth = 20
    #gapTolWidth = 20
    distanceTol = 220
    numberOfRows, numberOfColumns = img.shape
    cones = []
    numberOfWidthGapsOnPreviousLine = 10000000
    depthCount = 0
    for i in reversed(range(numberOfRows)):
        #ignore the first few lines
        if depthCount>gapTolDepth:
            row = img[i,:] 
            if (row!=0).any():
                conePositions = (numpy.where(row!=0))[0]
                if len(cones)>0:
                    previousCone = cones[0]
                    if math.sqrt(depthCount**2+(widthDepthRatio*(previousCone[1]-conePositions[0]))**2)>distanceTol:
                        cone = (i,conePositions[0])
                        cones.append(cone)
                        return cones
                    elif math.sqrt(depthCount**2+(widthDepthRatio*(previousCone[1]-conePositions[-1]))**2)>distanceTol:
                        cone = (i,conePositions[-1])
                        cones.append(cone)
                        return cones
                else:
                    cone = (i,conePositions[0])
                    cones.append(cone)
        depthCount += 1
    return cones

# This function takes as input a single channel image, as well as a number of dilation iterations and erosion iterations
# It outputs a single channel image where the specified number of dilation iterations have been followed by the specified
# number of erosion iterations on the input image
def DilateAndErode(img, dilationIterations = 20, erosionIterations = 20):
    # dilate
    kernel = numpy.ones([3,3])
    dilation = cv2.dilate(src = img, kernel = kernel, iterations = dilationIterations)
    # erode
    erosion = cv2.erode(src = dilation, kernel = kernel, iterations = erosionIterations)
    return erosion

# This function takes as input a single channel image, two conversion factors from pixels to width units and a reference point
# It then finds the active point in the image which is closest in euclidean distance to the reference point
# It then returns th distance from the ORIGIN to this point (not to be confused with the reference point, rather
# the origin is the car, whereas the reference point is a "typical" aim point)
def GetDistanceToClosestActivePoint(img,pixelsPerWidth,pixelsPerDepth,referencePoint):
    closestActivePoint = GetClosestActivePoint(img,referencePoint)
    if closestActivePoint is None:
        return None, None
    else:
        depthDistanceToClosestActivePoint, widthDistanceToClosestActivePoint = GetDistance(closestActivePoint,pixelsPerWidth,pixelsPerDepth)
        return depthDistanceToClosestActivePoint, widthDistanceToClosestActivePoint

# This function takes as input a single channel image and a reference point. Then it returns the indeces of the active point which is
# closest in euclidean distance to the reference point
def GetClosestActivePoint(img, referencePoint):
    if (img!=0).any():
        activePoints = numpy.transpose(numpy.array(numpy.where(img!=0)))
        distanceVector = numpy.linalg.norm(activePoints-referencePoint,axis=1)
        closestActivePointIndex = numpy.argmin(distanceVector)
        closestActivePoint = activePoints[closestActivePointIndex,:]
        return closestActivePoint
    else:
        return None

# This function takes as input a list of two pixel indeces and conversion units from pixels to length
# it then calculates the distance from the origin to this point
def GetDistance(point,pixelsPerWidth,pixelsPerDepth):
    pointRow = point[0]
    pointCol = point[1]

    pixelDepthDistance = ROWS-pointRow
    pixelWidthDistance = COLS/2-pointCol

    depthDistance = pixelDepthDistance/pixelsPerDepth
    widthDistance = pixelWidthDistance/pixelsPerWidth
    return depthDistance, widthDistance

#The following is for calibration of perspective transformation from empirical data

#This is the pixels where the cones were seen in the picture used for calibration
#upper 2
upperLeftSrc = [225,260] # upper left
upperRightSrc = [395,260] # upper right
#lower 2
lowerLeftSrc = [155,290] # lower left
lowerRightSrc = [465,290] # lower right

#This is how far forward we will try to approximate distances in the image, in real world centimeters
#The following two are original
#visionWidth = 150
#visionDepth = 220

visionWidth = 100
visionDepth = 220

#Here we define the conversion factors from pixels in the transformed image, to real world centimeters
pixelsPerWidthUnit = COLS/visionWidth
pixelsPerDepthUnit = ROWS/visionDepth

# Here, we enter the measured real-corld distances to the corresponding cones above
# DISTANCES ARE MEASURED FROM THE VERY FRONT OF THE CAR!!! THIS MAY NOT BE OPTIMAL!
#upper 2
upperLeftLen = [pixelsPerWidthUnit*(visionWidth/2-20),pixelsPerDepthUnit*(visionDepth-100)] # upper left
upperRightLen = [pixelsPerWidthUnit*(visionWidth/2+20),pixelsPerDepthUnit*(visionDepth-100)] # upper right
#lower 2
lowerLeftLen = [pixelsPerWidthUnit*(visionWidth/2-20),pixelsPerDepthUnit*(visionDepth-50)] # upper left
lowerRightLen = [pixelsPerWidthUnit*(visionWidth/2+20),pixelsPerDepthUnit*(visionDepth-50)] # upper left

# Here, the above entered empirical data, as well as desired field of vision, is used to find a suitable 
# transformation matrix
src_points = numpy.float32([upperLeftSrc,upperRightSrc,lowerLeftSrc,lowerRightSrc])
dst_points = numpy.float32([upperLeftLen,upperRightLen,lowerLeftLen,lowerRightLen])
projective_matrix = cv2.getPerspectiveTransform(src_points,dst_points)
#Calibration over!

# This can be used if we want the car to slow down in curves
# If you want constant velocity, set the following three to the same value
referencePedalPosition = 1
maxPedalAim = 1
minPedalAim = 1

actualMaximumPedalPosition = 0.1

pedalPosition = referencePedalPosition

# If you want the car to slow down in curves,
# these parameters help stabilize changes in pedal position

# the NEUTRALITY weight pushes the pedal position towards a pedal position
# considered neutral

# The STABILITY weight gives the pedal position inertia, so it will not change too fast

# The CONTROL weight gives responsiveness, so that the car actually reacts to the new situation

# These should sum to 1 (unless you REALLY know what you're doing)

neutralityPedalPositionWeight = .02
stabilityPedalPositionWeight = 0
controlPedalPositionWeight = 1 - neutralityPedalPositionWeight - stabilityPedalPositionWeight

# The following is meant to stabilize steering

# the NEUTRALITY weight pushes the steering angle towards an angle
# considered neutral. This is supposed to act as a counter-weight to over-steering

# The STABILITY weight gives the steering angle inertia, so it will not change too fast

# The CONTROL weight gives responsiveness, so that the car actually reacts to the new situation

neutralityWeight = 0
remainingWeight = 1-neutralityWeight
stabilityWeight = remainingWeight*0.2
controlWeight = 1- neutralityWeight - stabilityWeight

# Here is the reference point
# It represents a neutral position to steer towards, and is meant to act as a counterweight
# to over-steering

# Aim-points are also calculated in relationship to the reference point in the first place;
# the mean value of the yellow and blue pixels closest to the reference point
# is used as aim-point
referencePointDepthInCm = 30
referencePoint = numpy.array((ROWS - referencePointDepthInCm*pixelsPerDepthUnit,COLS/2))
aimpointDepth = (ROWS - referencePoint[0])/pixelsPerDepthUnit
aimpointWidth = (-(referencePoint[1]-COLS/2))/pixelsPerWidthUnit

steeringAngleSign = 1
steeringAngleAmplitude = 0
#numpy.set_printoptions(threshold=numpy.nan)

breakingTolerance = 1.1
stoppingTolerance = 0.8

distanceToObject = distances["front"] 
previousDistanceToObject = distanceToObject
distanceReadingStartTime = time.time()

depthCorrectionTerm = 10
angleCorrectionFactor = .25

stabilityWeightCones = .3
controlWeightConePositions = 1-stabilityWeightCones
yellowConePositions = []
blueConePositions = []
orangeConePositionsLeft = []
orangeConePositionsRight = []

paramsForBlobDetection = cv2.SimpleBlobDetector_Params()
paramsForBlobDetection.filterByArea = True
paramsForBlobDetection.minArea = 30
paramsForBlobDetection.maxArea = 1000000
paramsForBlobDetection.filterByCircularity = False
#paramsForBlobDetection.minCircularity = 0.5

detector = cv2.SimpleBlobDetector_create(paramsForBlobDetection)
carInSight = False
carPreviouslyDetectedToTheRight = True
carDetectionStart = time.time()

################################################################################
# Main loop to process the next image frame coming in.
while True:
    # This try/catch block is meant to return the car to 0-degree steering and 0 velocity
    # when the program is interrupted but it doesn't always work
    #try:
    # Wait for next notification.
    
    # The following try/except-block is an uggly hack that seems to help the outer try/catch-block
    # The outer try/catch block is meant to return the car to 0-degree steering and 0 velocity
    # but it doesn't always work
    a = "1"
    try:
        cond.Z()
    except:
        a = "aaa"
    b = int(a)
    # print ("Received new frame.")

    # Lock access to shared memory.
    mutex.acquire()
    # Attach to shared memory.
    shm.attach()
    # Read shared memory into own buffer.
    buf = shm.read()
    # Detach to shared memory.
    shm.detach()
    # Unlock access to shared memory.
    mutex.release()

    # Turn buf into img array (640 * 480 * 4 bytes (ARGB)) to be used with OpenCV.
    img = numpy.frombuffer(buf, numpy.uint8).reshape(480, 640, 4)
    img_blobs = img.copy()

    cannyBlob = cv2.Canny(img,30,90,3)
    cannyBlob = DilateAndErode(cannyBlob, dilationIterations = 8, erosionIterations = 15)
    cannyBlob[:200,:]=0
    cannyBlob[:,0:320]=0
    keypoints = detector.detect(cannyBlob)
    pointCoordinateList = []

    for keypoint in keypoints:
        pointCoordinates = (int(keypoint.pt[0]),int(keypoint.pt[1]))
        pointCoordinateList.append(pointCoordinates)

    for pointCoordinates in pointCoordinateList:
        cv2.rectangle(img_blobs, (pointCoordinates[0]-10,pointCoordinates[1]-10),(pointCoordinates[0]+10,pointCoordinates[1]+10),(0,0,255),2)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray[:240,:]=0
    #img_gray[:,0:320]=0
    maxNeighbourDistance = 40
    maxNumberOfCorners = 20
    minimumCornerQuality = 0.1
    clusterIsCarTol = 8

    corners = cv2.goodFeaturesToTrack(img_gray,maxNumberOfCorners,minimumCornerQuality,10)
    numberOfCorners = corners.shape[0]
    cornerCoordinates = corners[:,0,:]
    cornerDistances = scipy.spatial.distance.pdist(cornerCoordinates)
    adjacencyVector = numpy.zeros(cornerDistances.shape)
    adjacencyVector[cornerDistances<maxNeighbourDistance]=1
    clusters = GetClusters(adjacencyVector)
    carDetectedToTheRight = False
    if len(clusters)>0:
        biggestClusterSize = max([cluster[1] for cluster in clusters])
        if biggestClusterSize > clusterIsCarTol:
            biggestClusterIndex = numpy.argmax([cluster[1] for cluster in clusters])
            biggestClusterIndeces = list(clusters[biggestClusterIndex][0])
            coordinatesOfBiggestCluster = cornerCoordinates[biggestClusterIndeces,:]
            biggestClusterTop = int(numpy.max(coordinatesOfBiggestCluster[:,1]))
            biggestClusterBottom = int(numpy.min(coordinatesOfBiggestCluster[:,1]))
            biggestClusterRight = int(numpy.max(coordinatesOfBiggestCluster[:,0]))
            biggestClusterLeft = int(numpy.min(coordinatesOfBiggestCluster[:,0]))
            if biggestClusterRight > COLS/2:
                carDetectedToTheRight = True
    else:
        biggestClusterSize = 0
    meanDistance = numpy.mean(cornerDistances)
    corners = numpy.int0(corners)


    #img_blobs = cv2.drawKeypoints(img, keypoints, numpy.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    ############################################################################
    # TODO: Add some image processing logic here.

    # All behavior currently relies on the mode choice "colormap"
    if modeChoice == 'edges':
        # Edge detection
        canny = cv2.Canny(img, 30, 90, 3)
        #cv2.imshow("edges",canny)
        lines = cv2.HoughLines(canny, 1, math.pi/180, 150, 0, 0)
        hough = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        for i in range(0,len(lines)):
            print(lines.shape)
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b *rho

            pt1x = int(round(x0 + 1000*(-b)))
            pt1y = int(round(y0 + 1000*(a)))
            pt2x = int(round(x0 - 1000*(-b)))
            pt2y = int(round(y0 - 1000*(a)))
            pt1 = (pt1x, pt1y)
            pt2 = (pt2x, pt2y)
            cv2.line(hough, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.waitKey(2);
    if modeChoice == 'raw':
        # The following example is adding a red rectangle and displaying the result.
        cv2.rectangle(img, (220,255), (230,265), (0,0,255), 2)
        cv2.rectangle(img, (150,285), (160,295), (0,0,255), 2)
        cv2.rectangle(img, (390,255), (400,265), (0,0,255), 2)
        cv2.rectangle(img, (460,285), (470,295), (0,0,255), 2)
        cv2.waitKey(2)
    if modeChoice == 'colormap':

        # Here, the image is transformed to make pixel distances correspond to real-world distances
        warpedImage = cv2.warpPerspective(img, projective_matrix, (COLS,ROWS))
        canny = cv2.Canny(warpedImage, 30, 90, 3)

        #Color filtering
        blueCones = ColorFilterBlue(warpedImage)
        # Dilate and erode
        erodeBlue = DilateAndErode(blueCones, dilationIterations = 10, erosionIterations = 10)

        #Now for yellow
        yellowCones = ColorFilterYellow(warpedImage)
        # Dilate and erode
        erodeYellow = DilateAndErode(yellowCones, dilationIterations = 10, erosionIterations = 10)

        #Now for orange
        orangeCones = ColorFilterOrange(warpedImage)
        # Dilate and erode
        erodeOrange = DilateAndErode(orangeCones, dilationIterations = 10, erosionIterations = 10)

        # Remove irrelevent information (so we don't detect cables on the car)
        lastRelevantIndex = 570
        blueConeMap = erodeBlue[:lastRelevantIndex,:]
        yellowConeMap = erodeYellow[:lastRelevantIndex,:]
        orangeConeMap = erodeOrange[:lastRelevantIndex,:]

        orangeConeMapLeft = orangeConeMap.copy()
        orangeConeMapRight = orangeConeMap.copy()

        orangeConeMapLeft[:,int(COLS/2):] = 0
        #orangeConeMapLeft[:,:int(COLS/3.5)] = 0
        orangeConeMapRight[:,:int(COLS/2)] = 0
        orangeConeMapRight[:,(int(COLS/2)+int(COLS/4.2)):] = 0

        blueConeMap = blueConeMap + orangeConeMapRight
        yellowConeMap = yellowConeMap + orangeConeMapLeft

        yellowConePositionsAPriori = DetectCones(yellowConeMap, pixelsPerWidthUnit, pixelsPerDepthUnit)
        blueConePositionsAPriori = DetectCones(blueConeMap, pixelsPerWidthUnit, pixelsPerDepthUnit)
        orangeConePositionsLeftAPriori = DetectCones(orangeConeMapLeft, pixelsPerWidthUnit, pixelsPerDepthUnit)
        orangeConePositionsRightAPriori = DetectCones(orangeConeMapRight, pixelsPerWidthUnit, pixelsPerDepthUnit)

        if len(yellowConePositionsAPriori)==2:
            if len(yellowConePositions)>0:
                i1Previous = yellowConePositions[0][0]
                j1Previous = yellowConePositions[0][1]
            else:
                i1Previous = yellowConePositionsAPriori[0][0]
                j1Previous = yellowConePositionsAPriori[0][1]
            if len(yellowConePositions)>1:
                i2Previous = yellowConePositions[0][0]
                j2Previous = yellowConePositions[0][1]
            else:
                i2Previous = yellowConePositionsAPriori[0][0]
                j2Previous = yellowConePositionsAPriori[0][1]
            i1APriori = yellowConePositionsAPriori[0][0]
            j1APriori = yellowConePositionsAPriori[0][1]
            i2APriori = yellowConePositionsAPriori[1][0]
            j2APriori = yellowConePositionsAPriori[1][1]

            i1 = int(stabilityWeightCones*i1Previous + controlWeightConePositions*i1APriori)
            j1 = int(stabilityWeightCones*j1Previous + controlWeightConePositions*j1APriori)
            i2 = int(stabilityWeightCones*i2Previous + controlWeightConePositions*i2APriori)
            j2 = int(stabilityWeightCones*j2Previous + controlWeightConePositions*j2APriori)
            yellowConePositions = [(i1,j1),(i2,j2)]
        elif len(yellowConePositionsAPriori)==1:
            if len(yellowConePositions)>0:
                i1Previous = yellowConePositions[0][0]
                j1Previous = yellowConePositions[0][1]
            else:
                i1Previous = yellowConePositionsAPriori[0][0]
                j1Previous = yellowConePositionsAPriori[0][1]
            i1APriori = yellowConePositionsAPriori[0][0]
            j1APriori = yellowConePositionsAPriori[0][1]

            i1 = int(stabilityWeightCones*i1Previous + controlWeightConePositions*i1APriori)
            j1 = int(stabilityWeightCones*j1Previous + controlWeightConePositions*j1APriori)
            #i2 = int(i2Previous)
            #j2 = int(j2Previous)
            yellowConePositions = [(i1,j1)]
        else:
            #i1 = int(i1Previous)
            #j1 = int(j1Previous)
            #i2 = int(i2Previous)
            #j2 = int(j2Previous)
            yellowConePositions = []

        if len(blueConePositionsAPriori)==2:
            if len(blueConePositions)>0:
                i1Previous = blueConePositions[0][0]
                j1Previous = blueConePositions[0][1]
            else:
                i1Previous = blueConePositionsAPriori[0][0]
                j1Previous = blueConePositionsAPriori[0][1]
            if len(blueConePositions)>1:
                i2Previous = blueConePositions[0][0]
                j2Previous = blueConePositions[0][1]
            else:
                i2Previous = blueConePositionsAPriori[0][0]
                j2Previous = blueConePositionsAPriori[0][1]
            i1APriori = blueConePositionsAPriori[0][0]
            j1APriori = blueConePositionsAPriori[0][1]
            i2APriori = blueConePositionsAPriori[1][0]
            j2APriori = blueConePositionsAPriori[1][1]
            i1 = int(stabilityWeightCones*i1Previous + controlWeightConePositions*i1APriori)
            j1 = int(stabilityWeightCones*j1Previous + controlWeightConePositions*j1APriori)
            i2 = int(stabilityWeightCones*i2Previous + controlWeightConePositions*i2APriori)
            j2 = int(stabilityWeightCones*j2Previous + controlWeightConePositions*j2APriori)
            blueConePositions = [(i1,j1),(i2,j2)]
        elif len(blueConePositionsAPriori)==1:
            if len(blueConePositions)>0:
                i1Previous = blueConePositions[0][0]
                j1Previous = blueConePositions[0][1]
            else:
                i1Previous = blueConePositionsAPriori[0][0]
                j1Previous = blueConePositionsAPriori[0][1]
            i1APriori = blueConePositionsAPriori[0][0]
            j1APriori = blueConePositionsAPriori[0][1]
            i1 = int(stabilityWeightCones*i1Previous + controlWeightConePositions*i1APriori)
            j1 = int(stabilityWeightCones*j1Previous + controlWeightConePositions*j1APriori)
            #i2 = int(i2Previous)
            #j2 = int(j2Previous)
            blueConePositions = [(i1,j1)]
        else:
            #i1 = int(i1Previous)
            #j1 = int(j1Previous)
            #i2 = int(i2Previous)
            #j2 = int(j2Previous)
            blueConePositions = []

        if len(orangeConePositionsLeftAPriori)==2:
            if len(orangeConePositionsLeft)>0:
                i1Previous = orangeConePositionsLeft[0][0]
                j1Previous = orangeConePositionsLeft[0][1]
            else:
                i1Previous = orangeConePositionsLeftAPriori[0][0]
                j1Previous = orangeConePositionsLeftAPriori[0][1]
            if len(orangeConePositionsLeft)>1:
                i2Previous = orangeConePositionsLeft[0][0]
                j2Previous = orangeConePositionsLeft[0][1]
            else:
                i2Previous = orangeConePositionsLeftAPriori[0][0]
                j2Previous = orangeConePositionsLeftAPriori[0][1]
            i1APriori = orangeConePositionsLeftAPriori[0][0]
            j1APriori = orangeConePositionsLeftAPriori[0][1]
            i2APriori = orangeConePositionsLeftAPriori[1][0]
            j2APriori = orangeConePositionsLeftAPriori[1][1]
            i1 = int(stabilityWeightCones*i1Previous + controlWeightConePositions*i1APriori)
            j1 = int(stabilityWeightCones*j1Previous + controlWeightConePositions*j1APriori)
            i2 = int(stabilityWeightCones*i2Previous + controlWeightConePositions*i2APriori)
            j2 = int(stabilityWeightCones*j2Previous + controlWeightConePositions*j2APriori)
            orangeConePositionsLeft = [(i1,j1),(i2,j2)]
        elif len(orangeConePositionsLeftAPriori)==1:
            if len(orangeConePositionsLeft)>0:
                i1Previous = orangeConePositionsLeft[0][0]
                j1Previous = orangeConePositionsLeft[0][1]
            else:
                i1Previous = orangeConePositionsLeftAPriori[0][0]
                j1Previous = orangeConePositionsLeftAPriori[0][1]
            i1APriori = orangeConePositionsLeftAPriori[0][0]
            j1APriori = orangeConePositionsLeftAPriori[0][1]
            i1 = int(stabilityWeightCones*i1Previous + controlWeightConePositions*i1APriori)
            j1 = int(stabilityWeightCones*j1Previous + controlWeightConePositions*j1APriori)
            #i2 = int(i2Previous)
            #j2 = int(j2Previous)
            orangeConePositionsLeft = [(i1,j1)]
        else:
            #i1 = int(i1Previous)
            #j1 = int(j1Previous)
            #i2 = int(i2Previous)
            #j2 = int(j2Previous)
            orangeConePositionsLeft = []

        if len(orangeConePositionsRightAPriori)==2:
            if len(orangeConePositionsRight)>0:
                i1Previous = orangeConePositionsRight[0][0]
                j1Previous = orangeConePositionsRight[0][1]
            else:
                i1Previous = orangeConePositionsRightAPriori[0][0]
                j1Previous = orangeConePositionsRightAPriori[0][1]
            if len(orangeConePositionsRight)>1:
                i2Previous = orangeConePositionsRight[0][0]
                j2Previous = orangeConePositionsRight[0][1]
            else:
                i2Previous = orangeConePositionsRightAPriori[0][0]
                j2Previous = orangeConePositionsRightAPriori[0][1]
            i1APriori = orangeConePositionsRightAPriori[0][0]
            j1APriori = orangeConePositionsRightAPriori[0][1]
            i2APriori = orangeConePositionsRightAPriori[1][0]
            j2APriori = orangeConePositionsRightAPriori[1][1]
            i1 = int(stabilityWeightCones*i1Previous + controlWeightConePositions*i1APriori)
            j1 = int(stabilityWeightCones*j1Previous + controlWeightConePositions*j1APriori)
            i2 = int(stabilityWeightCones*i2Previous + controlWeightConePositions*i2APriori)
            j2 = int(stabilityWeightCones*j2Previous + controlWeightConePositions*j2APriori)
            orangeConePositionsRight = [(i1,j1),(i2,j2)]
        elif len(orangeConePositionsRightAPriori)==1:
            if len(orangeConePositionsRight)>0:
                i1Previous = orangeConePositionsRight[0][0]
                j1Previous = orangeConePositionsRight[0][1]
            else:
                i1Previous = orangeConePositionsRightAPriori[0][0]
                j1Previous = orangeConePositionsRightAPriori[0][1]
            i1APriori = orangeConePositionsRightAPriori[0][0]
            j1APriori = orangeConePositionsRightAPriori[0][1]
            i1 = int(stabilityWeightCones*i1Previous + controlWeightConePositions*i1APriori)
            j1 = int(stabilityWeightCones*j1Previous + controlWeightConePositions*j1APriori)
            #i2 = int(i2Previous)
            #j2 = int(j2Previous)
            orangeConePositionsRight = [(i1,j1)]
        else:
            #i1 = int(i1Previous)
            #j1 = int(j1Previous)
            #i2 = int(i2Previous)
            #j2 = int(j2Previous)
            orangeConePositionsRight = []



        blueConeRealWorldPositions = []
        for pixelCone in blueConePositions:
            i = (ROWS - pixelCone[0])/pixelsPerDepthUnit
            j = -((pixelCone[1] - COLS/2))/pixelsPerWidthUnit
            cone = (i,j)
            blueConeRealWorldPositions.append(cone)

        yellowConeRealWorldPositions = []
        for pixelCone in yellowConePositions:
            i = (ROWS - pixelCone[0])/pixelsPerDepthUnit
            j = -((pixelCone[1] - COLS/2))/pixelsPerWidthUnit
            cone = (i,j)
            yellowConeRealWorldPositions.append(cone)

        orangeConeRealWorldPositionsLeft = []
        for pixelCone in orangeConePositionsLeft:
            i = (ROWS - pixelCone[0])/pixelsPerDepthUnit
            j = -((pixelCone[1] - COLS/2))/pixelsPerWidthUnit
            cone = (i,j)
            orangeConeRealWorldPositionsLeft.append(cone)

        orangeConeRealWorldPositionsRight = []
        for pixelCone in orangeConePositionsRight:
            i = (ROWS - pixelCone[0])/pixelsPerDepthUnit
            j = -((pixelCone[1] - COLS/2))/pixelsPerWidthUnit
            cone = (i,j)
            orangeConeRealWorldPositionsRight.append(cone)

        blueConeInVision = len(blueConePositions)>0
        yellowConeInVision = len(yellowConePositions)>0
        orangeConeLeftInVision = len(orangeConePositionsLeft)>0
        orangeConeRightInVision = len(orangeConePositionsRight)>0

        if orangeConeLeftInVision and orangeConeRightInVision:
            if yellowConeInVision or blueConeInVision:
                # Find the closest orange cone
                firstLeftDepth = orangeConeRealWorldPositionsLeft[0][0]
                firstLeftWidth = orangeConeRealWorldPositionsLeft[0][1]
                firstRightDepth = orangeConeRealWorldPositionsRight[0][0]
                firstRightWidth = orangeConeRealWorldPositionsRight[0][1]
                leftDistance = math.sqrt(firstLeftDepth**2+firstLeftWidth**2)
                rightDistance = math.sqrt(firstRightDepth**2+firstRightWidth**2)
                if leftDistance>rightDistance:
                    closestOrangeConeDistance = rightDistance
                else:
                    closestOrangeConeDistance = leftDistance
                # Find the closest blue cone
                if blueConeInVision:
                    firstBlueDepth = blueConeRealWorldPositions[0][0]
                    firstBlueWidth = blueConeRealWorldPositions[0][1]
                    blueDistance = math.sqrt(firstBlueDepth**2+firstBlueWidth**2)
                else:
                    blueDistance = 1000000
                # Find the closest yellow cone
                if yellowConeInVision:
                    firstYellowDepth = yellowConeRealWorldPositions[0][0]
                    firstYellowWidth = yellowConeRealWorldPositions[0][1]
                    yellowDistance = math.sqrt(firstYellowDepth**2+firstYellowWidth**2)
                else:
                    yellowDistance = 1000000
                closestNonOrangeConeDistance = max(blueDistance,yellowDistance)
                useOrangeCones = closestNonOrangeConeDistance>closestOrangeConeDistance
            else:
                useOrangeCones = True
        else:
            useOrangeCones = False

        useOrangeCones = False
        if not useOrangeCones:
            if len(yellowConePositions)>0:
                yellowCone1Position = yellowConePositions[0]
                if len(yellowConePositions)>1:
                    yellowCone2Position = yellowConePositions[1]

                cv2.rectangle(warpedImage, (yellowCone1Position[1]-5,yellowCone1Position[0]-5), (yellowCone1Position[1]+5,yellowCone1Position[0]+5), (0,255,0), 2)
                if len(yellowConePositions)>1:
                    cv2.rectangle(warpedImage, (yellowCone2Position[1]-5,yellowCone2Position[0]-5), (yellowCone2Position[1]+5,yellowCone2Position[0]+5), (0,255,0), 2)

            if len(blueConePositions)>0:
                blueCone1Position = blueConePositions[0]
                if len(blueConePositions)>1:
                    blueCone2Position = blueConePositions[1]

                cv2.rectangle(warpedImage, (blueCone1Position[1]-5,blueCone1Position[0]-5), (blueCone1Position[1]+5,blueCone1Position[0]+5), (0,255,0), 2)
                if len(blueConePositions)>1:
                    cv2.rectangle(warpedImage, (blueCone2Position[1]-5,blueCone2Position[0]-5), (blueCone2Position[1]+5,blueCone2Position[0]+5), (0,255,0), 2)

        """
        else:
            if len(orangeConePositionsRight)>0:
                orangeCone1PositionRight = orangeConePositionsRight[0]
                if len(orangeConePositionsRight)>1:
                    orangeCone2PositionRight = orangeConePositionsRight[1]

                cv2.rectangle(warpedImage, (orangeCone1PositionRight[1]-5,orangeCone1PositionRight[0]-5), (orangeCone1PositionRight[1]+5,orangeCone1PositionRight[0]+5), (0,255,0), 2)
                if len(blueConePositions)>1:
                    cv2.rectangle(warpedImage, (orangeCone2PositionRight[1]-5,orangeCone2PositionRight[0]-5), (orangeCone2PositionRight[1]+5,orangeCone2PositionRight[0]+5), (0,255,0), 2)

            if len(orangeConePositionsLeft)>0:
                orangeCone1PositionLeft = orangeConePositionsLeft[0]
                if len(orangeConePositionsLeft)>1:
                    orangeCone2PositionLeft = orangeConePositionsLeft[1]

                cv2.rectangle(warpedImage, (orangeCone1PositionLeft[1]-5,orangeCone1PositionLeft[0]-5), (orangeCone1PositionLeft[1]+5,orangeCone1PositionLeft[0]+5), (0,255,0), 2)
                if len(blueConePositions)>1:
                    cv2.rectangle(warpedImage, (orangeCone2PositionLeft[1]-5,orangeCone2PositionLeft[0]-5), (orangeCone2PositionLeft[1]+5,orangeCone2PositionLeft[0]+5), (0,255,0), 2)

        """

        """
        if (blueConeInVision):
            if widthDistanceToClosestBlueCone > 0:
                print("\n\nThe closest blue cone is {} cm forward\nThe closest blue cone is {} cm to the left\n\n".format(round(depthDistanceToClosestBlueCone), round(widthDistanceToClosestBlueCone)))
            else:
                print("\n\nThe closest blue cone is {} cm forward\nThe closest blue cone is {} cm to the right\n\n".format(round(depthDistanceToClosestBlueCone), -round(widthDistanceToClosestBlueCone)))

        if (yellowConeInVision):
            if widthDistanceToClosestYellowCone > 0:
                print("\n\nThe closest Yellow cone is {} cm forward\nThe closest Yellow cone is {} cm to the left\n\n".format(round(depthDistanceToClosestYellowCone), round(widthDistanceToClosestYellowCone)))
            else:
                print("\n\nThe closest Yellow cone is {} cm forward\nThe closest Yellow cone is {} cm to the right\n\n".format(round(depthDistanceToClosestYellowCone), -round(widthDistanceToClosestYellowCone)))
        """      

        # Set a priori aimpoint between the blue and yellow pixels closest to the reference point
        if useOrangeCones:
            firstLeftDepth = orangeConeRealWorldPositionsLeft[0][0]
            secondLeftDepth = orangeConeRealWorldPositionsLeft[-1][0]

            firstRightDepth = orangeConeRealWorldPositionsRight[0][0]
            secondRightDepth = orangeConeRealWorldPositionsRight[-1][0]

            firstTrajectoryDepth = (firstLeftDepth+firstRightDepth)/2
            secondTrajectoryDepth = (secondLeftDepth+secondRightDepth)/2

            firstLeftWidth = orangeConeRealWorldPositionsLeft[0][1]
            secondLeftWidth = orangeConeRealWorldPositionsLeft[-1][1]

            firstRightWidth = orangeConeRealWorldPositionsRight[0][1]
            secondRightWidth = orangeConeRealWorldPositionsRight[-1][1]

            firstTrajectoryWidth = (firstLeftWidth+firstRightWidth)/2
            secondTrajectoryWidth = (secondLeftWidth+secondRightWidth)/2
            # ALL RELEVANT INFORMATION FOR UNDERSTANDING STEERING IS FOUND IN COMMENTS FOR THIS
            # FIRST IF CASE
            # Depth represents distance from the point to the front of the car
            aPrioriAimPointDepth = (firstTrajectoryDepth + secondTrajectoryDepth)/2
            referencePointDepth = (ROWS - referencePoint[0])/pixelsPerDepthUnit
            # Width represents real world distance from the point to the center of vision
            aPrioriAimPointWidth = (firstTrajectoryWidth+secondTrajectoryWidth)/2
            referencePointWidth = (-(referencePoint[1]-COLS/2))/pixelsPerWidthUnit
            # The aim point is set to a weighted average between:
            # the a priori aim point (GIVES CONTROL AND RESPONSIVENESS)
            # the previous aim point (GIVES INERTIA AND STABILITY IN STEERING)
            # the reference point (COUNTER-WEIGHTS OVER-STEERING)
            aimpointDepth = controlWeight*aPrioriAimPointDepth + stabilityWeight*aimpointDepth + neutralityWeight*referencePointDepth + depthCorrectionTerm
            aimpointWidth = controlWeight*aPrioriAimPointWidth + stabilityWeight*aimpointWidth + neutralityWeight*referencePointWidth
            # using trigonometry to turn the aim point into a steering angle
            aimPointAngle = numpy.arcsin(aimpointWidth/aimpointDepth)*angleCorrectionFactor
            steeringAngleSign = numpy.sign(aimPointAngle)
            steeringAngleAmplitude = min(MAXIMUM_STEERING_ANGLE_AMPLITUDE,abs(aimPointAngle))
            # The following two are only used for GUI reasons, to know where to plot the square representing
            # the aim point

        # If only yellow pixels are visible, steer a bit to the right
        elif blueConeInVision and yellowConeInVision:
            firstBlueDepth = blueConeRealWorldPositions[0][0]
            secondBlueDepth = blueConeRealWorldPositions[-1][0]

            firstYellowDepth = yellowConeRealWorldPositions[0][0]
            secondYellowDepth = yellowConeRealWorldPositions[-1][0]

            firstTrajectoryDepth = (firstBlueDepth+firstYellowDepth)/2
            secondTrajectoryDepth = (secondBlueDepth+secondYellowDepth)/2

            firstBlueWidth = blueConeRealWorldPositions[0][1]
            secondBlueWidth = blueConeRealWorldPositions[-1][1]

            firstYellowWidth = yellowConeRealWorldPositions[0][1]
            secondYellowWidth = yellowConeRealWorldPositions[-1][1]

            firstTrajectoryWidth = (firstBlueWidth+firstYellowWidth)/2
            secondTrajectoryWidth = (secondBlueWidth+secondYellowWidth)/2
            # ALL RELEVANT INFORMATION FOR UNDERSTANDING STEERING IS FOUND IN COMMENTS FOR THIS
            # FIRST IF CASE
            # Depth represents distance from the point to the front of the car
            aPrioriAimPointDepth = (firstTrajectoryDepth + secondTrajectoryDepth)/2
            referencePointDepth = (ROWS - referencePoint[0])/pixelsPerDepthUnit
            # Width represents real world distance from the point to the center of vision
            aPrioriAimPointWidth = (firstTrajectoryWidth+secondTrajectoryWidth)/2
            referencePointWidth = (-(referencePoint[1]-COLS/2))/pixelsPerWidthUnit
            # The aim point is set to a weighted average between:
            # the a priori aim point (GIVES CONTROL AND RESPONSIVENESS)
            # the previous aim point (GIVES INERTIA AND STABILITY IN STEERING)
            # the reference point (COUNTER-WEIGHTS OVER-STEERING)
            aimpointDepth = controlWeight*aPrioriAimPointDepth + stabilityWeight*aimpointDepth + neutralityWeight*referencePointDepth + depthCorrectionTerm
            aimpointWidth = controlWeight*aPrioriAimPointWidth + stabilityWeight*aimpointWidth + neutralityWeight*referencePointWidth
            # using trigonometry to turn the aim point into a steering angle
            aimPointAngle = numpy.arcsin(aimpointWidth/aimpointDepth)*angleCorrectionFactor
            steeringAngleSign = numpy.sign(aimPointAngle)
            steeringAngleAmplitude = min(MAXIMUM_STEERING_ANGLE_AMPLITUDE,abs(aimPointAngle))
            # The following two are only used for GUI reasons, to know where to plot the square representing
            # the aim point

        # If only yellow pixels are visible, steer a bit to the right
        elif yellowConeInVision:
            aimpointDepth = controlWeight*(ROWS - referencePoint[0])/pixelsPerDepthUnit + stabilityWeight*aimpointDepth + neutralityWeight*(ROWS - referencePoint[0])/pixelsPerDepthUnit + depthCorrectionTerm
            aimpointWidth = -controlWeight*20 + stabilityWeight*aimpointWidth + neutralityWeight*(-(referencePoint[1]-COLS/2))/pixelsPerWidthUnit
            aimPointAngle = numpy.arcsin(aimpointWidth/aimpointDepth)*angleCorrectionFactor
            steeringAngleSign = numpy.sign(aimPointAngle)
            steeringAngleAmplitude = min(MAXIMUM_STEERING_ANGLE_AMPLITUDE,abs(aimPointAngle))

        # If only blue pixels are visible, steer a bit to the left
        elif blueConeInVision:
            aimpointDepth = controlWeight*(ROWS - referencePoint[0])/pixelsPerDepthUnit + stabilityWeight*aimpointDepth + neutralityWeight*(ROWS - referencePoint[0])/pixelsPerDepthUnit + depthCorrectionTerm
            aimpointWidth = controlWeight*20 + stabilityWeight*aimpointWidth + neutralityWeight*(-(referencePoint[1]-COLS/2))/pixelsPerWidthUnit
            aimPointAngle = numpy.arcsin(aimpointWidth/aimpointDepth)*angleCorrectionFactor
            steeringAngleSign = numpy.sign(aimPointAngle)
            steeringAngleAmplitude = min(MAXIMUM_STEERING_ANGLE_AMPLITUDE,abs(aimPointAngle))

        # If no blue or yellow pixels are visible, only steer a bit towards the reference point
        else:
            if stabilityWeight+neutralityWeight>0:
                aimpointDepth = stabilityWeight*aimpointDepth/(stabilityWeight+neutralityWeight) + neutralityWeight*(ROWS - referencePoint[0])/(stabilityWeight+neutralityWeight)/pixelsPerDepthUnit + depthCorrectionTerm
                aimpointWidth = stabilityWeight*aimpointWidth/(stabilityWeight+neutralityWeight) + neutralityWeight*(-(referencePoint[1]-COLS/2))/(stabilityWeight+neutralityWeight)/pixelsPerDepthUnit

                aimPointAngle = numpy.arcsin(aimpointWidth/aimpointDepth)*angleCorrectionFactor
                steeringAngleSign = numpy.sign(aimPointAngle)
                steeringAngleAmplitude = min(MAXIMUM_STEERING_ANGLE_AMPLITUDE,abs(aimPointAngle))
    aimpointPixelDepth = ROWS - int((aimpointDepth-depthCorrectionTerm)*pixelsPerDepthUnit)
    aimpointPixelWidth = int(COLS/2 - aimpointWidth*pixelsPerWidthUnit)

    # draw a rectangle representing the aim point
    cv2.rectangle(warpedImage, (aimpointPixelWidth-5,aimpointPixelDepth-5), (aimpointPixelWidth+5,aimpointPixelDepth+5), (0,0,255), 2)
    # draw a rectangle representing the reference point
    cv2.rectangle(warpedImage, (int(referencePoint[1]-5),int(referencePoint[0]-5)), (int(referencePoint[1]+5),int(referencePoint[0]+5)), (0,0,255), 2)

        

    ############################################################################
    # Example: Accessing the distance readings.
    #print ("Front = %s" % (str(distances["front"])))
    #print ("Left = %s" % (str(distances["left"])))
    #print ("Right = %s" % (str(distances["right"])))
    #print ("Rear = %s" % (str(distances["rear"])))
    stabilityWeightDistanceReading = 0.1

    distanceReadingStopTime = time.time()
    distanceToObjectAPriori = distances["front"]
    distanceToObject = stabilityWeightDistanceReading*distanceToObject + (1-stabilityWeightDistanceReading)*distanceToObjectAPriori









    objectRelativeVelocity = (distanceToObject - previousDistanceToObject)/(distanceReadingStopTime-distanceReadingStartTime)
    #objectRelativeVelocity = stabilityWeight*objectRelativeVelocity + objectRelativeVelocityAPriori*(1-stabilityWeight)

    previousDistanceToObject = distanceToObject
    distanceReadingStartTime = time.time()

    os.system('clear')
    print("Using orange cones: {}".format(useOrangeCones))
    print("Distance: {}".format(distanceToObject))
    print("Relative velocity: {}".format(objectRelativeVelocity))
    print("mean distance: {}".format(meanDistance))
    print("Biggest cluster size: {}".format(biggestClusterSize))

    ############################################################################
    # Example for creating and sending a message to other microservices; can
    # be removed when not needed.
    angleReading = opendlv_standard_message_set_v0_9_6_pb2.opendlv_proxy_AngleReading()
    angleReading.angle = 123.45

    # 1038 is the message ID for opendlv.proxy.AngleReading
    session.send(1038, angleReading.SerializeToString());

    ############################################################################
    # Steering and acceleration/decelration.
    #
    # Uncomment the following lines to steer; range: +38deg (left) .. -38deg (right).
    # Value groundSteeringRequest.groundSteering must be given in radians (DEG/180. * PI).
    groundSteeringRequest = opendlv_standard_message_set_v0_9_6_pb2.opendlv_proxy_GroundSteeringRequest()
    groundSteeringRequest.groundSteering = steeringAngleAmplitude*steeringAngleSign
    session.send(1090, groundSteeringRequest.SerializeToString());

    # Uncomment the following lines to accelerate/decelerate; range: +0.25 (forward) .. -1.0 (backwards).
    # Be careful!
    aPrioriPedalAim = (math.cos(steeringAngleAmplitude*math.pi/maxPedalAim)*.5*(maxPedalAim-minPedalAim) + .5*(maxPedalAim-minPedalAim) + minPedalAim)
    #aPrioriPedalAim = (maxPedalAim - minPedalAim)*((1-(abs(steeringAngleAmplitude)/MAXIMUM_STEERING_ANGLE_AMPLITUDE))**2) + minPedalAim
    if blueConeInVision and yellowConeInVision:
        aPosterioriPedalAim = neutralityPedalPositionWeight*referencePedalPosition + pedalPosition*stabilityPedalPositionWeight + controlPedalPositionWeight*aPrioriPedalAim
    else:
        if (neutralityPedalPositionWeight + stabilityPedalPositionWeight)>0:
            aPosterioriPedalAim = neutralityPedalPositionWeight*referencePedalPosition/(neutralityPedalPositionWeight + stabilityPedalPositionWeight) + pedalPosition*stabilityPedalPositionWeight/(neutralityPedalPositionWeight + stabilityPedalPositionWeight)



    if distanceToObject < breakingTolerance:
        aPrioriPedalAim = (maxPedalAim)/(breakingTolerance-stoppingTolerance)*(distanceToObject-stoppingTolerance)
        aPosterioriPedalAim = max(neutralityPedalPositionWeight*referencePedalPosition+pedalPosition*stabilityPedalPositionWeight + controlPedalPositionWeight*aPrioriPedalAim,0)

    aPosterioriPedalAim = min(aPosterioriPedalAim, actualMaximumPedalPosition)

    pedalPosition = max(aPosterioriPedalAim,MINIMUM_PEDAL_POSITION)
    pedalPosition = min(aPosterioriPedalAim,MAXIMUM_PEDAL_POSITION)

    if carDetectedToTheRight:
        if not carInSight:
            if not carPreviouslyDetectedToTheRight:
                carDetectionStart = time.time()
            else:
                if (time.time()-carDetectionStart) > 0.5:
                    carInSight = True
    else:
        if carInSight:
            if carPreviouslyDetectedToTheRight:
                carDisappearanceStart = time.time()
            else:
                if (time.time()-carDisappearanceStart) > 2:
                    carInSight = False
    carPreviouslyDetectedToTheRight = carDetectedToTheRight
    
    if carInSight and orangeConeLeftInVision and orangeConeRightInVision:
        pedalPosition = 0
    
    #The following is just an extra safety measure
    pedalPositionRequest = opendlv_standard_message_set_v0_9_6_pb2.opendlv_proxy_PedalPositionRequest()
    pedalPositionRequest.position = pedalPosition
    session.send(1086, pedalPositionRequest.SerializeToString());

    #This is all the UI output
    if steeringAngleSign>0:
        print("steering {} degrees to the left".format(round(steeringAngleAmplitude*180/math.pi,2)))
    else:
        print("steering {} degrees to the right".format(round(steeringAngleAmplitude*180/math.pi,2)))
    print("Pedal position: {}".format(round(pedalPosition,2)))

    # Velocity meter for GUI, plotted to the left in the window
    maxPedalPixel = 100
    minPedalPixel = 400
    cv2.rectangle(warpedImage, (30,maxPedalPixel-3), (40,maxPedalPixel), (0,0,255), 2)
    cv2.rectangle(warpedImage, (30,minPedalPixel+3), (40,minPedalPixel), (0,0,255), 2)
    pedalPixel = int((maxPedalPixel - minPedalPixel)/(MAXIMUM_PEDAL_POSITION - 0)*(pedalPosition) + minPedalPixel)
    cv2.rectangle(warpedImage, (30,pedalPixel-3), (40,pedalPixel+3), (0,0,255), 2)

    # Steering angle indicator for GUI
    maximumAngleIndicatorInCm = math.sin(MAXIMUM_STEERING_ANGLE_AMPLITUDE)*referencePointDepthInCm
    angleIndicatorPositionInCm = math.sin(steeringAngleAmplitude)*referencePointDepthInCm
    angleIndicatorPosition = int(-steeringAngleSign*angleIndicatorPositionInCm*pixelsPerWidthUnit + COLS/2)
    minimumAngleIndicatorPosition = int(COLS/2 - maximumAngleIndicatorInCm*pixelsPerWidthUnit)
    maximumAngleIndicatorPosition = int(COLS/2 + maximumAngleIndicatorInCm*pixelsPerWidthUnit)
    cv2.rectangle(warpedImage, (minimumAngleIndicatorPosition,int(referencePoint[0]-5)), (minimumAngleIndicatorPosition+3,int(referencePoint[0]+5)), (0,0,255), 2)
    cv2.rectangle(warpedImage, (maximumAngleIndicatorPosition,int(referencePoint[0]-5)), (maximumAngleIndicatorPosition+3,int(referencePoint[0]+5)), (0,0,255), 2)
    cv2.rectangle(warpedImage, (angleIndicatorPosition-1,int(referencePoint[0]-5)), (angleIndicatorPosition+2,int(referencePoint[0]+5)), (0,0,255), 2)
    if len(clusters)>0:
        if biggestClusterSize > clusterIsCarTol:
            cv2.rectangle(img, (biggestClusterLeft,biggestClusterTop), (biggestClusterRight,biggestClusterBottom), (0,0,255), 2)
    for corner in corners:
            x,y = corner.ravel()
            cv2.circle(img,(x,y),5,255,-1)

    # Show everything
    #cv2.imshow('edges',canny)
    #cv2.imshow('cone_map', yellowConeMap+blueConeMap)
    #cv2.imshow('cone_map_yellow', yellowConeMap)
    #cv2.imshow('cone_map_blue', blueConeMap)
    #cv2.imshow('full_color_map_stable', warpedImage)
    #cv2.imshow('edges', cannyBlob)
    #cv2.imshow('blobs', img_blobs)
    #cv2.imshow('orangeConesRight',orangeConeMapRight+orangeConeMapLeft)
    #cv2.imshow('img', img)
    #cv2.waitKey(2)
    """
    except:
        print("ABORTING SLOWLY")
        while pedalPosition>0:
            try: 
                time.sleep(.2)
                pedalSign = numpy.sign(pedalPosition)
                pedalPosition = pedalSign*max(abs(pedalPosition)-.03,0)
                print("Setting pedal position to {}".format(pedalPosition))
                pedalPositionRequest = opendlv_standard_message_set_v0_9_6_pb2.opendlv_proxy_PedalPositionRequest()
                pedalPositionRequest.position = pedalPosition

                groundSteeringRequest = opendlv_standard_message_set_v0_9_6_pb2.opendlv_proxy_GroundSteeringRequest()
                groundSteeringRequest.groundSteering = 0
                session.send(1090, groundSteeringRequest.SerializeToString());
            except KeyboardInterrupt:
                pass
        print("shutting down application")
        exit()
    """

