import numpy as np
import cv2
import argparse


class faceDetector:
	def __init__(self,faceCascadeFile):
		self.faceCascade = cv2.CascadeClassifier(faceCascadeFile)

	def detect(self,image,scaleFactor = 1.2,minNeighbors = 5,minSize = (50,50)):
		faceRects = self.faceCascade.detectMultiScale(image,scaleFactor = scaleFactor,
		                             minNeighbors = minNeighbors,minSize = minSize,flags = cv2.CASCADE_SCALE_IMAGE)
		return faceRects


class eyeDetetor:
	def __init__(self,eyeCascadeFile):
		self.eyeCascade = cv2.CascadeClassifier(eyeCascadeFile)


	def detect(self,image):
		# allEyeRects = []
		# for (fx,fy,fw,fh) in faceRects:
		# 	# faceROI = image[fy:fy+fh,fx:fx+fw]
		eyeRects = self.eyeCascade.detectMultiScale(image,scaleFactor = 1.1,minNeighbors = 10,
			                                 minSize = (20,20),flags = cv2.CASCADE_SCALE_IMAGE)

		return eyeRects


print "Press q to quit !"

ap = argparse.ArgumentParser()
ap.add_argument("-f","--face",required = True,help = "Path haarCascade file")
#ap.add_argument("-i","--image",required = True,help = "paht to image")

ap.add_argument("-e","--eye",required = True,help = "path to eye haarCascade file")
args = vars(ap.parse_args())


faceCascadeFile  = args["face"]
eyeCascadeFile = args["eye"]

	
camera = cv2.VideoCapture(0)
while True:
	(_,frame) = camera.read()
	
	(h,w) = frame.shape[:2]
	# print "h = {},w = {}".format(h,w)
	r = 480/h
	newsize = (480,int(w*r))
	newFrame = cv2.resize(frame,newsize,interpolation = cv2.INTER_AREA)

	gray = cv2.cvtColor(newFrame,cv2.COLOR_BGR2GRAY)
	cv2.equalizeHist(gray,gray)
	fd = faceDetector(faceCascadeFile)
	faceRects  = fd.detect(gray)

	# print "Found {} faces".format(len(faceRects))

	ed = eyeDetetor(eyeCascadeFile)

	for (x,y,w,h) in faceRects:
		# draw face
		cv2.rectangle(newFrame,(x,y),(x+w,y+h+30),(0,255,0),2)

		faceROI = gray[y:y+h,x:x+w]
		eyeRects = ed.detect(faceROI)

		# draw eye
		eyeNum = 0
		for (ex,ey,ew,eh) in eyeRects:
			eyePt1 = (x+ex,y+ey)
			eyePt2 = (x+ex+ew,y+ey+eh)
			cv2.rectangle(newFrame,eyePt1,eyePt2,(0,255,0),2)
			eyeNum += 1
			if eyeNum == 2:
				break

	
	cv2.imshow("face detect",newFrame)

	if cv2.waitKey(1) & 0XFF == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()

