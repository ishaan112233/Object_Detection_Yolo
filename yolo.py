import cv2 
import numpy as np
import argparse #For Parsing Arguments
import os
import time

arg = argparse.ArgumentParser()
arg.add_argument("-i","--image",required=True,help="path to input image")
# arg.add_argument("-n","--name",required=True)
arg.add_argument("-y","--yolo",required=True,help="path to YOLO directory")
arg.add_argument("-c","--confidence",type=float,default=0.5,help="minimum probability to filter weak detections")
arg.add_argument("-t","--threshold",type=float,default=0.3,help="threshold when applying non-maxima supression")
args = vars(arg.parse_args()) # Return a Dict 

# print(args)
#Loading Coco Class names on which our model was trained
labelspath = os.path.sep.join([args["yolo"],"coco.names"]) 
labels = open(labelspath).read().strip().split("\n") #Strip: Removing Leading and Trailing spaces

#Initialize a list of colors to represent each class label
np.random.seed(42) # ?
COLORS = np.random.randint(0,255,size=(len(labels),3),dtype='uint8') #(len(labels),3)-> Matrix Shape

#Loading Yolo weight and config files
weight_path = os.path.sep.join([args["yolo"],"yolov3.weights"])
config_path = os.path.sep.join([args["yolo"],"yolov3.cfg"])

#Loading yolo trained on 80:Classes(Coco dataset)
print("[INFO]Loading Yolo from disk")
net = cv2.dnn.readNetFromDarknet(config_path,weight_path)

#Loading Image and getting it's dimensions
image = cv2.imread(args["image"])
h,w = image.shape[:2]

#Determine output layer names from yolo
ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]
'''
				OR
	To get names of unconnected layers use 
	ln = net.getUnconnectedOutLayersNames()

'''
# Creating a blob object from image
# This method creates 4-dimensional blob from input images
# blob = cv.dnn.blobFromImage(image, scalefactor, size, mean, swapRB, crop)
# swapRB: for swapping 1 and 3 channel from 3 channel image (Boolean)
blob = cv2.dnn.blobFromImage(image,1/255.0,(416,416),swapRB=True,crop=False)

#Setting Input as blob
net.setInput(blob)

#Making a forward pass of yolo from blob giving boundry boxes and probabilities
start = time.time()
layerOutput = net.forward(ln)
end = time.time()

print("[INFO] Yolo took {:.6f} seconds".format(end-start))

#Initializing list of boundry-box, confidence, class_name
boxes = []
confidences = []
class_names = []

#Loop over each of layer output 
for output in layerOutput:
	#Loop over each detection
	for detection in output:
		#Extracting class_id, confidence 
		score = detection[5:]
		class_name = np.argmax(score)
		confidence = score[class_name]

		#Filtering Out weak probabilities
		if confidence > args["confidence"]:
			#Scaling boundry box
			# YOLO returns bounding box coordinates in the form: (centerX, centerY, width, and height).
			box = detection[0:4] * np.array([w,h,w,h])
			(center_x, center_y, wt, ht) = box.astype('int')

			#Using center_x and center_y cordinates to detect top and let-corner of box
			x = int(center_x - (wt/2))
			y = int(center_y - (ht/2))

			#Updating box-cordinates
			boxes.append([x,y,int(wt),int(ht)])
			confidences.append(float(confidence))
			class_names.append(class_name)
#Applying non-maxima supression to surpass weak boundry boxes
# NMS also ensures that we do not have any redundant or extraneous bounding boxes.
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

#Ensuring Atleast one detection exists
if len(idxs)>0:
	#looping over detections:
	for i in idxs.flatten():
		(x,y) = (boxes[i][0],boxes[i][1])
		(w,h) = (boxes[i][2],boxes[i][3])

		#Drawing rectangle on images
		color = [int(c) for c in COLORS[class_names[i]]]
		cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
		text = "{}: {:.4f}".format(labels[class_names[i]],confidences[i])
		cv2.putText(image,text,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
cv2.imshow("Image",image)
cv2.waitKey(0)	