import numpy as np
import os
import time
import argparse
import cv2

arg = argparse.ArgumentParser()
arg.add_argument("-i","--input",required=True,
	help="Path to input video")
arg.add_argument("-o","--output",required=True,
	help="Path to output video directory")
arg.add_argument("-y","--yolo",required=True,
	help="Path  to yolo directory")
arg.add_argument("-c","--confidence",type=float,default=0.5,
	help="minimum probability to filter weak detections")
arg.add_argument("-t","--threshold",type=float,default=0.3,
	help="threshold when applying non-maxima supression")
args = vars(arg.parse_args())

labelspath = os.path.sep.join([args["yolo"],"coco.names"]) 
labels = open(labelspath).read().strip().split("\n") #Strip: Removing Leading and Trailing spaces


weight_path = os.path.sep.join([args['yolo'],'yolov3.weights'])
config_path = os.path.sep.join([args['yolo'],'yolov3.cfg'])

#Initialzing Colors
np.random.seed(42)
COLORS = np.random.randint(0,255,size=(len(labels),3),dtype='uint8')

#Loading Yolo

print("[INFO] Loading Yolo From Disk")
net = cv2.dnn.readNetFromDarknet(config_path,weight_path)
ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

#Initialize video-stream, pointer to output video file
video = cv2.VideoCapture(args["input"])
writer = None
(W,H) = (None,None)

#Trying To determine frames in video
try:
	prop = cv2.CAP_PROP_FRAME_COUNT
	total = int(video.get(prop))
	print("[INFO] {} total frames in video".format(total))
# an error occurred while trying to determine the total
# number of frames in the video file
except:	
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

#Loop Over each video frame
while True:
	#read next frame from file
	(grabbed, frame) = video.read()

	if not grabbed: # If not frame grabbed means we reach at end of video
		break
	# If W and H is none	
	if W is None and H is None:
		(H,W) = frame.shape[:2] 	


#Construct a blob from current frame and forward pass of yolo detector, giving boundry boxes and probablity
	blob = cv2.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB=True,crop=False)
	net.setInput(blob)

	start = time.time()
	layer_output = net.forward(blob)
	end = time.time()

	boxes=[]
	confidences=[]
	class_names=[]
	#Iterating Over Layer's Output
	for output in layer_output:
		#Looping Over each detection
		for detection in output:
			score = detection[5:]
			class_name = np.argmax(score)
			confidence = score[class_name]

			if confidence > args["confidence"]:
				#Scaling Boundry Boxes
				box = detection[0:4] * np.array([W,H,W,H])
				(center_x, center_y, width, height) = box.astype('int')

				#Using center_x and center_y cordinates to detect top and let-corner of box
				x = int(center_x - (width/2))
				y = int(center_y - (height/2))

				#Updating box-cordinates
				boxes.append([x,y,int(width),int(height)])
				confidences.append(float(confidence))
				class_names.append(class_name)
	#Applying non-maxima supression to surpass weak boundry boxes
	# NMS also ensures that we do not have any redundant or extraneous bounding boxes.	
	idx = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],args["threshold"])
	#Ensuring AtLeast One detection occurs:
	if len(idx) > 0:
		#Looping each index
		for i in idx.flatten():
			# extract the bounding box coordinates
			(x,y) = (boxes[i][0],boxes[i][1])
			(w,h) = (boxes[i][2],boxes[i][3])

			#Drawing Box Rectangle and label frame
			color = [int(c) for c in COLORS[class_names[i]]]
			cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
			text = "{}: {:.4f}".format(labels[class_names[i]],confidences[i])
			cv2.putText(frame, text, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	#Checking If Writer is None
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"],fourcc,30,(frame.shape[1],frame.shape[0]),True)	
		# some information on processing single frame	
		if total>0:
			elap = (end-start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated Image to finish: {:.4f}".format(elap*total))
	#Writing To disk		
	writer.write(frame)

#Releasing All pointers
print("[INFO] Cleaning Up")
writer.release()
video.release()			 
