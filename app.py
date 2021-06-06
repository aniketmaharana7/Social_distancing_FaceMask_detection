import cv2
import time
import imutils
import argparse
import numpy as np
from math import pow, sqrt
from imutils.video import FPS
from imutils.video import VideoStream
from flask import Flask, render_template, Response,request
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from scipy.spatial import distance as dist
from werkzeug.utils import secure_filename
import os

l=[]

upf="C:/Users/lenovo/Desktop/project1111/Main_proj/static/upload"
#Initialize Flask
app = Flask(__name__)

app.config['upload_folder']=upf

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/facemask',methods=['POST'])
def facemask():
    return render_template('facemask.html')

@app.route('/browsefile',methods=['POST'])
def browsefile():
    return render_template('browsefile.html')

@app.route('/social',methods=['POST'])
def social():
    return render_template('social.html')

@app.route('/uploader',methods=['POST'])
def upandso():
	if(request.method=="POST"):
		f=request.files["video"]
		f.save(os.path.join(app.config['upload_folder'],secure_filename(f.filename)))
		l.append(f.filename)
		print(l[0])
		return render_template('social1.html')

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of image tag in html code
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/videofootage_feed')
def videofootage_feed():
    #Video streaming route. Put this in the src attribute of image tag in html code
    return Response(gen1(), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen1():
	NMS_THRESH = 0.3
	MIN_CONF = 0.3
	MIN_DISTANCE = 50

	# load the COCO class labels our YOLO model was trained on
	labelsPath = os.path.sep.join(["C:/Users/lenovo/Desktop/project1111/Main_proj/coco/coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join(["C:/Users/lenovo/Desktop/project1111/Main_proj/coco/yolov3.weights"])
	configPath = os.path.sep.join(["C:/Users/lenovo/Desktop/project1111/Main_proj/coco/yolov3.cfg"])

	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# initialize the video stream and pointer to output video file
	print("[INFO] accessing video stream...")
	cp="C:/Users/lenovo/Desktop/project1111/Main_proj/static/upload/"+l[0]+""
	vs = cv2.VideoCapture(cp)
	# vs=cv2.VideoCapture(0)
	# writer = None

	# loop over the frames from the video stream
	while True:
		# read the next frame from the file
		(grabbed, frame) = vs.read()

		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			break

		# resize the frame and then detect people (and only people) in it
		frame = imutils.resize(frame, height=600,width=700)
		results = detect_people(frame, net, ln,
								personIdx=LABELS.index("person"))

		# initialize the set of indexes that violate the minimum social
		# distance
		violate = set()

		# ensure there are *at least* two people detections (required in
		# order to compute our pairwise distance maps)
		if len(results) >= 2:
			# extract all centroids from the results and compute the
			# Euclidean distances between all pairs of the centroids
			centroids = np.array([r[2] for r in results])
			D = dist.cdist(centroids, centroids, metric="euclidean")

			# loop over the upper triangular of the distance matrix
			for i in range(0, D.shape[0]):
				for j in range(i + 1, D.shape[1]):
					# check to see if the distance between any two
					# centroid pairs is less than the configured number
					# of pixels
					if D[i, j] < MIN_DISTANCE:
						# update our violation set with the indexes of
						# the centroid pairs
						violate.add(i)
						violate.add(j)

		# loop over the results
		for (i, (prob, bbox, centroid)) in enumerate(results):
			# extract the bounding box and centroid coordinates, then
			# initialize the color of the annotation
			(startX, startY, endX, endY) = bbox
			(cX, cY) = centroid
			color = (0, 255, 0)

			# if the index pair exists within the violation set, then
			# update the color
			if i in violate:
				color = (0, 0, 255)

			# draw (1) a bounding box around the person and (2) the
			# centroid coordinates of the person,
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.circle(frame, (cX, cY), 5, color, 1)

		# draw the total number of social distancing violations on the
		# output frame
		text = "Social Distancing Violations: {}".format(len(violate))
		cv2.putText(frame, text, (10, frame.shape[0] - 25),
					cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

		frame = cv2.imencode('.jpg', frame)[1].tobytes()
		yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
	cv2.destroyAllWindows()



def gen():
    prototxtPath = r"C:\Users\lenovo\Desktop\project1111\Main_proj\models\deploy.prototxt"
    weightsPath = r"C:\Users\lenovo\Desktop\project1111\Main_proj\models\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model("C:/Users/lenovo/Desktop/project1111/Main_proj/models/mask_detector.model")
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    while True:
    	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, height=600,width=600)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()


def detect_and_predict_mask(frame, faceNet, maskNet):
    	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

def detect_people(frame, net, ln, personIdx=0):
    	# grab the dimensions of the frame and  initialize the list of
	# results
	NMS_THRESH=0.3
	MIN_CONF=0.3
	(H, W) = frame.shape[:2]
	results = []

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	# initialize our lists of detected bounding boxes, centroids, and
	# confidences, respectively
	boxes = []
	centroids = []
	confidences = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter detections by (1) ensuring that the object
			# detected was a person and (2) that the minimum
			# confidence is met
			if classID == personIdx and confidence > MIN_CONF:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# centroids, and confidences
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# update our results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)

	# return the list of results
	return results
