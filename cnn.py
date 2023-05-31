import numpy as np
import cv2
import util

#global variables
index = 0
rate = 0.1
#related to yolo
label_path = "yolo-coco/coco.names"
weights_path = "yolo-coco/yolov3.weights"
config_path = "yolo-coco/yolov3.cfg"
confidence_param = 0.5
threshold_param = 0.3
#accuracy
counter = 0
sum = 0
#convert all info from annotation.txt file into array of Image objects
images = []
with open("WiderSelected/annotations.txt", "r") as read_file:
    while True:
        image_name = read_file.readline().strip()
        if(image_name == ''):
            break 
        else:
            n = int(read_file.readline().strip())
            faces_info = []
            for i in range(n):
                faces_info.append(read_file.readline().strip().split(" "))
            images.append(util.Image(image_name, faces_info))

#split image list into test & train images
split = int(rate * len(images))
test_images = images[:split]
train_images = images[split:]

#_________________________________________________________________________
labels = open(label_path).read().strip().split("\n")
# initialize random color for each label
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

# load YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
choice = input("Choose one : \n1-result on test data \n2-test new image\n")
if choice == str(1) :
	# show the output image
	while index < len(test_images):
		image = cv2.imread("WiderSelected/train/" + test_images[index].name)
		(H, W) = image.shape[:2]     #size of image
		# determine only the *output* layer names that we need from YOLO
		ln = net.getLayerNames()
		ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
		# construct a blob from the input image and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes and
		# associated probabilities
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
		net.setInput(blob)
		layerOutputs = net.forward(ln)

		# initialize our lists of detected bounding boxes, confidences, and
		# class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		for output in layerOutputs:
			for detection in output:
				# extract the class ID and confidence (i.e., probability) of
				# the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]
				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if classID == 0: 				#it's person
					if confidence > confidence_param:
						# scale the bounding box coordinates back relative to the
						# size of the image, keeping in mind that YOLO actually
						# returns the center (x, y)-coordinates of the bounding
						# box followed by the boxes' width and height
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")
						# use the center (x, y)-coordinates to derive the top and
						# and left corner of the bounding box
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))
						# update our list of bounding box coordinates, confidences,
						# and class IDs
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)
		# apply non-maxima suppression to suppress weak, overlapping bounding
		# boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_param, threshold_param)

		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in range(len(idxs.flatten())):
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				if(i < len(test_images[index].bounding_box)):
					box_in_image = test_images[index].bounding_box[i]
					accuracy = util.accuracy(x, y, w, h, box_in_image.x, box_in_image.y, box_in_image.w, box_in_image.h)
				else:
					accuracy = 0
				sum += accuracy
				counter += 1
				# draw a bounding box rectangle and label on the image
				color = [int(c) for c in colors[classIDs[i]]]
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
				cv2.putText(image, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				cv2.putText(image, "accuracy : " + str(accuracy), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

		cv2.imshow("Image", image)
		key = cv2.waitKey(30) & 0xFF
		if key == ord('n'):
			index += 1
		elif key == ord('p') and not index == 0:
			index -= 1
		elif key == ord('q'):
			break  # quit

	cv2.destroyAllWindows()
	print("Mean of shown image accuracy : " + str(sum / counter))

elif choice == str(2):
	path = input("Enter the path\n")
	# num_of_bounding_box = int(input("Enter number of bounding box in image\n"))
	image = cv2.imread(path)
	(H, W) = image.shape[:2]     #size of image
	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)
	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	for output in layerOutputs:
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if classID == 0: 				#it's person
				if confidence > confidence_param:
					# scale the bounding box coordinates back relative to the
					# size of the image, keeping in mind that YOLO actually
					# returns the center (x, y)-coordinates of the bounding
					# box followed by the boxes' width and height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")
					# use the center (x, y)-coordinates to derive the top and
					# and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))
					# update our list of bounding box coordinates, confidences,
					# and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)
	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_param, threshold_param)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in range(len(idxs.flatten())):
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# if(i < num_of_bounding_box):
			# 	box_in_image = num_of_bounding_box
			# 	accuracy = util.accuracy(x, y, w, h, box_in_image.x, box_in_image.y, box_in_image.w, box_in_image.h)
			# else:
			# 	accuracy = 0
			# sum += accuracy
			# counter += 1
			# draw a bounding box rectangle and label on the image
			color = [int(c) for c in colors[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
			cv2.putText(image, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			# cv2.putText(image, "accuracy : " + str(accuracy), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
while(True):
	cv2.imshow("Image", image)
	key = cv2.waitKey(30) & 0xFF

	if key == ord('q'):
		break  # quit
cv2.destroyAllWindows()
# print("Mean of accuracy : " + str(sum / counter))