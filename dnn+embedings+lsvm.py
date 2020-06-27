from imutils import paths
import numpy as np
import imutils
import cv2.dnn
import os
import pickle
from PIL import Image
model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
path=[os.path.join("data",f) for f in os.listdir("data")]
knownEmbeddings=[]
knownid=[]
total=0
for imagePath in path:
    img = cv2.imread(imagePath)
    id=int(os.path.split(imagePath)[1].split(".")[1])
    image = imutils.resize(img, width=600)
    (h, w) = img.shape[:2]


	# construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize faces in the input image
    model.setInput(imageBlob)
    detections = model.forward()

	# ensure at least one face was found
    if len(detections) > 0:
		# we're making the assumption that each image has only ONE face, so find the bounding box with the largest probability
	    i = np.argmax(detections[0, 0, :, 2])
	    confidence = detections[0, 0, i, 2]

		# ensure that the detection with the largest probability also means our minimum probability test (thus helping filter out weak detections)
	    if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the face
		    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		    (startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI and grab the ROI dimensions
		    face = image[startY:endY, startX:endX]
		    (fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
		    if fW < 20 or fH < 20:
			    continue

			# construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
		    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
		    embedder.setInput(faceBlob)
		    vec = embedder.forward()

			# add the name of the person + corresponding face embedding to their respective lists
		    knownid.append(id)
		    knownEmbeddings.append(vec.flatten())
		    total += 1

# dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "ids": knownid}
f = open("output/embeddings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
