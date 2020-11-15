import cv2
from yolo import *


def  get_model():
	global labels,net_h, net_w, obj_thresh, nms_thresh, anchors

	weights_path = "yolov3.weights"
	# set some parameters
	net_h, net_w = 416, 416
	obj_thresh, nms_thresh = 0.5, 0.45
	anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
	labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
	          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
	          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
	          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
	          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
	          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
	          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
	          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
	          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
	          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

	# make the yolov3 model to predict 80 classes on COCO
	yolov3 = make_yolov3_model()

	# load the weights trained on COCO into the model
	weight_reader = WeightReader(weights_path)
	weight_reader.load_weights(yolov3)

	return yolov3

def draw_boxes(image, boxes, labels, obj_thresh):
	annotations = []
	for box in boxes:
		label_str = ''
		label = -1
		for i in range(len(labels)):
			if box.classes[i] > obj_thresh:
				label_str += labels[i]
				label = i
				print(labels[i] + ': ' + str(box.classes[i]*100) + '%')
				annotations.append(labels[i])

                
		if label >= 0:
			cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (0,255,0), 3)
			cv2.putText(image, 
                        label_str + ' ' + str(box.get_score()), 
                        (box.xmin, box.ymin - 13), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1e-3 * image.shape[0], 
                        (0,255,0), 2)	
	return annotations

def process(image):
    image_h, image_w, _ = image.shape
    new_image = preprocess_input(image, net_h, net_w)

    # run the prediction
    yolos = yolov3.predict(new_image)
    boxes = []

    for i in range(len(yolos)):
        # decode the output of the network
        boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)

    # correct the sizes of the bounding boxes
    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

    # suppress non-maximal boxes
    do_nms(boxes, nms_thresh)     

    # draw bounding boxes on the image using labels
    annotations = draw_boxes(image, boxes, labels, obj_thresh) 

    return image,annotations


outputpath = 'output.avi'
fps = 1



yolov3 = get_model()
vidcap = cv2.VideoCapture('input.avi')
success,image = vidcap.read()
size =  (image.shape[1],image.shape[0])

count = 0
image_array = []



while success:
  # try:
	  vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*int(1000/fps)))    # added this line 
	  # cv2.imwrite("res2/frame%d.jpg" % count, image)     # save frame as JPEG file      

	  success,image = vidcap.read()
	  print('Read a new frame: ', success)
	  count += 1
	  img,annotations = process(image)
	  print('*' + str(annotations))
	  if 'bicycle' in annotations:
	      cv2.imwrite("thieves/frame%d.jpg" % count, image)     # save frame as JPEG file      
	  image_array.append(img)
	  # cv2.imshow(image)
	  # cv2.waitKey(1)
  # except:
  # 	pass


fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
out = cv2.VideoWriter(outputpath,fourcc, fps, size)
for i in range(len(image_array)):
   out.write(image_array[i])
out.release()