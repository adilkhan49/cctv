import cv2
outputpath = 'output.avi'
fps = 2


vidcap = cv2.VideoCapture('input.avi')
success,image = vidcap.read()
size =  (image.shape[1],image.shape[0])

count = 0

image_array = []
while success:
  vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*500))    # added this line 
  # cv2.imwrite("res2/frame%d.jpg" % count, image)     # save frame as JPEG file      

  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
  image_array.append(image)


fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
out = cv2.VideoWriter(outputpath,fourcc, fps, size)
for i in range(len(image_array)):
   out.write(image_array[i])
out.release()