# cctv


reference:
https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/


## Step 1: Get Weights
```
 $ wget https://pjreddie.com/media/files/yolov3.weights 
 ```

## Step 2: Get video

I used a video recorded on an iPhone 11 and used ffmpeg to convert it to avi
```
$ ffmpeg -i input.MOV input.avi
```

## Step 3: Process video


Change the code as needed. You can change the fps depending on video length and desired quality. 

I'm trying to detect potential bicycle thieves so in addition to annotating the video I'm outputting the frames with bicycles to the thieves folder.

 
```
$ python analyse.py
```

