import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import time
import datetime as dt
import argparse
#  ----
# import numpy as np
# import os
# import six.moves.urllib as urllib
# import sys
# import tarfile
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import zipfile

# from collections import defaultdict
# from io import StringIO
# from matplotlib import pyplot as plt
# # from PIL import Image
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util

# Contoh IP RTSP
# rtsp://admin:klapaucius@192.168.0.100:8554/live
# admin adalah username perangkatnya
# klapaucius adalah password prangkatnya
# setelah @, adalah IP dan host nya
# biasnaya setiap perangkat punya username dan passwordnya sendiri, jadi harus tau terlebih dahulu


class App:
    def __init__(self, window, window_title, video_source='rtsp://admin:klapaucius@192.168.0.175:8554/live'):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.ok=False

        #timer
        self.timer=ElapsedTimeClock(self.window)

        # open video source (by default this will try to open the computer webcam)
        self.vid = VideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot=tk.Button(window, text="Snapshot", command=self.snapshot)
        self.btn_snapshot.pack(side=tk.LEFT)

        #video control buttons

        self.btn_start=tk.Button(window, text='START', command=self.open_camera)
        self.btn_start.pack(side=tk.LEFT)

        self.btn_stop=tk.Button(window, text='STOP', command=self.close_camera)
        self.btn_stop.pack(side=tk.LEFT)

        # quit button
        self.btn_quit=tk.Button(window, text='QUIT', command=quit)
        self.btn_quit.pack(side=tk.LEFT)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay=10
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret,frame=self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-"+time.strftime("%d-%m-%Y-%H-%M-%S")+".jpg",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

    def open_camera(self):
        self.ok = True
        self.timer.start()
        print("camera opened => Recording")



    def close_camera(self):
        self.ok = False
        self.timer.stop()
        print("camera closed => Not Recording")

       
    def update(self):

        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if self.ok:
            self.vid.out.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        self.window.after(self.delay,self.update)


class VideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        
        # # ---
        # # cap = cv2.VideoCapture('arwana.mp4')  # Change only if you have more than one webcams

        # # What model to download.
        # # Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
        # MODEL_NAME = 'inference_graph'
        # #MODEL_FILE = MODEL_NAME + '.tar.gz'
        # #DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

        # # Path to frozen detection graph. This is the actual model that is used for the object detection.
        # PATH_TO_CKPT = 'frozen_inference_graph.pb'

        # # List of the strings that is used to add correct label for each box.
        # PATH_TO_LABELS = os.path.join('data', 'labelmap.pbtxt')

        # # Number of classes to detect
        # NUM_CLASSES = 7

        # # Download Model
        # #tar_file = tarfile.open(MODEL_NAME)
        # #for file in tar_file.getmembers():
        # #    file_name = os.path.basename(file.name)
        # #    if 'frozen_inference_graph.pb' in file_name:
        # #        tar_file.extract(file, os.getcwd())


        # # Load a (frozen) Tensorflow model into memory.
        # detection_graph = tf.Graph()
        # with detection_graph.as_default():
        #     od_graph_def = tf.GraphDef()
        #     with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        #         serialized_graph = fid.read()
        #         od_graph_def.ParseFromString(serialized_graph)
        #         tf.import_graph_def(od_graph_def, name='')


        # # Loading label map
        # # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        # label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        # categories = label_map_util.convert_label_map_to_categories(
        #     label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        # category_index = label_map_util.create_category_index(categories)


        # # Helper code
        # def load_image_into_numpy_array(image):
        #     (im_width, im_height) = image.size
        #     return np.array(image.getdata()).reshape(
        #         (im_height, im_width, 3)).astype(np.uint8)


        # # Detection
        # with detection_graph.as_default():
        #     with tf.Session() as sess:
        #         while True:

        #             # Read frame from camera
        #             ret, image_np = self.vid.read()
        #             # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        #             image_np_expanded = np.expand_dims(image_np, axis=0)
        #             # Extract image tensor
        #             image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        #             # Extract detection boxes
        #             boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        #             # Extract detection scores
        #             scores = detection_graph.get_tensor_by_name('detection_scores:0')
        #             # Extract detection classes
        #             classes = detection_graph.get_tensor_by_name('detection_classes:0')
        #             # Extract number of detectionsd
        #             num_detections = detection_graph.get_tensor_by_name(
        #                 'num_detections:0')
        #             # Actual detection.
        #             (boxes, scores, classes, num_detections) = sess.run(
        #                 [boxes, scores, classes, num_detections],
        #                 feed_dict={image_tensor: image_np_expanded})
        #             # Visualization of the results of a detection.
        #             vis_util.visualize_boxes_and_labels_on_image_array(
        #                 image_np,
        #                 np.squeeze(boxes),
        #                 np.squeeze(classes).astype(np.int32),
        #                 np.squeeze(scores),
        #                 category_index,
        #                 use_normalized_coordinates=True,
        #                 line_thickness=8)

        #             # Display output
        #             cv2.imshow('Deteksi Ikan Cikalong', cv2.resize(image_np, (800, 600)))

        #             if cv2.waitKey(25) & 0xFF == ord('q'):
        #                 cv2.destroyAllWindows()
        #                 break
        # # ---

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Command Line Parser
        args=CommandLineParser().args

        
        #create videowriter

        # 1. Video Type
        VIDEO_TYPE = {
            'avi': cv2.VideoWriter_fourcc(*'XVID'),
            #'mp4': cv2.VideoWriter_fourcc(*'H264'),
            'mp4': cv2.VideoWriter_fourcc(*'XVID'),
        }

        self.fourcc=VIDEO_TYPE[args.type[0]]

        # 2. Video Dimension
        STD_DIMENSIONS =  {
            '480p': (640, 480),
            '720p': (1280, 720),
            '1080p': (1920, 1080),
            '4k': (3840, 2160),
        }
        res=STD_DIMENSIONS[args.res[0]]
        print(args.name,self.fourcc,res)
        self.out = cv2.VideoWriter(args.name[0]+'.'+args.type[0],self.fourcc,15,res)

        #set video sourec width and height
        self.vid.set(3,res[0])
        self.vid.set(4,res[1])

        # Get video source width and height
        self.width,self.height=res


    # To get frames
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            self.out.release()
            cv2.destroyAllWindows()


class ElapsedTimeClock:
    def __init__(self,window):
        self.T=tk.Label(window,text='00:00:00',font=('times', 20, 'bold'), bg='green')
        self.T.pack(fill=tk.BOTH, expand=1)
        self.elapsedTime=dt.datetime(1,1,1)
        self.running=0
        self.lastTime=''
        t = time.localtime()
        self.zeroTime = dt.timedelta(hours=t[3], minutes=t[4], seconds=t[5])
        # self.tick()

 
    def tick(self):
        # get the current local time from the PC
        self.now = dt.datetime(1, 1, 1).now()
        self.elapsedTime = self.now - self.zeroTime
        self.time2 = self.elapsedTime.strftime('%H:%M:%S')
        # if time string has changed, update it
        if self.time2 != self.lastTime:
            self.lastTime = self.time2
            self.T.config(text=self.time2)
        # calls itself every 200 milliseconds
        # to update the time display as needed
        # could use >200 ms, but display gets jerky
        self.updwin=self.T.after(100, self.tick)


    def start(self):
            if not self.running:
                self.zeroTime=dt.datetime(1, 1, 1).now()-self.elapsedTime
                self.tick()
                self.running=1

    def stop(self):
            if self.running:
                self.T.after_cancel(self.updwin)
                self.elapsedTime=dt.datetime(1, 1, 1).now()-self.zeroTime
                self.time2=self.elapsedTime
                self.running=0


class CommandLineParser:
    
    def __init__(self):

        # Create object of the Argument Parser
        parser=argparse.ArgumentParser(description='Script to record videos')

        # Create a group for requirement 
        # for now no required arguments 
        # required_arguments=parser.add_argument_group('Required command line arguments')

        # Only values is supporting for the tag --type. So nargs will be '1' to get
        parser.add_argument('--type', nargs=1, default=['avi'], type=str, help='Type of the video output: for now we have only AVI & MP4')

        # Only one values are going to accept for the tag --res. So nargs will be '1'
        parser.add_argument('--res', nargs=1, default=['480p'], type=str, help='Resolution of the video output: for now we have 480p, 720p, 1080p & 4k')

        # Only one values are going to accept for the tag --name. So nargs will be '1'
        parser.add_argument('--name', nargs=1, default=['output'], type=str, help='Enter Output video title/name')

        # Parse the arguments and get all the values in the form of namespace.
        # Here args is of namespace and values will be accessed through tag names
        self.args = parser.parse_args()



def main():
    # Create a window and pass it to the Application object
    App(tk.Tk(),'Video Recorder')

main()