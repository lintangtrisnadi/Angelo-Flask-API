# Commented out IPython magic to ensure Python compatibility.
# import dependencies
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from google.colab.patches import cv2_imshow
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import html
import time
import matplotlib.pyplot as plt
# %matplotlib inline

from google.colab import files

# Mengecek apakah file model ada di direktori saat ini
model_path = '/models/fall_detection_model.h5'
if os.path.exists(model_path):
    print("File model ditemukan!")
else:
    print("File model tidak ditemukan.")

from keras.models import load_model
# load fall detection model
fall_detect = load_model('fall_detection_model.h5')

# clone darknet repo# clone darknet repo
!git clone https://github.com/AlexeyAB/darknet

# Commented out IPython magic to ensure Python compatibility.
# change makefile to have GPU, OPENCV and LIBSO enabled
# %cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
!sed -i 's/LIBSO=0/LIBSO=1/' Makefile

# make darknet (builds darknet so that you can then use the darknet.py file and have its dependencies)
!make

# get bthe scaled yolov4 weights file that is pre-trained to detect 80 classes (objects) from shared google drive
!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1V3vsIaxAlGWvK4Aar9bAiK5U0QFttKwq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1V3vsIaxAlGWvK4Aar9bAiK5U0QFttKwq" -O yolov4-csp.weights && rm -rf /tmp/cookies.txt

# import darknet functions to perform object detections
from darknet import *
# load in our YOLOv4 architecture network
network, class_names, class_colors = load_network("cfg/yolov4-csp.cfg", "cfg/coco.data", "yolov4-csp.weights")
width = network_width(network)
height = network_height(network)

# darknet helper function to run detection on image
def darknet_helper(img, width, height):
  darknet_image = make_image(width, height, 3)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_resized = cv2.resize(img_rgb, (width, height),
                              interpolation=cv2.INTER_LINEAR)

  # get image ratios to convert bounding boxes to proper size
  img_height, img_width, _ = img.shape
  width_ratio = img_width/width
  height_ratio = img_height/height

  # run model on darknet style image to get detections
  copy_image_from_bytes(darknet_image, img_resized.tobytes())
  detections = detect_image(network, class_names, darknet_image)
  free_image(darknet_image)
  return detections, width_ratio, height_ratio

import cv2

def preprocess_input(img_path):
    # Load the image
    img = cv2.imread(img_path)

    # Resize the image to the expected shape
    resized_img = cv2.resize(img, (150, 150))

    # You may also need to normalize or preprocess the image further based on your model requirements

    return resized_img

import numpy as np
import cv2
from tensorflow.keras.utils import load_img, img_to_array

# run test on person.jpg image that comes with repository
image = cv2.imread("/content/video_test (10).jpeg")
detections, width_ratio, height_ratio = darknet_helper(image, width, height)

for label, confidence, bbox in detections:
  left, top, right, bottom = bbox2points(bbox)
  left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
  cv2.rectangle(image, (left, top), (right, bottom), class_colors[label], 2)
  cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    class_colors[label], 2)

#cv2.imwrite(filename,image)
# Placeholder function for fall detection (replace this with your actual fall detection model)
def detect_fall1(image):

# Assuming 'image_data' is your input image data
    frame_reshape = np.reshape(image, (150, 150, 3))


    # Convert the frame to array
    x = img_to_array(frame_reshape)

    # Normalize pixel values to be between 0 and 1
    x /= 255.0

    # Add batch dimension
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    # Perform fall detection
    classes = fall_detect.predict(images)

    # Print the prediction
    print(classes[0])

    # Check if the prediction indicates a fall
    if classes[0] < 0.87:
        print("Jatuh")
    else:
        print("Tidak Jatuh")

    return classes
#print("Fall Detected!" if fall_detected else "No Fall Detected")
resized_img = cv2.resize(image, (150, 150))
resized_img = np.expand_dims(resized_img, axis=0)
fall_detected = detect_fall1(resized_img)
cv2_imshow(image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# function to convert the JavaScript object into an OpenCV image
def js_to_image(js_reply):
  """
  Params:
          js_reply: JavaScript object containing image from webcam
  Returns:
          img: OpenCV BGR image
  """
  # decode base64 image
  image_bytes = b64decode(js_reply.split(',')[1])
  # convert bytes to numpy array
  jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
  # decode numpy array into OpenCV BGR image
  img = cv2.imdecode(jpg_as_np, flags=1)

  return img

# function to convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream
def bbox_to_bytes(bbox_array):
  """
  Params:
          bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.
  Returns:
        bytes: Base64 image byte string
  """
  # convert array into PIL image
  bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
  iobuf = io.BytesIO()
  # format bbox into png for return
  bbox_PIL.save(iobuf, format='png')
  # format return string
  bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))

  return bbox_bytes

import cv2

def preprocess_input(img_path):
    # Load the image
    img = cv2.imread(img_path)

    # Resize the image to the expected shape
    resized_img = cv2.resize(img, (150, 150))

    # You may also need to normalize or preprocess the image further based on your model requirements

    return resized_img

from google.colab.patches import cv2_imshow
def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)

  # get photo data
  data = eval_js('takePhoto({})'.format(quality))
  # get OpenCV format image
  img = js_to_image(data)

  # call our darknet helper on webcam image
  detections, width_ratio, height_ratio = darknet_helper(img, width, height)



  # loop through detections and draw them on webcam image
  for label, confidence, bbox in detections:
    left, top, right, bottom = bbox2points(bbox)
    left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
    cv2.rectangle(img, (left, top), (right, bottom), class_colors[label], 2)
    cv2.putText(img, "{} [{:.2f}]".format(label, float(confidence)),
                      (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                      class_colors[label], 2)
  # save image
  cv2.imwrite(filename, img)
  def detect_fall2(img):

  # Assuming 'image_data' is your input image data
      frame_reshape = np.reshape(img, (150, 150, 3))


      # Convert the frame to array
      x = img_to_array(frame_reshape)

      # Normalize pixel values to be between 0 and 1
      x /= 255.0

      # Add batch dimension
      x = np.expand_dims(x, axis=0)
      images = np.vstack([x])

      # Perform fall detection
      classes = fall_detect.predict(images)

      # Print the prediction
      print(classes[0])

      # Check if the prediction indicates a fall
      if classes[0] < 0.75:
          print("Jatuh")
      else:
          print("Tidak Jatuh")

      return classes
  #print("Fall Detected!" if fall_detected else "No Fall Detected")
  resized_img = cv2.resize(img, (150, 150))
  resized_img = np.expand_dims(resized_img, axis=0)
  fall_detected = detect_fall2(resized_img)

    # Display the cropped frame with the bounding box overlay
  #stacked_images = np.hstack([resized_img[0], other_image])
  #cv2_imshow('Cropped Frame with Bounding Box', roi_frame)

  #print("Fall Detected!" if fall_detected else "No Fall Detected")

  return filename

from google.colab.patches import cv2_imshow
try:
  filename = take_photo('photo.jpg')
  print('Saved to {}'.format(filename))

  # Assuming img is your input image with shape (300, 300, 3)
  #resized_img = cv2.resize(img, (150, 150))

  # Show the image which was just taken.
  display(Image(filename))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))

# JavaScript to properly create our live video stream using our webcam as input
def video_stream():
  js = Javascript('''
    var video;
    var div = null;
    var stream;
    var captureCanvas;
    var imgElement;
    var labelElement;

    var pendingResolve = null;
    var shutdown = false;

    function removeDom() {
       stream.getVideoTracks()[0].stop();
       video.remove();
       div.remove();
       video = null;
       div = null;
       stream = null;
       imgElement = null;
       captureCanvas = null;
       labelElement = null;
    }

    function onAnimationFrame() {
      if (!shutdown) {
        window.requestAnimationFrame(onAnimationFrame);
      }
      if (pendingResolve) {
        var result = "";
        if (!shutdown) {
          captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
          result = captureCanvas.toDataURL('image/jpeg', 0.8)
        }
        var lp = pendingResolve;
        pendingResolve = null;
        lp(result);
      }
    }

    async function createDom() {
      if (div !== null) {
        return stream;
      }

      div = document.createElement('div');
      div.style.border = '2px solid black';
      div.style.padding = '3px';
      div.style.width = '100%';
      div.style.maxWidth = '600px';
      document.body.appendChild(div);

      const modelOut = document.createElement('div');
      modelOut.innerHTML = "<span>Status:</span>";
      labelElement = document.createElement('span');
      labelElement.innerText = 'No data';
      labelElement.style.fontWeight = 'bold';
      modelOut.appendChild(labelElement);
      div.appendChild(modelOut);

      video = document.createElement('video');
      video.style.display = 'block';
      video.width = div.clientWidth - 6;
      video.setAttribute('playsinline', '');
      video.onclick = () => { shutdown = true; };
      stream = await navigator.mediaDevices.getUserMedia(
          {video: { facingMode: "environment"}});
      div.appendChild(video);

      imgElement = document.createElement('img');
      imgElement.style.position = 'absolute';
      imgElement.style.zIndex = 1;
      imgElement.onclick = () => { shutdown = true; };
      div.appendChild(imgElement);

      const instruction = document.createElement('div');
      instruction.innerHTML =
          '<span style="color: red; font-weight: bold;">' +
          'When finished, click here or on the video to stop this demo</span>';
      div.appendChild(instruction);
      instruction.onclick = () => { shutdown = true; };

      video.srcObject = stream;
      await video.play();

      captureCanvas = document.createElement('canvas');
      captureCanvas.width = 640; //video.videoWidth;
      captureCanvas.height = 480; //video.videoHeight;
      window.requestAnimationFrame(onAnimationFrame);

      return stream;
    }
    async function stream_frame(label, imgData) {
      if (shutdown) {
        removeDom();
        shutdown = false;
        return '';
      }

      var preCreate = Date.now();
      stream = await createDom();

      var preShow = Date.now();
      if (label != "") {
        labelElement.innerHTML = label;
      }

      if (imgData != "") {
        var videoRect = video.getClientRects()[0];
        imgElement.style.top = videoRect.top + "px";
        imgElement.style.left = videoRect.left + "px";
        imgElement.style.width = videoRect.width + "px";
        imgElement.style.height = videoRect.height + "px";
        imgElement.src = imgData;
      }

      var preCapture = Date.now();
      var result = await new Promise(function(resolve, reject) {
        pendingResolve = resolve;
      });
      shutdown = false;

      return {'create': preShow - preCreate,
              'show': preCapture - preShow,
              'capture': Date.now() - preCapture,
              'img': result};
    }
    ''')

  display(js)

def video_frame(label, bbox):
  data = eval_js('stream_frame("{}", "{}")'.format(label, bbox))
  return data

# start streaming video from webcam
video_stream()
# label for video
label_html = 'Capturing...'
# initialze bounding box to empty
bbox = ''
'''count = 0
roi_top = 100
roi_bottom = 400
roi_left = 200
roi_right = 500'''

while True:
    js_reply = video_frame(label_html, bbox)
    if not js_reply:
        break

    # convert JS response to OpenCV Image
    frame = js_to_image(js_reply["img"])

        # Crop the frame to the defined ROI
    #roi_frame = frame[roi_top:roi_bottom, roi_left:roi_right]

    # create transparent overlay for bounding box
    bbox_array = np.zeros([480,640,4], dtype=np.uint8)

    # call our darknet helper on video frame
    detections, width_ratio, height_ratio = darknet_helper(frame, width, height)

    # loop through detections and draw them on transparent overlay image
    for label, confidence, bbox in detections:
      left, top, right, bottom = bbox2points(bbox)
      left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
      bbox_array = cv2.rectangle(bbox_array, (left, top), (right, bottom), class_colors[label], 2)
      bbox_array = cv2.putText(bbox_array, "{} [{:.2f}]".format(label, float(confidence)),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        class_colors[label], 2)


    bbox_array[:,:,3] = (bbox_array.max(axis = 2) > 0 ).astype(int) * 255
    # convert overlay of bbox into bytes
    bbox_bytes = bbox_to_bytes(bbox_array)
    # update bbox so next frame gets new overlay
    bbox = bbox_bytes
       # Perform fall detection processing on 'roi_frame' here
    def detect_fall2(bbox_array):

    # Assuming 'image_data' is your input image data
        #frame_reshape = np.reshape(bbox_array, (150, 150, 3))
        resized_frame = cv2.resize(frame,(150,150))

        # Convert the frame to array
        x = img_to_array(resized_frame)

        # Normalize pixel values to be between 0 and 1
        x /= 255.0

        # Add batch dimension
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])

        # Perform fall detection
        classes = fall_detect.predict(images)

        # Print the prediction
        print(classes[0])

        # Check if the prediction indicates a fall
        if classes[0] < 0.25:
            print("Jatuh")
        else:
            print("Tidak Jatuh")

        return classes
    #print("Fall Detected!" if fall_detected else "No Fall Detected")
    resized_img = cv2.resize(bbox_array, (150, 150))
    resized_img = np.expand_dims(resized_img, axis=0)
    fall_detected = detect_fall2(resized_img)


    # Display the original frame with the bounding box overlay
    #cv2.imshow('Original Frame with Bounding Box', frame)

    # Display the cropped frame with the bounding box overlay
    #cv2.imshow('Cropped Frame with Bounding Box', roi_frame)

    #print("Fall Detected!" if fall_detected else "No Fall Detected")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break