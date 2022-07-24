import argparse
import cv2
import matplotlib.pyplot as plt
import math
from pyrsistent import b
import scipy.ndimage as ndimage
import numpy as np
from scipy import ndimage


import tensorflow as tf
import torch
#import torch.backends.cudnn as cudnn
#from ggcnn.models import ggcnn2
from ggcnn.utils.dataset_processing import  image
import sys
from skimage.feature import peak_local_max
from scipy.ndimage.measurements import center_of_mass, label
from agnostic_segmentation import agnostic_segmentation

#from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder

import rospy
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32MultiArray
from PIL import Image, ImageDraw, ImageFont
from six import BytesIO
from IPython.display import display, Javascript
#from IPython.display import Image as IPyImage
from copy import deepcopy
from sensor_msgs.msg import Image, CameraInfo

import warnings
warnings.filterwarnings("ignore")

##### Upload GGCNN model
device2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_FILE = 'ggcnn/ggcnn2_093'
model = torch.load(MODEL_FILE, map_location=device2)

tf_matrix=[319.5, 239.5,525,525]


def parse_args():
    parser = argparse.ArgumentParser(description='Predict grasps')
    parser.add_argument('--agnostic_segmentation', type=int, default=0, help='True if using agnostic segmentation to detect objects in the bin')
    parser.add_argument('--bin_segmentation', type=int, default=1, help='Segment bin with retina net before objects detection')
    parser.add_argument('--load_files', type=int, default=1, help='True if using saved rbg and depth images')
    parser.add_argument('--rgb_path', type=str, default='imagenes/rgb17.jpg' , help='Path to rgb image')
    parser.add_argument('--depth_path', type=str, default='imagenes/depth17.jpg' , help='Path to depth image')
    parser.add_argument('--box_index', type=int, default=0, help='object to use for grasp prediction')
    parser.add_argument('--obj_zoom', type=float, default=0.25, help='zoom of object segmentation rectangles')
    parser.add_argument('--bin_zoomx', type=float, default=0, help='zoom of bin segmentation rectangle')
    parser.add_argument('--bin_zoomy', type=float, default=0, help='zoom of bin segmentation rectangle')
    parser.add_argument('--off_x', type=float, default=0.02, help='offset x percentage between Depth and RGB')
    parser.add_argument('--off_y', type=float, default=0.045, help='offset y percentage between Depth and RGB')

    parser.add_argument('--threshold', type=float, default=0.75, help='Initial threshold for grasping')
    parser.add_argument('--use_masks', type=int, default=0, help='If using masks after image segmentation')
    parser.add_argument('--preprocess', type=int, default=1, help='preprocessing depth image')
    parser.add_argument('--show_obj', type=int, default=1, help='if show detected objects')
    parser.add_argument('--mask_factor', type=int, default=10, help='Dilation size')
    parser.add_argument('--confidence', type=float, default=0.3, help='Confidence value agnostic segmentation')


    args = parser.parse_args()
    return args


@tf.function
def detect(detection_model,input_tensor):
  """Run detection on an input image.

  Args:
    input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.

  Returns:
    A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
      and `detection_scores`).
  """
  preprocessed_image, shapes = detection_model.preprocess(input_tensor)
  prediction_dict = detection_model.predict(preprocessed_image, shapes)
  return detection_model.postprocess(prediction_dict, shapes)

def create_retina(num_classes=2, pipeline_config=None, checkpoint_path=None):
  tf.keras.backend.clear_session()
  print('Building model and restoring weights for fine-tuning...', flush=True)
  configs = config_util.get_configs_from_pipeline_file(pipeline_config)
  model_config = configs['model']
  model_config.ssd.num_classes = num_classes
  model_config.ssd.freeze_batchnorm = True
  detection_model = model_builder.build(
        model_config=model_config, is_training=False)

  fake_box_predictor = tf.compat.v2.train.Checkpoint(
      _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
      _prediction_heads=detection_model._box_predictor._prediction_heads,
      _box_prediction_head=detection_model._box_predictor._box_prediction_head,
      )
  fake_model = tf.compat.v2.train.Checkpoint(
            _feature_extractor=detection_model._feature_extractor,
            _box_predictor=fake_box_predictor)
  ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
  ckpt.restore(checkpoint_path).expect_partial()

  # Run model through a dummy image so that variables are created
  image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
  prediction_dict = detection_model.predict(image, shapes)
  _ = detection_model.postprocess(prediction_dict, shapes)
  print('Weights restored!')
  return detection_model

def predict_grasps(depth_img,  filters=(5.0, 3.0, 1.0), width_scale=70, preprocess=True ):
  #ix=detections['detection_boxes'][0][0][1]
  #iy=detections['detection_boxes'][0][0][0]
  #depth_crop = depth_img[ix+roi[1]:ix+roi[3],iy+roi[0]:iy+roi[2]]
  #depth_crop=depth_img/10

  depth_crop = cv2.resize(depth_img, (300, 300))
  if preprocess:
    depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)
    depth_crop[depth_nan_mask==1] = 0
    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    depth_scale = np.abs(depth_crop).max()
    depth_crop = depth_crop.astype(np.float32) / depth_scale 
    # with TimeIt('Inpainting'):
    depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)
    depth_crop = depth_crop[1:-1, 1:-1]
    depth_crop = depth_crop * depth_scale
    depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
  depth = depth_crop
  depthT = torch.from_numpy(depth.reshape(1, 1, depth.shape[0], depth.shape[1]).astype(np.float32)).to(device2)


  with torch.no_grad():
      pred_out = model(depthT)
  points_out = pred_out[0].cpu().numpy().squeeze()
  cos_out = pred_out[1].cpu().numpy().squeeze()
  sin_out = pred_out[2].cpu().numpy().squeeze()
  ang_out = np.arctan2(sin_out, cos_out) / 2.0
  width_out = pred_out[3].cpu().numpy().squeeze() * width_scale  # Scaled 0-150:0-1
  points_out = ndimage.filters.gaussian_filter(points_out, filters[0])  # 3.0
  ang_out = ndimage.filters.gaussian_filter(ang_out, filters[1])
  width_out = ndimage.filters.gaussian_filter(width_out, filters[2])
  points_out = np.clip(points_out, 0.0, 1.0-1e-3)
  return points_out, ang_out, width_out


def detect_grasps(q_img, ang_img, threshold, width_img=None, no_grasps=10):
    is_peak = peak_local_max(q_img, min_distance=30, threshold_abs=threshold, indices=False)
    labels = label(is_peak)[0]
    merged_peaks = center_of_mass(is_peak, labels, range(1, np.max(labels)+1))
    local_max = np.array(merged_peaks)
    print("len", len(local_max))

    ######### Used for pushing detection
    #arr = []
    #for i in range(len(local_max)):
    #    py,px = local_max[i]
        #if py <= 60 or py>=340 or px<=113 or px >= 288 :
        #  arr = np.append(arr, i).astype(np.int)
    #cont = 0
    #for i in range(len(arr)):
    #    print(arr[i])
        #local_max = np.delete(local_max,arr[i]-cont,0)
    #    cont = cont+1
    #local_max = np.delete(local_max,i,0)

    print("localn", local_max)
    local_max = np.round(local_max).astype(np.int)
    grasps = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)
        #print(grasp_point)
        g = dict()
        g["pix"]=grasp_point
        g["ang"]=ang_img[grasp_point]
        g["q"]=q_img[grasp_point]      
        if width_img is not None:
            length = width_img[grasp_point]
            g["width"] = length/2
        grasps.append(g)
    if grasps == []:
        if threshold >0.5:
            threshold= threshold-0.1
            print(threshold)
            grasps = detect_grasps(q_img, ang_img,  threshold, width_img=None, no_grasps=1)
        return
    return grasps

def from_obj_frame(px, py, roi):
  pxb=px+int(roi[1])
  pyb=py+int(roi[0])
  return pxb, pyb

def from_bin_frame(pxb, pyb, detections):
  pxo=pxb+int(detections['detection_boxes'][0][0][1])
  pyo=pyb+int(detections['detection_boxes'][0][0][0])
  return pxo, pyo

def to_distance(u, v,z=1.3, mx=tf_matrix):
  cx, cy, fx, fy=mx
  x=(u-cx)*z/fx
  y=(v-cy)*z/fy
  return x,y

def rvalues(grasp_info, depth, roi, detections):
    px=grasp_info["pix"][0]
    py=grasp_info["pix"][1]

    pxb, pyb=from_obj_frame(px,py, roi) 
    pxo, pyo=from_bin_frame(pxb, pyb, detections)

    z = np.median(depth)
    x, y=to_distance(pxo, pyo)
    
    x1, y1 =to_distance(pxo+grasp_info["width"]*math.cos(grasp_info["ang"])/2, pyo+grasp_info["width"]*math.sin(grasp_info["ang"])/2)
    x2, y2 =to_distance(pxo-grasp_info["width"]*math.cos(grasp_info["ang"])/2, pyo-grasp_info["width"]*math.sin(grasp_info["ang"])/2)
    #rwidth= Width in meters
    rwidth =math.sqrt(math.pow((x1-x2),2)+math.pow((y1-y2),2))
    return x,y,z, grasp_info["ang"], rwidth


def draw_angled_rec(x0, y0, width, height, angle, color, img):
    #print("angulo: ", angle)
    _angle=angle
    #_angle = angle * math.pi / 180.0
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    cv2.line(img, pt0, pt1, (), 2)
    cv2.line(img, pt1, pt2, color, 2)
    cv2.line(img, pt2, pt3, color, 2)
    cv2.line(img, pt3, pt0, color, 2)

    return img

def main():
  args = parse_args()

  ######Download data
  if args.load_files:
    #depth_img = image.DepthImage.from_tiff(args.depth_path)
    depth_img = image.Image.from_file(args.depth_path)
    depth_img = np.asarray(depth_img)
    rgb_img = image.Image.from_file(args.rgb_path)
    rgb_img =  np.asarray(rgb_img)
  else:
    rospy.init_node('save_img')
    bridge = CvBridge()
    cmd_pub = rospy.Publisher('ggcnn/rvalues', Float32MultiArray, queue_size=1)
    rate = rospy.Rate(1) # ROS Rate at 5Hz
    print("waiting")
    rgbo = rospy.wait_for_message('/camera/rgb/image_color', Image)
    deptho = rospy.wait_for_message('/camera/depth/image_raw', Image)
    depth_img = bridge.imgmsg_to_cv2(deptho)
    depth_img = np.asarray(depth_img)
    rgb_img = bridge.imgmsg_to_cv2(rgbo)
    rgb_img= cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

  #depth_img=depth_img/3
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(1, 2, 1)
  ax.imshow(rgb_img)
  ax.set_title('rgb')
  ax.axis('off')

  ax = fig.add_subplot(1, 2, 2)
  ax.imshow(depth_img, cmap='gray')
  ax.set_title('Depth')
  ax.axis('off')
  plt.show()


  if args.bin_segmentation:
    detection_model=create_retina(num_classes=1, pipeline_config = 'models/retina_config/config/pipeline.config', checkpoint_path = "models/retina_config/checkpoint/ckpt-1") 
    input_tensor = tf.convert_to_tensor(np.expand_dims(rgb_img, axis=0), dtype=tf.float32)
    detections = detect(detection_model,input_tensor)
    off_x=args.off_x
    off_y=args.off_y
    x1=detections['detection_boxes'][0][0][1]+off_x
    x2=detections['detection_boxes'][0][0][3]+off_x
    y1=detections['detection_boxes'][0][0][0]-off_y
    y2=detections['detection_boxes'][0][0][2]-off_y
    detections = {"detection_boxes": [[[y1+(y2-y1)*args.bin_zoomy, x1+(x2-x1)*args.bin_zoomx, y2-(y2-y1)*args.bin_zoomy, x2-(x2-x1)*args.bin_zoomx]]]}
  else:
    detections = {"detection_boxes": [[[0, 0, 1, 1]]]}

  #print(detections)

  shape=rgb_img.shape
  bin_img=rgb_img[int(shape[0]*(detections['detection_boxes'][0][0][0]+off_y)):int(shape[0]*(detections['detection_boxes'][0][0][2]+off_y)),int(shape[1]*(detections['detection_boxes'][0][0][1]-off_x)):int(shape[1]*(detections['detection_boxes'][0][0][3]-off_x))]
  bin_img_depth=depth_img[int(shape[0]*detections['detection_boxes'][0][0][0]):int(shape[0]*detections['detection_boxes'][0][0][2]),int(shape[1]*detections['detection_boxes'][0][0][1]):int(shape[1]*detections['detection_boxes'][0][0][3])]

  #bin_img=bin_img[46:210,42:324]
  #bin_img_depth=bin_img_depth[46:210,42:324]  

  if args.bin_segmentation:  
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(bin_img)
    ax.set_title('rgb')
    ax.axis('off')
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(bin_img_depth, cmap='gray')
    ax.set_title('Depth')
    ax.axis('off')
    plt.show()
  if args.agnostic_segmentation:
    MODEL_PATH="models/FAT_trained_Ml2R_bin_fine_tuned.pth"
    predictions = agnostic_segmentation.segment_image(bin_img, MODEL_PATH, confidence=args.confidence)
    pred_boxes={}
    for i in range(len(predictions["instances"])):
      pred_boxes[i]=predictions["instances"][i].get_fields()["pred_boxes"].tensor.numpy()[0]
    
    masks=predictions['instances'].pred_masks

    print("# Objects: ", len(pred_boxes))
    zoom_p = args.obj_zoom
    roi = (pred_boxes[args.box_index]*np.array([1-zoom_p, 1-zoom_p, 1+zoom_p, 1+zoom_p])).astype(int)
    print("roi: ",roi)

    roi_w=roi[3]-roi[1]
    roi_h=roi[2]-roi[0]
    roi2=roi
    if roi_w>roi_h:
      roi2[0]=roi2[0]+round(roi_h/2)-round(roi_w/2)
      roi2[2]=roi2[2]-round(roi_h/2)+round(roi_w/2)
    elif roi_h>roi_w:
      roi2[1]=roi2[1]+round(roi_w/2)-round(roi_h/2)
      roi2[3]=roi2[3]-round(roi_w/2)+round(roi_h/2)
    print("New roi: ",roi2)

    if ((roi2[0]) or (roi2[1]) or (roi2[2]) or (roi2[3]) )< 0:
      print("Out of region")
    else:
      roi=roi2
  else:
    print("shape: ",bin_img.shape)
    roi=[0,0,bin_img.shape[1],bin_img.shape[0]]
    print(roi)

  if args.show_obj:
    try:
      plt.figure(figsize=(30, 15))
      for i in range(len(pred_boxes)):
        plt.subplot(1, round(len(pred_boxes)), i+1)
        roi_aux = (pred_boxes[i]*np.array([1-zoom_p, 1-zoom_p, 1+zoom_p, 1+zoom_p])).astype(int)
        plt.imshow(bin_img[roi_aux[1]:roi_aux[3], roi_aux[0]:roi_aux[2]])
      plt.show()
    except:
      print("Agnostic segmentation not working")

  
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(1, 1, 1)
  ax.imshow(bin_img[roi[1]:roi[3], roi[0]:roi[2]])
  ax.set_title('rgb')
  ax.axis('off')
  plt.show()

  bin_img_depth=deepcopy(bin_img_depth)
  if args.agnostic_segmentation:
    if args.use_masks:
      kernel_size=int((roi2[3]-roi2[1])/args.mask_factor)
      kernel = np.ones((kernel_size,kernel_size), np.uint8)
      mask=masks[args.box_index].cpu().detach().numpy()
      print(mask.shape)
      mask = ndimage.binary_dilation(mask, structure=kernel, iterations=1)
      #mask = cv2.dilate(mask, kernel, iterations=2)
      #depth_img_2=depth_img.copy()
      d=np.median(bin_img_depth)
      print("Median: ",d)
      for j in range(bin_img_depth.shape[0]):
        for i in range(bin_img_depth.shape[1]):
          if not mask[j][i]:
              bin_img_depth[j][i] = d

  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(1, 1, 1)
  ax.imshow(bin_img_depth[roi[1]:roi[3], roi[0]:roi[2]], cmap='gray')
  ax.set_title('Depth')
  ax.axis('off')
  plt.show()


  points_out, ang_out, width_out=predict_grasps(bin_img_depth[roi[1]:roi[3], roi[0]:roi[2]], filters=(5.0, 3.0, 2.0), width_scale=120, preprocess=args.preprocess)
  points_out = cv2.resize(points_out, (roi[2]-roi[0], roi[3]-roi[1]))
  ang_out = cv2.resize(ang_out, (roi[2]-roi[0], roi[3]-roi[1]))
  width_out = cv2.resize(width_out, (roi[2]-roi[0], roi[3]-roi[1]))

  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(1, 1, 1)
  plot = ax.imshow(points_out, cmap='jet', vmin=0, vmax=1)
  ax.set_title('Depth')
  ax.axis('off')
  plt.show()

  # Detect grasp hypothesis
  grasps = detect_grasps(points_out, ang_out, args.threshold, width_img=width_out, no_grasps=1)
  for g in grasps:
      print(g)


  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(2, 2, 1)
  depth_img_2=bin_img_depth[roi[1]:roi[3], roi[0]:roi[2]]
  for g in grasps:
    depth_img_2=draw_angled_rec(g["pix"][1], g["pix"][0], g["width"], g["width"]/2, g["ang"], color=(0.1,0.1,0), img=depth_img_2)
  ax.imshow(depth_img_2, cmap='gray')
  ax.set_title('Depth')
  ax.axis('off')
  for g in grasps:
    points_out2=draw_angled_rec(g["pix"][1], g["pix"][0], g["width"], g["width"]/2, g["ang"], color=(0.1,0.1,0), img=points_out)
  ax = fig.add_subplot(2, 2, 2)
  plot = ax.imshow(points_out2, cmap='jet', vmin=0, vmax=1)
  ax.set_title('quality')
  ax.axis('off')

  ax = fig.add_subplot(2, 2, 3)
  plot = ax.imshow(ang_out, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
  ax.set_title('Angle')
  plt.colorbar(plot)
  ax.axis('off')

  ax = fig.add_subplot(2, 2, 4)
  plot = ax.imshow(width_out, cmap='hsv', vmin=0, vmax=150)
  ax.set_title('width')
  ax.axis('off')
  plt.colorbar(plot)
  plt.show()

  #Transform to distance
  x, y, z, ang, rwidth =rvalues(grasps[0], depth_img, roi, detections)
  print("x: ", x)
  print("y: ", y)
  print("z: ", z)
  print("ang: ", ang)
  print("rwidth: ", rwidth)

  #Publish data
  if args.load_files==0:
    cmd_msg = Float32MultiArray()
    cmd_msg.data = [x, y, z, ang, rwidth]
    cmd_pub.publish(cmd_msg)

  #pose = geometry_msgs.msg.Pose()
  #pose.position.x = z
  #pose.position.y = -x
  #pose.position.z = -y
  #q = tft.quaternion_from_euler(0, 1.5, ang)

if __name__ == '__main__':
    main()
    sys.exit(0)