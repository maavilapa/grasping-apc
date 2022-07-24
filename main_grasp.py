import rospy
import cv2
import matplotlib.pyplot as plt
import math
import scipy.ndimage as ndimage
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage
import tensorflow as tf
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from agnostic_segmentation import agnostic_segmentation
from ggcnn.models import ggcnn2
from ggcnn.utils.dataset_processing import  image
from skimage.feature import peak_local_max
from scipy.ndimage.measurements import center_of_mass, label

from cv_bridge import CvBridge, CvBridgeError
import tf.transformations as tft
from std_msgs.msg import Float32MultiArray
import geometry_msgs.msg


##### Upload GGCNN model
device2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_FILE = '/content/grasping_task/ggcnn/ggcnn2_093'
model = torch.load(MODEL_FILE, map_location=device2)

rospy.init_node('save_img')
bridge = CvBridge()
cmd_pub = rospy.Publisher('ggcnn/rvalues', Float32MultiArray, queue_size=1)
rate = rospy.Rate(1) # ROS Rate at 5Hz
tf_matrix=[319.5, 239.5,525,525]

def predict_grasps(depth_img, detections, roi, filters=(5.0, 4.0, 2.0), width_scale=70, preprocess=True ):
  ix=detections['detection_boxes'][0][0][1]
  iy=detections['detection_boxes'][0][0][0]
  depth_crop = depth_img[ix+roi[1]:ix+roi[3],iy+roi[0]:iy+roi[2]]
  if preprocess:
    depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)
    depth_crop[depth_nan_mask==1] = 0
    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    depth_scale = np.abs(depth_crop).max()
    depth_crop = depth_crop.astype(np.float32) / depth_scale 
    # with TimeIt('Inpainting'):
    # depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)
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


def detect_grasps(q_img, ang_img, threshold, width_img=None, no_grasps=5):
    is_peak = peak_local_max(q_img, min_distance=15, threshold_abs=threshold, indices=False)
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
        if threshold >0.2:
            threshold= threshold-0.1
            print(threshold)
            grasps = detect_grasps(q_img, ang_img,  threshold, width_img=None, no_grasps=5)
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

    z = depth[pyo,pxo]
    x, y=to_distance(pxo, pyo)
    
    x1, y1 =to_distance(pxo+grasp_info["width"]*math.cos(grasp_info["ang"])/2, pyo+grasp_info["width"]*math.sin(grasp_info["ang"])/2)
    x2, y2 =to_distance(pxo-grasp_info["width"]*math.cos(grasp_info["ang"])/2, pyo-grasp_info["width"]*math.sin(grasp_info["ang"])/2)
    #rwidth= Width in meters
    rwidth =math.sqrt(math.pow((x1-x2),2)+math.pow((y1-y2),2))
    return x,y,z, grasp_info["ang"], rwidth

def main():
  ######Download data
  #rgbo = rospy.wait_for_message('/camera/color/image_raw', Image)
  #deptho = rospy.wait_for_message('/camera/depth/image_raw', Image)
  #depth_img = bridge.imgmsg_to_cv2(deptho)
  #rgb_img = bridge.imgmsg_to_cv2(rgbo)
  depth_img = image.DepthImage.from_tiff('/home/felipe/code/APC/ggcnn/datasets/cornell/03/pcd0313d.tiff')
  rgb_img = image.Image.from_file('/home/felipe/code/APC/ggcnn/datasets/cornell/03/pcd0313r.png')

  #rgb_img=cv2.imread("/content/grasping_task/test_img/bin.jpg", cv2.IMREAD_UNCHANGED)
  rgb_img= cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
  plt.imshow(rgb_img)
  plt.show()

  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(1, 1, 1)
  ax.imshow(depth_img, cmap='gray')
  ax.set_title('Depth')
  ax.axis('off')
  plt.show()

  # Crop first obhect detected
  detections = {"detection_boxes": [[[0, 0, 1, 1]]]}
  roi=[0,0,640, 480]
  points_out, ang_out, width_out=predict_grasps(depth_img, detections, roi, filters=(5.0, 4.0, 2.0), width_scale=70, preprocess=True )


  # Detect grasp hypothesis
  grasps = detect_grasps(points_out, ang_out, 0.5, width_img=width_out, no_grasps=10)
  for g in grasps:
      print(g)


  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(2, 2, 1)
  ax.imshow(depth_img, cmap='gray')
  ax.set_title('Depth')
  ax.axis('off')

  ax = fig.add_subplot(2, 2, 2)
  plot = ax.imshow(points_out, cmap='jet', vmin=0, vmax=1)
  ax.set_title('quality')
  ax.axis('off')

  ax = fig.add_subplot(2, 2, 3)
  plot = ax.imshow(ang_out, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
  ax.set_title('Angle')
  ax.axis('off')

  ax = fig.add_subplot(2, 2, 4)
  plot = ax.imshow(width_out, cmap='hsv', vmin=0, vmax=150)
  ax.set_title('width')
  ax.axis('off')
  plt.colorbar(plot)
  plt.show()

  #Transform to distance
  x, y, z, ang, rwidth =rvalues(grasps, depth_img, roi, detections)



  #Publish data
  #cmd_msg = Float32MultiArray()
  #cmd_msg.data = [x, y, z, ang, rwidth]
  #cmd_pub.publish(cmd_msg)

  #pose = geometry_msgs.msg.Pose()
  #pose.position.x = z
  #pose.position.y = -x
  #pose.position.z = -y
  #q = tft.quaternion_from_euler(0, 1.5, ang)
  #punto1.orientation.x =q[0]
  #punto1.orientation.y =q[1]
  #punto1.orientation.z =q[2]
  #punto1.orientation.w =q[3]
  #punto2 = convert_pose(punto1,"cam","world")
while not rospy.is_shutdown():
    main()
    rate.sleep()