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
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import matplotlib.pyplot as plt
from agnostic_segmentation import agnostic_segmentation
from ggcnn.models import ggcnn2
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
  rgbo = rospy.wait_for_message('/camera/color/image_raw', Image)
  deptho = rospy.wait_for_message('/camera/depth/image_raw', Image)
  depth_img = bridge.imgmsg_to_cv2(deptho)
  rgb_img = bridge.imgmsg_to_cv2(rgbo)
  #rgb_img=cv2.imread("/content/grasping_task/test_img/bin.jpg", cv2.IMREAD_UNCHANGED)
  rgb_img= cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
  plt.imshow(rgb_img)
  plt.show()
  ###### Detect bin
  use_retina=True
  if use_retina:
    detection_model=create_retina(num_classes=2, pipeline_config = '/content/grasping_task/models/retina_config/config/pipeline.config', checkpoint_path = "/content/grasping_task/models/retina_config/checkpoint/ckpt-1") 
  resized = cv2.resize(rgb_img, (640,480), interpolation = cv2.INTER_AREA)
  if use_retina:
    label_id_offset = 1
    input_tensor = tf.convert_to_tensor(np.expand_dims(resized, axis=0), dtype=tf.float32)
    detections = detect(detection_model,input_tensor)
  else:
    detections = {"detection_boxes": [[[0, 0, 1, 1]]]}
  ###### Crop bin image
  shape=resized.shape
  bin_img=resized[int(shape[0]*detections['detection_boxes'][0][0][0]):int(shape[0]*detections['detection_boxes'][0][0][2]),int(shape[1]*detections['detection_boxes'][0][0][1]):int(shape[1]*detections['detection_boxes'][0][0][3])]
  plt.imshow(bin_img) 

  #### Agnostic_segmentation
  MODEL_PATH="/content/FAT_trained_Ml2R_bin_fine_tuned.pth"
  predictions = agnostic_segmentation.segment_image(bin_img, MODEL_PATH)
  seg_img = agnostic_segmentation.draw_segmented_image(bin_img, predictions)
  pred_boxes={}
  for i in range(len(predictions["instances"])):
    pred_boxes[i]=predictions["instances"][i].get_fields()["pred_boxes"].tensor.numpy()[0]
  print("# Objects: ", len(pred_boxes))
  zoom_p = 0.2
  roi = (pred_boxes[0]*np.array([1-zoom_p, 1-zoom_p, 1+zoom_p, 1+zoom_p])).astype(int)
  # Crop first obhect detected
  rgb_img=bin_img[roi[1]:roi[3], roi[0]:roi[2]]

  #Predict grasps
  points_out, ang_out, width_out=predict_grasps(depth_img, detections, roi, filters=(5.0, 4.0, 2.0), width_scale=70, preprocess=True )

  # Detect grasp hypothesis
  grasps = detect_grasps(points_out, ang_out, 0.5, width_img=width_out, no_grasps=10)
  for g in grasps:
      print(g)

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