#!/usr/bin/python3

import sys
from tokenize import Number

from rospkg import RosPack
import rospy
import dlib
import cv2
import numpy as np
import tf2_geometry_msgs
import tf2_ros
import os
from os import listdir

import actionlib

from os.path import dirname, join

#import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Int32, String
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray
from sound_play.libsoundplay import SoundClient
import face_recognition
from random import random, seed
from rospy_message_converter import message_converter
from enum import Enum

import torch
from torchvision import datasets, models, transforms
import os
import numpy
import PIL
import matplotlib.pyplot as plt
import time
import numpy as np

start_images = []


class Person(Enum):
    ANA = 0
    NINA = 1
    GARGAMEL = 2
    IRENA = 3
    MATEJA = 4


def recognize(image):
    input_size = 224
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    class_dict = {0: 'Ana',
                  1: 'Gargamel',
                  2: 'Irena',
                  3: 'Mateja',
                  4: 'Nina'}

    model_path = '/home/miha/FRI/RIS/people_rec/Foodero_dataset/Foodero/best_foodero_model.pt'

    model = torch.load(model_path)
    model.eval()

    img_p = PIL.Image.open(image)

    img = data_transforms['train'](img_p).unsqueeze(0)
    pred = model(img)

    pred_np = pred.cpu().detach().numpy().squeeze()
    class_ind = np.argmax(pred_np)
    return class_dict[class_ind]


def dist(pose1, pose2, min_dist):
    """
    Return true if poses closer that min_dist
    """
    p1 = np.array([pose1.x, pose1.y, pose1.z])
    p2 = np.array([pose2.x, pose2.y, pose2.z])
    dist = np.linalg.norm(p1 - p2)

    return dist != np.nan and dist < min_dist


def detected(pose, pose_array, min_dist=0.5):
    """
    Return true, if we have not yet detected the pose in pose_array
    """
    for temp in pose_array:
        if dist(pose.position, temp.position, min_dist):
            return False

    return True


def robot_location(msg):
    global robot_pose
    robot_pose = PoseWithCovarianceStamped()
    robot_pose = msg


class face_localizer:
    def __init__(self):
        rospy.init_node('face_localizer', anonymous=True)

        self.sounds_dir = rospy.get_param('/sound_play/sounds_dir', './')

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # The function for performin HOG face detection
        #self.face_detector = dlib.get_frontal_face_detector()
        protoPath = join(dirname(__file__), "deploy.prototxt.txt")
        modelPath = join(dirname(__file__),
                         "res10_300x300_ssd_iter_140000.caffemodel")

        self.face_net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        # A help variable for holding the dimensions of the image
        self.dims = (0, 0, 0)

        # Marker array object used for showing markers in Rviz
        self.marker_array = MarkerArray()
        self.marker_num = 1

        # Subscribe to the image and/or depth topic
        # self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        # self.depth_sub = rospy.Subscriber("/camera/depth_registered/image_raw", Image, self.depth_callback)

        # Publiser for the visualization markers
        self.markers_pub = rospy.Publisher('face_markers',
                                           MarkerArray,
                                           queue_size=1000)
        self.location_sub = rospy.Subscriber('/amcl_pose',
                                             PoseWithCovarianceStamped,
                                             robot_location)
        self.face_num_pub = rospy.Publisher('/num_faces', Int32, queue_size=10)
        self.face_poses_pub = rospy.Publisher(
            '/face_poses', PoseArray, queue_size=10)
        self.face_ix_pub = rospy.Publisher('/face_ix', String, queue_size=10)
        # Object we use for transforming between coordinate frames
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        # Initialize action client
        self.ac = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.ac.wait_for_server()

        # Initialize sound handler
        self.soundhandle = SoundClient()

        self.faces_found = 0
        self.pose_array = []
        self.face_markers = {}
        self.face_poses = ''
        self.pose_arr = PoseArray()
        
    def get_pose(self, coords, dist, stamp):
        # Calculate the position of the detected face

        k_f = 554  # kinect focal length in pixels

        x1, x2, y1, y2 = coords

        face_x = self.dims[1] / 2 - (x1 + x2) / 2.
        face_y = self.dims[0] / 2 - (y1 + y2) / 2.

        angle_to_target = np.arctan2(face_x, k_f)

        # Get the angles in the base_link relative coordinate system
        x, y = dist * np.cos(angle_to_target), dist * np.sin(angle_to_target)

        # Define a stamped message for transformation - directly in "base_link"
        #point_s = PointStamped()
        #point_s.point.x = x
        #point_s.point.y = y
        #point_s.point.z = 0.3
        #point_s.header.frame_id = "base_link"
        #point_s.header.stamp = rospy.Time(0)

        # Define a stamped message for transformation - in the "camera rgb frame"
        point_s = PointStamped()
        point_s.point.x = -y
        point_s.point.y = 0
        point_s.point.z = x
        point_s.header.frame_id = "camera_rgb_optical_frame"
        point_s.header.stamp = stamp

        # Get the point in the "map" coordinate system
        try:
            point_world = self.tf_buf.transform(point_s, "map")
            pose = Pose()
            pose.position.x = point_world.point.x
            pose.position.y = point_world.point.y
            pose.position.z = point_world.point.z
        except Exception as e:
            print(e)
            pose = None

        return pose

    def find_faces(self):
        # print('I got a new image!')

        # Get the next rgb and depth images that are posted from the camera
        try:
            rgb_image_message = rospy.wait_for_message("/camera/rgb/image_raw",
                                                       Image)
        except Exception as e:
            print(e)
            return 0

        try:
            depth_image_message = rospy.wait_for_message(
                "/camera/depth/image_raw", Image)
        except Exception as e:
            print(e)
            return 0

        # Convert the images into a OpenCV (numpy) format

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_message, "bgr8")
        except CvBridgeError as e:
            print(e)

        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_message,
                                                    "32FC1")
        except CvBridgeError as e:
            print(e)

        # Set the dimensions of the image
        self.dims = rgb_image.shape
        h = self.dims[0]
        w = self.dims[1]

        # Detect the faces in the image
        #face_rectangles = self.face_detector(rgb_image, 0)
        blob = cv2.dnn.blobFromImage(cv2.resize(rgb_image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        self.face_net.setInput(blob)
        face_detections = self.face_net.forward()

        for i in range(0, face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]
            if confidence > 0.5:
                box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                box = box.astype('int')
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                # Extract region containing face
                face_region = rgb_image[y1:y2, x1:x2]

                # cv2.imshow("depth", depth_image)
                # cv2.waitKey(0)
                # Find the distance to the detected face
                face_distance = float(np.nanmean(depth_image[y1:y2, x1:x2]))
                print('Distance to face', face_distance)

                # Get the time that the depth image was recieved
                depth_time = depth_image_message.header.stamp

                # Find the location of the detected face
                pose = self.get_pose((x1, x2, y1, y2), face_distance,
                                     depth_time)

                if pose is not None:
                    # Check if pose is valid and if we have already detected it
                    if detected(pose, self.pose_array):
                        # Add pose to detected poses
                        self.pose_array.append(pose)
                        print("::: pose ::: ", pose)

                        # FACE RECOGNITION
                        face_region = cv2.normalize(
                            face_region, face_region, 0, 255, cv2.NORM_MINMAX)

                        # path = '/home/miha/ROS/src/homework4/scripts/recognize_image'
                        # name = 'image.jpg'
                        # cv2.imwrite(os.path.join(path , name), face_region)

                        # name = recognize(os.path.join(path , name))
                        # print("NAME:")
                        # print(name)

                        # Create a marker used for visualization
                        self.marker_num += 1
                        marker = Marker()
                        #marker.text = name
                        marker.header.stamp = rospy.Time(0)
                        marker.header.frame_id = 'map'
                        marker.pose = pose
                        marker.type = Marker.SPHERE
                        marker.action = Marker.ADD
                        marker.frame_locked = False
                        marker.lifetime = rospy.Duration.from_sec(300)
                        marker.id = self.marker_num
                        marker.scale = Vector3(0.2, 0.2, 0.2)
                        marker.color = ColorRGBA(0, 1, 0, 1)
                        self.marker_array.markers.append(marker)

                        self.markers_pub.publish(self.marker_array)

                        # Move towards detected face
                        map_goal = MoveBaseGoal()
                        map_goal.target_pose.header.frame_id = "map"
                        map_goal.target_pose.pose.orientation.z = robot_pose.pose.pose.orientation.z
                        map_goal.target_pose.pose.orientation.w = robot_pose.pose.pose.orientation.w
                        # map_goal.target_pose.pose.position.x = np.mean(
                        #     [robot_pose.pose.pose.position.x, pose.position.x])
                        # map_goal.target_pose.pose.position.y = np.mean(
                        #     [robot_pose.pose.pose.position.y, pose.position.y])
                        x1 = pose.position.x
                        y1 = pose.position.y
                        x0 = robot_pose.pose.pose.position.x
                        y0 = robot_pose.pose.pose.position.y
                        # dn = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
                        t = 2/3
                        map_goal.target_pose.pose.position.x = (
                            1 - t)*x0 + t*x1
                        map_goal.target_pose.pose.position.y = (
                            1 - t)*y0 + t*y1
                        map_goal.target_pose.header.stamp = rospy.Time()

                        print("MARKER:")
                        print(self.marker_array.markers)

                        face_region = cv2.normalize(
                            face_region, face_region, 0, 255, cv2.NORM_MINMAX)

                        path = '/home/miha/ROS/src/homework4/scripts/recognize_image'
                        name = 'image.jpg'
                        #cv2.imwrite(os.path.join(path, name), face_region)

                        name = recognize(os.path.join(path, name))
                        print("NAME:")
                        self.soundhandle.say("Hello")
                        # self.soundhandle.say(name)

                        name = name + ' '
                        self.pose_arr.poses.append(map_goal.target_pose.pose)
                        self.face_poses = self.face_poses + name
                        self.faces_found += 1

                        # message = message_converter.convert_dictionary_to_ros_message('std_msgs/String', self.face_markers)
                        self.face_poses_pub.publish(self.pose_arr)
                        self.face_num_pub.publish(self.faces_found)
                        self.face_ix_pub.publish(self.face_poses)


    def depth_callback(self, data):

        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)

        # Do the necessairy conversion so we can visuzalize it in OpenCV

        image_1 = depth_image / np.nanmax(depth_image)
        image_1 = image_1 * 255

        image_viz = np.array(image_1, dtype=np.uint8)

        #cv2.imshow("Depth window", image_viz)
        # cv2.waitKey(1)

        # plt.imshow(depth_image)
        # plt.show()


def main():

    # for img in os.listdir('/home/miha/ROS/src/homework4/scripts/images'):
    #     path = '/home/miha/ROS/src/homework4/scripts/images' + '/' + img
    #     image = cv2.imread(path)

    #     th = 20 # defines the value below which a pixel is considered "black"
    #     black_pixels = np.where(
    #         (image[:, :, 0] == 255) &
    #         (image[:, :, 1] == 255) &
    #         (image[:, :, 2] == 255)
    #     )

    # set those pixels to white
    #image[black_pixels] = [0, 0, 0]

    #start_images.append((img, image))
    # print(start_images)

    face_finder = face_localizer()
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        face_finder.find_faces()
        rate.sleep()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    #print("I launched")
    seed(1)
    main()

#!/usr/bin/python3

# vid koda start

# import sys
# import rospy
# import dlib
# import cv2
# import numpy as np
# import tf2_geometry_msgs
# import tf2_ros
# import os
# from os import listdir

# import actionlib

# from os.path import dirname, join

# #import matplotlib.pyplot as plt
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import PointStamped, Vector3, Pose
# from cv_bridge import CvBridge, CvBridgeError
# from visualization_msgs.msg import Marker, MarkerArray
# from std_msgs.msg import ColorRGBA
# from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
# from geometry_msgs.msg import PoseWithCovarianceStamped
# from sound_play.libsoundplay import SoundClient

# start_images = []
# names = []

# def dist(pose1, pose2, min_dist):
#     """
#     Return true if poses closer that min_dist
#     """
#     p1 = np.array([pose1.x, pose1.y, pose1.z])
#     p2 = np.array([pose2.x, pose2.y, pose2.z])
#     dist = np.linalg.norm(p1 - p2)

#     return dist != np.nan and dist < min_dist


# def detected(pose, pose_array, min_dist=0.5):
#     """
#     Return true, if we have not yet detected the pose in pose_array
#     """
#     for temp in pose_array:
#         if dist(pose.position, temp.position, min_dist):
#             return False

#     return True


# def robot_location(msg):
#     global robot_pose
#     robot_pose = PoseWithCovarianceStamped()
#     robot_pose = msg


# class face_localizer:
#     def __init__(self):
#         rospy.init_node('face_localizer', anonymous=True)

#         self.sounds_dir = rospy.get_param('/sound_play/sounds_dir', './')

#         # An object we use for converting images between ROS format and OpenCV format
#         self.bridge = CvBridge()

#         # The function for performin HOG face detection
#         #self.face_detector = dlib.get_frontal_face_detector()
#         protoPath = join(dirname(__file__), "deploy.prototxt.txt")
#         modelPath = join(dirname(__file__),
#                          "res10_300x300_ssd_iter_140000.caffemodel")

#         self.face_net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

#         # A help variable for holding the dimensions of the image
#         self.dims = (0, 0, 0)

#         # Marker array object used for showing markers in Rviz
#         self.marker_array = MarkerArray()
#         self.marker_num = 1

#         # Subscribe to the image and/or depth topic
#         #self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
#         #self.depth_sub = rospy.Subscriber("/camera/depth_registered/image_raw", Image, self.depth_callback)

#         # Publiser for the visualization markers
#         self.markers_pub = rospy.Publisher('face_markers',
#                                            MarkerArray,
#                                            queue_size=1000)
#         self.location_sub = rospy.Subscriber('/amcl_pose',
#                                              PoseWithCovarianceStamped,
#                                              robot_location)

#         # Object we use for transforming between coordinate frames
#         self.tf_buf = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

#         # Initialize action client
#         self.ac = actionlib.SimpleActionClient('move_base', MoveBaseAction)
#         self.ac.wait_for_server()

#         # Initialize sound handler
#         self.soundhandle = SoundClient()

#         self.pose_array = []

#     def get_pose(self, coords, dist, stamp):
#         # Calculate the position of the detected face

#         k_f = 554  # kinect focal length in pixels

#         x1, x2, y1, y2 = coords

#         face_x = self.dims[1] / 2 - (x1 + x2) / 2.
#         face_y = self.dims[0] / 2 - (y1 + y2) / 2.

#         angle_to_target = np.arctan2(face_x, k_f)

#         # Get the angles in the base_link relative coordinate system
#         x, y = dist * np.cos(angle_to_target), dist * np.sin(angle_to_target)

#         ### Define a stamped message for transformation - directly in "base_link"
#         #point_s = PointStamped()
#         #point_s.point.x = x
#         #point_s.point.y = y
#         #point_s.point.z = 0.3
#         #point_s.header.frame_id = "base_link"
#         #point_s.header.stamp = rospy.Time(0)

#         # Define a stamped message for transformation - in the "camera rgb frame"
#         point_s = PointStamped()
#         point_s.point.x = -y
#         point_s.point.y = 0
#         point_s.point.z = x
#         point_s.header.frame_id = "camera_rgb_optical_frame"
#         point_s.header.stamp = stamp

#         # Get the point in the "map" coordinate system
#         try:
#             point_world = self.tf_buf.transform(point_s, "map")

#             # Create a Pose object with the same position
#             pose = Pose()
#             pose.position.x = point_world.point.x
#             pose.position.y = point_world.point.y
#             pose.position.z = point_world.point.z
#         except Exception as e:
#             print(e)
#             pose = None

#         return pose

#     def find_faces(self):
#         # print('I got a new image!')

#         # Get the next rgb and depth images that are posted from the camera
#         try:
#             rgb_image_message = rospy.wait_for_message("/camera/rgb/image_raw",
#                                                        Image)
#         except Exception as e:
#             print(e)
#             return 0

#         try:
#             depth_image_message = rospy.wait_for_message(
#                 "/camera/depth/image_raw", Image)
#         except Exception as e:
#             print(e)
#             return 0

#         # Convert the images into a OpenCV (numpy) format

#         try:
#             rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_message, "bgr8")
#         except CvBridgeError as e:
#             print(e)

#         try:
#             depth_image = self.bridge.imgmsg_to_cv2(depth_image_message,
#                                                     "32FC1")
#         except CvBridgeError as e:
#             print(e)

#         # Set the dimensions of the image
#         self.dims = rgb_image.shape
#         h = self.dims[0]
#         w = self.dims[1]

#         # Tranform image to gayscale
#         #gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

#         # Do histogram equlization
#         #img = cv2.equalizeHist(gray)

#         # Detect the faces in the image
#         #face_rectangles = self.face_detector(rgb_image, 0)
#         blob = cv2.dnn.blobFromImage(cv2.resize(rgb_image, (300, 300)), 1.0,
#                                      (300, 300), (104.0, 177.0, 123.0))


#         self.face_net.setInput(blob)
#         face_detections = self.face_net.forward()

#         # print("BLOB:")
#         # print(np.shape(face_detections))
#         # cv2.imshow("blob",face_detections)
#         # cv2.waitKey(0)

#         for i in range(0, face_detections.shape[2]):
#             confidence = face_detections[0, 0, i, 2]
#             if confidence > 0.5:
#                 box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 box = box.astype('int')
#                 x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

#                 # Extract region containing face
#                 face_region = rgb_image[y1:y2, x1:x2]
#                 maxErr = 100000
#                 bestName = 'bestName'

#                 #print("START IMAGES")
#                 # print(start_images)

#                 # for name,image in start_images:
#                 #     print(name)
#                 #     print(image)
#                 #     w,h,_ = np.shape(face_region)

#                 #     print(np.shape(image))
#                 #     print(np.shape(face_region))

#                 #     resized = cv2.resize(image,(h,w), interpolation= cv2.INTER_AREA)

#                 #     print(np.shape(resized))
#                 #     print(np.shape(face_region))

#                 #     face_region = cv2.normalize(face_region,face_region,0,255,cv2.NORM_MINMAX)

#                 #     errorL2 = cv2.norm( face_region, resized, cv2.NORM_L2 )
#                 #     # similarity = 1 - errorL2 / ( w * h)

#                 #     if errorL2 < maxErr:
#                 #         maxErr = errorL2
#                 #         bestName = name[:-4]

#                 # print(bestName)
#                 # Visualize the extracted face
#                 #cv2.imshow("ImWindow", face_region)
#                 #cv2.waitKey(1)

#                 # Find the distance to the detected face
#                 face_distance = float(np.nanmean(depth_image[y1:y2, x1:x2]))

#                 print('Distance to face', face_distance)

#                 # Get the time that the depth image was recieved
#                 depth_time = depth_image_message.header.stamp

#                 # Find the location of the detected face
#                 pose = self.get_pose((x1, x2, y1, y2), face_distance,
#                                      depth_time)

#                 if pose is not None:
#                     # Check if pose is valid and if we have already detected it
#                     if detected(pose, self.pose_array):
#                         # Add pose to detected poses
#                         self.pose_array.append(pose)
#                         print("::: pose ::: ", pose)

#                         # Create a marker used for visualization
#                         self.marker_num += 1
#                         marker = Marker()
#                         marker.header.stamp = rospy.Time(0)
#                         marker.header.frame_id = 'map'
#                         marker.pose = pose
#                         marker.type = Marker.SPHERE
#                         marker.action = Marker.ADD
#                         marker.frame_locked = False
#                         marker.lifetime = rospy.Duration.from_sec(300)
#                         marker.id = self.marker_num
#                         marker.scale = Vector3(0.2, 0.2, 0.2)
#                         marker.color = ColorRGBA(0, 1, 0, 1)
#                         self.marker_array.markers.append(marker)

#                         self.markers_pub.publish(self.marker_array)

#                         # Move towards detected face
#                         map_goal = MoveBaseGoal()
#                         map_goal.target_pose.header.frame_id = "map"
#                         map_goal.target_pose.pose.orientation.z = robot_pose.pose.pose.orientation.z
#                         map_goal.target_pose.pose.orientation.w = robot_pose.pose.pose.orientation.w
#                         # map_goal.target_pose.pose.position.x = np.mean(
#                         #     [robot_pose.pose.pose.position.x, pose.position.x])
#                         # map_goal.target_pose.pose.position.y = np.mean(
#                         #     [robot_pose.pose.pose.position.y, pose.position.y])
#                         x1 = pose.position.x
#                         y1 = pose.position.y
#                         x0 = robot_pose.pose.pose.position.x
#                         y0 = robot_pose.pose.pose.position.y
#                         # dn = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
#                         t = 2/3
#                         map_goal.target_pose.pose.position.x = (1 - t)*x0 + t*x1
#                         map_goal.target_pose.pose.position.y = (1 - t)*y0 + t*y1
#                         map_goal.target_pose.header.stamp = rospy.Time()
#                         self.ac.cancel_all_goals()
#                         self.ac.send_goal(map_goal)
#                         self.ac.wait_for_result(rospy.Duration(3))

#                         print("MARKER:")
#                         print(self.marker_array.markers)

#                         for name,image in start_images:
#                             print(name)
#                             print(image)
#                             w,h,_ = np.shape(face_region)

#                             print(np.shape(image))
#                             print(np.shape(face_region))

#                             resized = cv2.resize(image,(h,w), interpolation= cv2.INTER_AREA)

#                             print(np.shape(resized))
#                             print(np.shape(face_region))

#                             face_region = cv2.normalize(face_region,face_region,0,255,cv2.NORM_MINMAX)

#                             errorL2 = cv2.norm( face_region, resized, cv2.NORM_L2 )

#                             # cv2.imshow("mokotardevica",face_region)
#                             # cv2.waitKey(1)
#                             # cv2.imshow("vidpeder",resized)
#                             # cv2.waitKey(1)
#                             # similarity = 1 - errorL2 / ( w * h)

#                             if errorL2 < maxErr:
#                                 maxErr = errorL2
#                                 bestName = name[:-4]

#                         print(bestName)

#                         # "Greet" the face
#                         self.soundhandle.playWave(
#                             self.sounds_dir + "hey-baby.wav"
#                         )

#     def depth_callback(self, data):

#         try:
#             depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
#         except CvBridgeError as e:
#             print(e)

#         # Do the necessairy conversion so we can visuzalize it in OpenCV

#         image_1 = depth_image / np.nanmax(depth_image)
#         image_1 = image_1 * 255

#         image_viz = np.array(image_1, dtype=np.uint8)

#         #cv2.imshow("Depth window", image_viz)
#         #cv2.waitKey(1)

#         #plt.imshow(depth_image)
#         #plt.show()


# def main():

#     for img in os.listdir('/home/miha/ROS/src/homework4/scripts/images'):
#         #print("SADDSDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
#         path = '/home/miha/ROS/src/homework4/scripts/images' + '/' + img
#         image  = cv2.imread(path)
#         #print(image)
#         start_images.append((img, image))
#     #print(start_images)

#     face_finder = face_localizer()
#     rate = rospy.Rate(1)
#     while not rospy.is_shutdown():
#         face_finder.find_faces()
#         rate.sleep()

#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     main()


# vid koda end

#!/usr/bin/python3

# import sys
# import rospy
# import dlib
# import cv2
# import numpy as np
# import tf2_geometry_msgs
# import tf2_ros

# import actionlib

# from os.path import dirname, join

# #import matplotlib.pyplot as plt
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import PointStamped, Vector3, Pose
# from cv_bridge import CvBridge, CvBridgeError
# from visualization_msgs.msg import Marker, MarkerArray
# from std_msgs.msg import ColorRGBA
# from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
# from geometry_msgs.msg import PoseWithCovarianceStamped
# from sound_play.libsoundplay import SoundClient


# def dist(pose1, pose2, min_dist):
#     """
#     Return true if poses closer that min_dist
#     """
#     p1 = np.array([pose1.x, pose1.y, pose1.z])
#     p2 = np.array([pose2.x, pose2.y, pose2.z])
#     dist = np.linalg.norm(p1 - p2)

#     return dist != np.nan and dist < min_dist


# def detected(pose, pose_array, min_dist=0.5):
#     """
#     Return true, if we have not yet detected the pose in pose_array
#     """
#     for temp in pose_array:
#         if dist(pose.position, temp.position, min_dist):
#             return False

#     return True


# def robot_location(msg):
#     global robot_pose
#     robot_pose = PoseWithCovarianceStamped()
#     robot_pose = msg


# class face_localizer:
#     def __init__(self):
#         rospy.init_node('face_localizer', anonymous=True)

#         self.sounds_dir = rospy.get_param('/sound_play/sounds_dir', './')

#         # An object we use for converting images between ROS format and OpenCV format
#         self.bridge = CvBridge()

#         # The function for performin HOG face detection
#         #self.face_detector = dlib.get_frontal_face_detector()
#         protoPath = join(dirname(__file__), "deploy.prototxt.txt")
#         modelPath = join(dirname(__file__),
#                          "res10_300x300_ssd_iter_140000.caffemodel")

#         self.face_net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

#         # A help variable for holding the dimensions of the image
#         self.dims = (0, 0, 0)

#         # Marker array object used for showing markers in Rviz
#         self.marker_array = MarkerArray()
#         self.marker_num = 1

#         # Subscribe to the image and/or depth topic
#         # self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
#         # self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)

#         # Publiser for the visualization markers
#         self.markers_pub = rospy.Publisher('face_markers',
#                                            MarkerArray,
#                                            queue_size=1000)
#         self.location_sub = rospy.Subscriber('/amcl_pose',
#                                              PoseWithCovarianceStamped,
#                                              robot_location)

#         # Object we use for transforming between coordinate frames
#         self.tf_buf = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

#         # Initialize action client
#         self.ac = actionlib.SimpleActionClient('move_base', MoveBaseAction)
#         self.ac.wait_for_server()

#         # Initialize sound handler
#         self.soundhandle = SoundClient()

#         self.pose_array = []

#     def get_pose(self, coords, dist, stamp):
#         # Calculate the position of the detected face

#         k_f = 554  # kinect focal length in pixels

#         x1, x2, y1, y2 = coords

#         face_x = self.dims[1] / 2 - (x1 + x2) / 2.
#         face_y = self.dims[0] / 2 - (y1 + y2) / 2.

#         angle_to_target = np.arctan2(face_x, k_f)

#         # Get the angles in the base_link relative coordinate system
#         x, y = dist * np.cos(angle_to_target), dist * np.sin(angle_to_target)

#         ### Define a stamped message for transformation - directly in "base_link"
#         #point_s = PointStamped()
#         #point_s.point.x = x
#         #point_s.point.y = y
#         #point_s.point.z = 0.3
#         #point_s.header.frame_id = "base_link"
#         #point_s.header.stamp = rospy.Time(0)

#         # Define a stamped message for transformation - in the "camera rgb frame"
#         point_s = PointStamped()
#         point_s.point.x = -y
#         point_s.point.y = 0
#         point_s.point.z = x
#         point_s.header.frame_id = "camera_rgb_optical_frame"
#         point_s.header.stamp = stamp

#         # Get the point in the "map" coordinate system
#         try:
#             point_world = self.tf_buf.transform(point_s, "map")

#             # Create a Pose object with the same position
#             pose = Pose()
#             pose.position.x = point_world.point.x
#             pose.position.y = point_world.point.y
#             pose.position.z = point_world.point.z
#         except Exception as e:
#             print(e)
#             pose = None

#         return pose

#     def find_faces(self):
#         # print('I got a new image!')

#         # Get the next rgb and depth images that are posted from the camera
#         try:
#             rgb_image_message = rospy.wait_for_message("/camera/rgb/image_raw",
#                                                        Image)
#         except Exception as e:
#             print(e)
#             return 0

#         try:
#             depth_image_message = rospy.wait_for_message(
#                 "/camera/depth/image_raw", Image)
#         except Exception as e:
#             print(e)
#             return 0

#         # Convert the images into a OpenCV (numpy) format

#         try:
#             rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_message, "bgr8")
#         except CvBridgeError as e:
#             print(e)

#         try:
#             depth_image = self.bridge.imgmsg_to_cv2(depth_image_message,
#                                                     "32FC1")
#         except CvBridgeError as e:
#             print(e)

#         # Set the dimensions of the image
#         self.dims = rgb_image.shape
#         h = self.dims[0]
#         w = self.dims[1]

#         # Tranform image to gayscale
#         #gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

#         # Do histogram equlization
#         #img = cv2.equalizeHist(gray)

#         # Detect the faces in the image
#         #face_rectangles = self.face_detector(rgb_image, 0)
#         blob = cv2.dnn.blobFromImage(cv2.resize(rgb_image, (300, 300)), 1.0,
#                                      (300, 300), (104.0, 177.0, 123.0))
#         self.face_net.setInput(blob)
#         face_detections = self.face_net.forward()

#         for i in range(0, face_detections.shape[2]):
#             confidence = face_detections[0, 0, i, 2]
#             if confidence > 0.5:
#                 box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 box = box.astype('int')
#                 x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

#                 # Extract region containing face
#                 face_region = rgb_image[y1:y2, x1:x2]

#                 # Visualize the extracted face
#                 #cv2.imshow("ImWindow", face_region)
#                 #cv2.waitKey(1)

#                 # Find the distance to the detected face
#                 face_distance = float(np.nanmean(depth_image[y1:y2, x1:x2]))

#                 print('Distance to face', face_distance)

#                 # Get the time that the depth image was recieved
#                 depth_time = depth_image_message.header.stamp

#                 # Find the location of the detected face
#                 pose = self.get_pose((x1, x2, y1, y2), face_distance,
#                                      depth_time)

#                 if pose is not None:
#                     # Check if pose is valid and if we have already detected it
#                     if detected(pose, self.pose_array):
#                         # Add pose to detected poses
#                         self.pose_array.append(pose)
#                         print("::: pose ::: ", pose)

#                         # Create a marker used for visualization
#                         self.marker_num += 1
#                         marker = Marker()
#                         marker.header.stamp = rospy.Time(0)
#                         marker.header.frame_id = 'map'
#                         marker.pose = pose
#                         marker.type = Marker.SPHERE
#                         marker.action = Marker.ADD
#                         marker.frame_locked = False
#                         marker.lifetime = rospy.Duration.from_sec(300)
#                         marker.id = self.marker_num
#                         marker.scale = Vector3(0.2, 0.2, 0.2)
#                         marker.color = ColorRGBA(0, 1, 0, 1)
#                         self.marker_array.markers.append(marker)

#                         self.markers_pub.publish(self.marker_array)

#                         # Move towards detected face
#                         map_goal = MoveBaseGoal()
#                         map_goal.target_pose.header.frame_id = "map"
#                         map_goal.target_pose.pose.orientation.z = robot_pose.pose.pose.orientation.z
#                         map_goal.target_pose.pose.orientation.w = robot_pose.pose.pose.orientation.w
#                         map_goal.target_pose.pose.position.x = np.mean(
#                             [robot_pose.pose.pose.position.x, pose.position.x])
#                         map_goal.target_pose.pose.position.y = np.mean(
#                             [robot_pose.pose.pose.position.y, pose.position.y])
#                         map_goal.target_pose.header.stamp = rospy.Time()
#                         self.ac.cancel_all_goals()
#                         self.ac.send_goal(map_goal)
#                         self.ac.wait_for_result(rospy.Duration(3))

#                         # "Greet" the face
#                         self.soundhandle.playWave(
#                             self.sounds_dir + "hey-baby.wav"
#                         )

#     def depth_callback(self, data):

#         try:
#             depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
#         except CvBridgeError as e:
#             print(e)

#         # Do the necessairy conversion so we can visuzalize it in OpenCV

#         image_1 = depth_image / np.nanmax(depth_image)
#         image_1 = image_1 * 255

#         image_viz = np.array(image_1, dtype=np.uint8)

#         #cv2.imshow("Depth window", image_viz)
#         #cv2.waitKey(1)

#         #plt.imshow(depth_image)
#         #plt.show()


# def main():

#     face_finder = face_localizer()

#     rate = rospy.Rate(1)
#     while not rospy.is_shutdown():
#         face_finder.find_faces()
#         rate.sleep()

#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     main()
