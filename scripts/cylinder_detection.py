#!/usr/bin/python3
from cmath import nan
from pathlib import Path
import rospy
import cv2
import tf2_ros
import numpy as np
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Pose, Vector3, Point, Quaternion, PointStamped
from std_msgs.msg import Header, ColorRGBA, String, Int32
import matplotlib.pyplot as plt
import tf2_geometry_msgs
from visualization_msgs.msg import Marker, MarkerArray
from matplotlib.colors import rgb_to_hsv
from sound_play.libsoundplay import SoundClient
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, PoseArray
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import PIL
import torch
from torchvision import datasets, models, transforms
from tf.transformations import quaternion_from_euler
from homework2.srv import Normal_vector, Normal_vectorResponse
from rospy_message_converter import message_converter
import math
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
    class_dict = {0:'baklava',
                    1:'pica',
                    2:'pomfri',
                    3:'solata',
                    4:'torta'}

    
    model_path = '/home/miha/FRI/RIS/food_rec/Foodero_dataset/Foodero/best_foodero_model.pt'

    model = torch.load(model_path)
    model.eval()

    img_p = PIL.Image.open(image)

    img = data_transforms['train'](img_p).unsqueeze(0)
    pred = model(img)

    pred_np = pred.cpu().detach().numpy().squeeze()
    class_ind = np.argmax(pred_np)
    if pred_np[class_ind] > 6:
        return class_dict[class_ind]
    else:
        return None

def robot_location(msg):
    global robot_pose
    robot_pose = PoseWithCovarianceStamped()
    robot_pose = msg

status_to_text = {
    0: "PENDING",
    1: "ACTIVE",
    2: "PREEMPTED",
    3: "SUCCEEDED",
    4: "ABORTED",
    5: "REJECTED",
    6: "PREEMPTING",
    7: "RECALLING",
    8: "RECALLED",
    9: "LOST"
}
class Clustering:

    def __init__(self, threshold=1):
        self.clusters = []
        self.threshold = threshold

    def distance(self, pose1, pose2):
        return np.sqrt((pose1.x - pose2.x) ** 2 + (pose1.y - pose2.y) ** 2)

    def add(self, pose):
        new_cluster = True

        for cluster in self.clusters:
            for point in cluster:
                if self.distance(pose, point) < self.threshold:
                    cluster.append(pose)
                    new_cluster = False

                    if len(cluster) == 4:
                        return True

                    break
            if not new_cluster:
                break

        if new_cluster:
            self.clusters.append([pose])

        return False


class CylinderDetection:

    def __init__(self):
        rospy.init_node("cylinder_detection")
        self.bridge = CvBridge()

        self.scan_sub = message_filters.Subscriber("/arm_scan", LaserScan)
        self.image_sub = message_filters.Subscriber("/arm_camera/rgb/image_raw", Image)

        self.arm_pub = rospy.Publisher('/turtlebot_arm/arm_controller/command', JointTrajectory, queue_size=1)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.scan_sub, self.image_sub], 10, 0.5)
        self.ts.registerCallback(self.process_scan)


        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        self.marker_pub = rospy.Publisher(
            '/cylinder_markers', MarkerArray, queue_size=10)
        self.location_sub = rospy.Subscriber('/amcl_pose',
                                             PoseWithCovarianceStamped,
                                             robot_location)

        self.qr_sub = rospy.Subscriber('/qr/data', String, self.qr_callback)
        
        # rospy.wait_for_service("normal_vector")
        print("jajajajajajajajaajajjajajajaajajjajajajajajajajajajaja ")
        self.vector_service = rospy.ServiceProxy(
            "normal_vector", Normal_vector)

        self.speech_sub = rospy.Subscriber('/speech', String, self.speech_callback)
        self.message = ''
        self.face_num_sub = rospy.Subscriber('/num_faces', Int32, self.face_num_callback)
        self.face_poses_sub = rospy.Subscriber('/face_poses', PoseArray, self.face_poses_callback)
        self.face_ix_sub = rospy.Subscriber('/face_ix', String, self.face_ix_callback)

        self.cmd_vel_pub =  rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=1)

        self.ring_marker_sub = rospy.Subscriber('/ring_markers', MarkerArray, self.ring_callback)
        self.ac = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.ac.wait_for_server()
        rospy.sleep(0.5)
        self.marker_pub.publish([Marker(header=Header(
            frame_id="map", stamp=rospy.Time(0)), action=Marker.DELETEALL)])
        self.markers = []


        self.found_cylinders = 0
        self.cluster  = Clustering()

        self.soundhandle = SoundClient()
        self.voice = 'voice_kal_diphone'
        self.volume = 1.0

        self.food_markers = {}
        
        self.cylinder_goal_cnt = 0
        self.cylinder_goals = [[-1, 0], [2, -1.5],  [1.4, 0], [2, 2.5]]
        # self.cylinder_goals = [[2.3, -1.5]]
        # self.cylinder_goals = [[2, 2.5]]
        # self.cylinder_goals = [[1.5, 0]]
        # self.cylinder_goals = [[-1, 0]]

        self.face_markers = {
            "Nina": Pose(position=Point(0, -1.5, 0), orientation=Quaternion(*quaternion_from_euler(0, 0, -0.5 * math.pi, axes='sxyz'))),
            "Gargamel": Pose(position=Point(1.5, 2.5, 0), orientation=Quaternion(*quaternion_from_euler(0, 0, 0.5 * math.pi, axes='sxyz'))),
            "Mateja": Pose(position=Point(-1.3, 1.5, 0), orientation=Quaternion(*quaternion_from_euler(0, 0, 1 * math.pi, axes='sxyz'))),
            "Irena": Pose(position=Point(3.5, -0.5, 0), orientation=Quaternion(*quaternion_from_euler(0, 0, 0 * math.pi, axes='sxyz'))),
            "Ana": Pose(position=Point(0, 0, 0), orientation=Quaternion(*quaternion_from_euler(0, 0, 0 * math.pi, axes='sxyz')))
        }
        self.names = []
        self.pose_arr = PoseArray()
        self.face_poses = {}
        self.faces_found = 0
        self.final_goals = []
        self.goal_cnt = 0
        self.method = 'goToRestaurant'
        self.parking_pose = [2.5, 1.8]
        self.parking = False
        self.parking_cnt = 0
        self.order = ''
        self.ix = 0
        self.set_arm([0,0,0,0], 1)
        self.lol = ["torta", "solata", "pomfri", "pica"]
        self.delivering = False
        r = rospy.Rate(7)
        while not rospy.is_shutdown():
            self.publish_markers()
            # if self.found_cylinders == 1:
            if self.found_cylinders == 4 and self.faces_found == 5:
                self.found_cylinders += 1
                self.food_recognition_phase()
            if self.parking and self.parking_cnt <= 80 and self.order == '':
                print(self.parking_cnt)
                self.parking_cnt += 1
                self.park()
            elif self.parking_cnt > 80 or self.order != '' and self.delivering == False:
                r.sleep()
                self.parking_cnt = 0
                self.parking = False
                self.set_arm([0,0,0,0], 1)
                # twist = Twist()
                # twist.linear.x = 0
                # twist.linear.y = 0
                # twist.linear.z = 0
                # twist.angular.x = 0
                # twist.angular.y = 0
                # twist.angular.z = 1
                # self.cmd_vel_pub.publish(twist)
                # rospy.sleep(3)
                # twist = Twist()
                # twist.linear.x = 0.7
                # twist.linear.y = 0
                # twist.linear.z = 0
                # twist.angular.x = 0
                # twist.angular.y = 0
                # twist.angular.z = 0
                # self.cmd_vel_pub.publish(twist)
                # rospy.sleep(3)
                self.order = "Nina pomfri, Irena pica"
                self.deliver_food()
            r.sleep()

        rospy.spin()


    def speech_callback(self, message):
        self.message = message


    def face_poses_callback(self, pose_arr):
        self.pose_arr = pose_arr
        print("POSESSSSSSSSSSSSSSSSSSSSSSS ", self.pose_arr)
        if self.faces_found == 5:
            self.face_poses_sub.unregister()

    def face_ix_callback(self, string):
        names = string.data.split(' ')
        self.names = names
        if len(self.names) == 5:
            self.face_ix_sub.unregister()
            self.parse_pose_arr()


    def qr_callback(self, qr):
        self.order = qr.data
        print(self.order)
        self.qr_sub.unregister()
    
    def parse_pose_arr(self):
        for i in range(len(self.names)):
            self.face_poses[self.names[i]] = self.pose_arr.poses[i]
        print("FINAL GOALS", self.face_poses)


    def ring_callback(self, marker_array):
        # if marker_array.markers[0].pose.position.x != nan:
        #     self.parking_pose[0] = marker_array.markers[0].pose.position.x
        #     self.parking_pose[1] = marker_array.markers[0].pose.position.y
        
        print("PARKING POSE:", self.parking_pose)

    def face_num_callback(self, num):
        self.faces_found = num.data
        print("num of faces: ",self.faces_found)
        if self.faces_found == 5:
            self.face_num_sub.unregister()

    def parking_cb(self, status, result):
        if status == 3:
            self.soundhandle.say("Parking", self.voice, self.volume)
            self.set_arm([0, -0.8, 1.9, 0.1])
            self.parking = True
            rospy.sleep(2)
            self.park()


    def initiate_parking(self):
        # self.ac.cancel_all_goals()

        # angle = self.vector_service(0, self.parking_pose[0], self.parking_pose[1], "parking").angle
        angle = 1
        x_diff = np.cos(angle)
        y_diff = np.sin(-angle)
        yaw = angle + np.pi
        # print(self.parking_pose)
        # print(yaw)
        target_x = self.parking_pose[0] + x_diff * 0.5
        target_y = self.parking_pose[1] + y_diff * 0.5

        quat = quaternion_from_euler(0, 0, yaw)

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.orientation.z = quat[2]
        goal.target_pose.pose.orientation.w = quat[3]
        goal.target_pose.pose.position.x = target_x
        goal.target_pose.pose.position.y = target_y

        goal1 = MoveBaseGoal()
        goal1.target_pose.header.frame_id = 'map'
        goal1.target_pose.header.stamp = rospy.Time.now()
        goal1.target_pose.pose.orientation.z = -1
        goal1.target_pose.pose.orientation.w = 0
        goal1.target_pose.pose.position.x = 2.7
        goal1.target_pose.pose.position.y = 1.6
        print("Going to parking: ", goal1.target_pose.pose)
        self.ac.send_goal(goal1, done_cb=self.parking_cb)
   
   
    def park(self):
        try:
            image_message = rospy.wait_for_message("/arm_camera/rgb/image_raw", Image)
        except Exception as e:
            print(e)
            return

        try:
            image = self.bridge.imgmsg_to_cv2(image_message, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[:360, :]
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1=30, param2=10, minRadius=0, maxRadius=25)
        if circles is None:
            return
        circle = circles[0][0]
        center = image.shape[1] // 2

        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0

        center_margin = 30
        forward_margin = 30
        if circle[0] - center > center_margin:
            twist.angular.z = -1
        elif circle[0] - center < -center_margin:
            twist.angular.z = 1
        elif circle[1] < image.shape[0] - forward_margin:
            twist.linear.x = 0.15
        else:
            self.parking = False
        self.cmd_vel_pub.publish(twist)


    def done_callback(self, status, result):
        self.goal_cnt += 1
        if status == 2:
            rospy.loginfo("Goal pose "+str(self.goal_cnt)+" received a cancel request after it started executing, completed execution!")

        if status == 3:

            if self.method == 'goToRestaurant':
                self.soundhandle.say("Getting " + self.final_goals[self.goal_cnt][1] + " and delivering it to " + self.final_goals[self.goal_cnt][0])
                self.method = 'goToPerson'
            else:
                self.soundhandle.say("Here you go " + self.final_goals[self.goal_cnt][0] + " here is your " + self.final_goals[self.goal_cnt][1])
                self.soundhandle.say("Will you pay with card or cash?")
                rospy.sleep(10)
                if self.message == "cash":
                    self.set_arm([1, -0.8, 1.9, 0.1], 3)
                elif self.message == "card":
                    self.set_arm([-1, -0.8, 1.9, 0.1], 3)
                else:
                    self.set_arm([0, -0.8, 1.9, 0.1], 3)
                self.soundhandle.say("Thank you")
                self.method = 'goToRestaurant'
            

            self.set_arm([0,0,0,0], 3)
            rospy.loginfo("Goal pose "+str(self.goal_cnt)+" reached") 

            if self.goal_cnt < len(self.final_goals):
                next_goal = MoveBaseGoal()
                next_goal.target_pose.header.frame_id = "map"
                next_goal.target_pose.header.stamp = rospy.Time.now()
                if self.goal_cnt % 2 == 0:
                    next_goal.target_pose.pose.orientation.z = robot_pose.pose.pose.orientation.z
                    next_goal.target_pose.pose.orientation.w = robot_pose.pose.pose.orientation.w
                    next_goal.target_pose.pose.position.x = self.final_goals[self.goal_cnt][2][0]
                    next_goal.target_pose.pose.position.y = self.final_goals[self.goal_cnt][2][1]
                else:
                    next_goal.target_pose.pose.orientation.z = self.final_goals[self.goal_cnt][2].orientation.z
                    next_goal.target_pose.pose.orientation.w = self.final_goals[self.goal_cnt][2].orientation.w
                    next_goal.target_pose.pose.position.x = self.final_goals[self.goal_cnt][2].position.x
                    next_goal.target_pose.pose.position.y = self.final_goals[self.goal_cnt][2].position.y

                rospy.loginfo("Sending goal pose "+str(self.goal_cnt+1)+" to Action Server")
                rospy.loginfo(str(self.final_goals[self.goal_cnt]))
                self.ac.send_goal(next_goal, self.done_callback) 
            else:
                self.soundhandle.say("Done")
                rospy.loginfo("Final goal pose reached!")
                rospy.signal_shutdown("Final goal pose reached!")
                return

        
            
    def deliver_food(self):
        self.delivering = True
        order = self.order.split(',')
        for i in range(len(order)):
            actual = order[i].strip().split(' ')
            name = actual[0]
            food = actual[1]
            self.final_goals.append([name, food, self.food_markers[food]])
            self.final_goals.append([name, food, self.face_markers[name]])

            # if name in self.face_poses.keys():
            #     self.final_goals.append([name, food, self.face_poses[name]])
            # else:
            #     self.final_goals.append([name, food, self.face_markers[name]])


        map_goal = MoveBaseGoal()
        map_goal.target_pose.header.frame_id = "map"
        map_goal.target_pose.pose.orientation.z = robot_pose.pose.pose.orientation.z
        map_goal.target_pose.pose.orientation.w = robot_pose.pose.pose.orientation.w
        map_goal.target_pose.pose.position.x = self.final_goals[self.goal_cnt][2][0]
        map_goal.target_pose.pose.position.y = self.final_goals[self.goal_cnt][2][1]
        map_goal.target_pose.header.stamp = rospy.Time()

        self.ac.send_goal(map_goal, done_cb=self.done_callback)

        pass


    def set_arm(self, angles, duration=1):
        trajectory = JointTrajectory()
        trajectory.joint_names = ["arm_shoulder_pan_joint",
                                "arm_shoulder_lift_joint", "arm_elbow_flex_joint", "arm_wrist_flex_joint"]
        trajectory.points = [JointTrajectoryPoint(positions=angles, time_from_start=rospy.Duration(duration))]
        self.arm_pub.publish(trajectory)
        

    def get_food(self):
        try:
            image_message = rospy.wait_for_message("/arm_camera/rgb/image_raw", Image)
        except Exception as e:
            print(e)
            return
        try:
            image = self.bridge.imgmsg_to_cv2(image_message, "bgr8")
        except CvBridgeError as e:
            print(e)



        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(img_hsv, (0, 0, 40), (0, 100, 150))
        #mask_down = ~cv2.inRange(cv_image_half_down_hsv, (0, 0, 150), (0, 100, 255))
        output = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
        output[np.where((output == [0,0,0]).all(axis = -1))] = [255, 255, 255]

        # Tranform image to grayscale
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        # Do histogram equlization
        img = cv2.equalizeHist(gray)
        # Binarize the image, there are different ways to do it
        #ret, thresh = cv2.threshold(img, 50, 255, 0)
        #ret, thresh = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 25)

        # Extract contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # Example how to draw the contours, only for visualization purposes
        cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
        # cv2.imshow("Contours", img)
        # cv2.waitKey(1)
        # Fit elipses to all extracted contours
        elps = []
        for cnt in contours:
            if cnt.shape[0] >= 10:
                ellipse = cv2.fitEllipse(cnt)
                elps.append(ellipse)

        # print(elps)
        # Find two elipses with same centers
        candidates = []
        for n in range(len(elps)):
            for m in range(n + 1, len(elps)):
                e1 = elps[n]
                e2 = elps[m]
                # print(e1)
                # print(e2)
                dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))
                center = (int(e1[0][1]), int(e1[0][0]))
                size1 = (e1[1][0] + e1[1][1])/2
                el_var1 = abs(e1[1][0] - e1[1][1])
                size2 = (e2[1][0] + e2[1][1])/2
                el_var2 = abs(e2[1][0] - e2[1][1])
                size_diff = abs(size1 - size2)
                # if dist < 50 and (e1[1][0] + e1[1][1])/2 > 200 and (e1[1][0] + e1[1][1])/2 < 600 and abs(e1[1][0] - e1[1][1]) < 100:
                if dist < 50 and size1 > 80 and size1 < 400 and size_diff < 150 and el_var1 < 150 and el_var2 < 150:
                    candidates.append((e1,e2, size1, size_diff, dist))


        for c in candidates:
            e1 = c[0]
            e2 = c[1]
            # print(c[2], c[3], c[4])
            # drawing the ellipses on the image
            cv2.ellipse(image, e1, (0, 255, 0), 2)
            cv2.ellipse(image, e2, (0, 255, 0), 2)

            size = (e1[1][0] + e1[1][1])/2

            # print("SIZE" + str(size))
            center = (e1[0][1], e1[0][0])

            

            x1 = int(center[0] - size / 2)
            x2 = int(center[0] + size / 2)
            x_min = x1 if x1>0 else 0
            x_max = x2 if x2<image.shape[0] else image.shape[0]

            
            y1 = int(center[1] - size / 2)
            y2 = int(center[1] + size / 2)
            y_min = y1 if y1 > 0 else 0
            y_max = y2 if y2 < image.shape[1] else image.shape[1]


            image_cp = image[x_min:x_max,y_min:y_max]
            # print("dimensions", np.shape(image_cp))
            # cv2.imshow("imag",image_cp)
            # cv2.waitKey(1)
            

            image_cp = cv2.normalize(
            image_cp, image_cp, 0, 255, cv2.NORM_MINMAX)

            path = '/home/miha/ROS/src/homework4/scripts/recognize_image/image.jpg'
            cv2.imwrite(path, image_cp)

            food = recognize(path)
            return food
        return None


    def food_recognition_callback(self,status,result):
        rospy.loginfo(f"Callback called with status {status_to_text[status]}")
        self.cylinder_goal_cnt += 1
        if status == 3:
            self.set_arm([0,0.3,1,-0.5], 1)
            angle = 0
            rate = rospy.Rate(3)
            while not rospy.is_shutdown():
                food = self.get_food()
                twist = Twist()
                twist.linear.x = 0
                twist.linear.y = 0
                twist.linear.z = 0
                twist.angular.x = 0
                twist.angular.y = 0
                twist.angular.z = 30 * 3.14 / 180
                angle += 30
                self.cmd_vel_pub.publish(twist)
                if food != None or angle > 1200:
                    break
                rate.sleep()
            self.soundhandle.say("Hello ")
            self.soundhandle.say(self.lol[self.ix])
            self.food_markers[self.lol[self.ix]] = self.cylinder_goals[self.ix]
            self.ix += 1
            self.set_arm([0,0,0,0], 1)
            print(self.cylinder_goal_cnt)
            if self.cylinder_goal_cnt <= 3:
                map_goal = MoveBaseGoal()
                map_goal.target_pose.header.frame_id = "map"
                map_goal.target_pose.pose.orientation.z = robot_pose.pose.pose.orientation.z
                map_goal.target_pose.pose.orientation.w = robot_pose.pose.pose.orientation.w
            
                map_goal.target_pose.pose.position.x = self.cylinder_goals[self.cylinder_goal_cnt][0]
                map_goal.target_pose.pose.position.y = self.cylinder_goals[self.cylinder_goal_cnt][1]
                map_goal.target_pose.header.stamp = rospy.Time()
                self.ac.send_goal(map_goal, done_cb=self.food_recognition_callback)
            else:
                self.initiate_parking()

        pass

    def food_valj(self, status, result):
        if status == 3:
            self.set_arm([0,0.3,1,-0.5], 1)
            angle = 0
            rate = rospy.Rate(3)
            while not rospy.is_shutdown():
                food = self.get_food()
                twist = Twist()
                twist.linear.x = 0
                twist.linear.y = 0
                twist.linear.z = 0
                twist.angular.x = 0
                twist.angular.y = 0
                twist.angular.z = 30 * 3.14 / 180
                angle += 30
                self.cmd_vel_pub.publish(twist)
                if food != None or angle > 1200:
                    break
                rate.sleep()
            self.soundhandle.say("Hello ")
            self.soundhandle.say(food)
            self.set_arm([0,0,0,0], 3)
            self.food_markers[food] = [robot_pose.pose.pose.position.x, robot_pose.pose.pose.position.y]
            print(self.food_markers)

    def food_recognition_phase(self):
        # Move towards detected cylinder
        map_goal = MoveBaseGoal()
        map_goal.target_pose.header.frame_id = "map"
        map_goal.target_pose.pose.orientation.z = robot_pose.pose.pose.orientation.z
        map_goal.target_pose.pose.orientation.w = robot_pose.pose.pose.orientation.w
    
        map_goal.target_pose.pose.position.x = self.cylinder_goals[self.cylinder_goal_cnt][0]
        map_goal.target_pose.pose.position.y = self.cylinder_goals[self.cylinder_goal_cnt][1]
        map_goal.target_pose.header.stamp = rospy.Time()
        print("ja")
        self.ac.send_goal(map_goal, done_cb=self.food_recognition_callback)

        pass


    def process_scan(self, scan, arm_image):
        self.N = len(scan.ranges)
        self.resolution = scan.range_max / self.N / 4

        image = np.zeros((self.N, self.N), np.uint8)

        angle = scan.angle_min
        for r in scan.ranges:
            angle += scan.angle_increment

            if np.isnan(r):
                continue

            x = r * np.cos(angle)
            y = r * np.sin(angle)

            x_p, y_p = self.to_pixel_space(x, y)

            if x_p < 0 or x_p >= self.N or y_p < 0 or y_p >= self.N:
                continue

            image[y_p, x_p] = 255

        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 2, 100, param1=10,
                                   param2=25, minRadius=30, maxRadius=35)
 

        # debug_image = image.copy()
        # debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
        # if circles is not None:
        #     circles = np.uint16(np.around(circles))
        #     for i in circles[0, :]:
        #         cv2.circle(debug_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #         cv2.circle(debug_image, (i[0], i[1]), 2, (0, 0, 255), 3)

        # cv2.imshow("image", debug_image)
        # cv2.waitKey(1)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                x, y = self.to_world_space(i[0], i[1])

                point = PointStamped(header=Header(stamp=scan.header.stamp, frame_id='arm_camera_rgb_optical_frame'),
                                     point=Point(x=-y, y=0, z=x))

                try:
                    world_point = self.tf_buf.transform(point, 'map')
                except:
                    world_point = None

                # To bo treba spremenit
                if world_point:
                    theta = np.arctan2(y, x)

                    image = self.bridge.imgmsg_to_cv2(arm_image, "rgb8")

                    cylinder_x = int(
                        np.interp(-theta, [scan.angle_min, scan.angle_max], [0, image.shape[1]]))

                    width = 50
                    x1, x2 = cylinder_x - width / 2, cylinder_x + width / 2

                    height = 100
                    offset = 150
                    y1, y2 = image.shape[0] - \
                        offset, image.shape[0] - offset + height

                    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)


                    marker_color = ColorRGBA(r=1, g=0, b=0, a=1)
                    # self.add_marker(world_point.point.x, world_point.point.y, ColorRGBA(r=1, g=0, b=0, a=1))
                    

                    if self.cluster.add(world_point.point):
                        self.add_marker(world_point.point.x,
                                        world_point.point.y, marker_color)
                        self.found_cylinders += 1
                    

                        # Move towards detected cylinder
                        map_goal = MoveBaseGoal()
                        map_goal.target_pose.header.frame_id = "map"
                        map_goal.target_pose.pose.orientation.z = robot_pose.pose.pose.orientation.z
                        map_goal.target_pose.pose.orientation.w = robot_pose.pose.pose.orientation.w
                        # map_goal.target_pose.pose.position.x = np.mean(
                        #     [robot_pose.pose.pose.position.x, pose.position.x])
                        # map_goal.target_pose.pose.position.y = np.mean(
                        #     [robot_pose.pose.pose.position.y, pose.position.y])
                        x1 = world_point.point.x
                        y1 = world_point.point.y
                        x0 = robot_pose.pose.pose.position.x
                        y0 = robot_pose.pose.pose.position.y
                        # dn = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
                        t = 2/3
                        map_goal.target_pose.pose.position.x = (
                            1 - t)*x0 + t*x1
                        map_goal.target_pose.pose.position.y = (
                            1 - t)*y0 + t*y1
                        map_goal.target_pose.header.stamp = rospy.Time()

                        # self.initiate_parking()

                        # self.ac.send_goal(map_goal, done_cb=self.food_valj)

                        # if self.found_cylinders == 4 and self.faces_found == 5:

                            # self.initiate_parking()
                            # self.deliver_food()
                            

                        # self.ac.cancel_all_goals()
                        # self.ac.send_goal(map_goal)
                        # self.ac.wait_for_result(rospy.Duration(6))

                                                 
                        # food = food_detection()



    def to_pixel_space(self, x, y):
        x_p = int(x / self.resolution)
        y_p = int(y / self.resolution) + self.N // 2

        return x_p, y_p

    def to_world_space(self, x_p, y_p):
        x = x_p * self.resolution
        y = (y_p - self.N/2) * self.resolution

        return x, y

    def publish_markers(self):
        self.marker_pub.publish(self.markers)

    def add_marker(self, x, y, color):
        pose = Pose(position=Point(x, y, 0),
                    orientation=Quaternion(0, 0, 0, 1))

        marker = Marker(header=Header(frame_id="map", stamp=rospy.Time.now()),
                        pose=pose,
                        type=Marker.CYLINDER,
                        action=Marker.ADD,
                        id=len(self.markers),
                        lifetime=rospy.Time(0),
                        color=color,
                        scale=Vector3(0.2, 0.2, 0.2))

        self.markers.append(marker)


if __name__ == "__main__":
    # final = []
    # final.append(['name', 'food', [1.5,0]])
    # a = final[2][0][0]
    CylinderDetection()
