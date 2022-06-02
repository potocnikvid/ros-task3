#!/usr/bin/python3

from doctest import OutputChecker
import sys

from sympy import false
import rospy
import cv2
from tf.transformations import quaternion_from_euler
import numpy as np
import tf2_geometry_msgs
import tf2_ros
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from sound_play.libsoundplay import SoundClient
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from geometry_msgs.msg import PointStamped, Point, Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

goal = [-1.3, 0.8]

class The_Ring:
    def __init__(self):
        rospy.init_node('image_converter', anonymous=True)

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # A help variable for holding the dimensions of the image
        self.dims = (0, 0, 0)

        # Marker array object used for visualizations
        self.marker_array = MarkerArray()
        self.marker_num = 1

        # Subscribe to the image and/or depth topic
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)

        self.depth_sub = rospy.Subscriber("/camera/depth_registered/image_raw", Image, self.depth_callback)

        # Publiser for the visualization markers
        self.markers_pub = rospy.Publisher('/ring_markers', MarkerArray, queue_size=1000)

        # Object we use for transforming between coordinate frames
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        self.colors = ["green", "red", "black", "blue"]
        # self.colors = ["green"]

        self.color = " "

        self.final_goal = [None, None]
        self.soundclient = SoundClient()

        self.cmd_vel_pub = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=1000)

        

    def calc_dist(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def marker_exists(self, x, z):
        if len(self.marker_array.markers) == 0:
            return False
        for i in range(len(self.marker_array.markers)):
            if self.calc_dist(x, z, self.marker_array.markers[i].pose.position.x, self.marker_array.markers[i].pose.position.z) < 0.3:
                return True
        return False

    def get_pose(self,e,dist):
        # Calculate the position of the detected ellipse

        k_f = 525 # kinect focal length in pixels

        elipse_x = self.dims[1] / 2 - e[0][0]
        elipse_y = self.dims[0] / 2 - e[0][1]

        angle_to_target = np.arctan2(elipse_x,k_f)
        # print(angle_to_target)
        # print(np.cos(angle_to_target))
        # print(np.sin(angle_to_target))
        # Get the angles in the base_link relative coordinate system
        x, y = dist*np.cos(angle_to_target), dist*np.sin(angle_to_target)
        
        # print(x, y)
        ### Define a stamped message for transformation - directly in "base_frame"
        # point_s = PointStamped()
        # point_s.point.x = x
        # point_s.point.y = y
        # point_s.point.z = 0.3
        # point_s.header.frame_id = "base_link"
        # point_s.header.stamp = rospy.Time(0)

        # Define a stamped message for transformation - in the "camera rgb frame"
        point_s = PointStamped()
        point_s.point.x = -y
        point_s.point.y = 0
        point_s.point.z = x
        point_s.header.frame_id = "camera_rgb_optical_frame"
        point_s.header.stamp = rospy.Time(0)

        # Get the point in the "map" coordinate system
        try:
            point_world = self.tf_buf.transform(point_s, "map")
        except Exception as e:
            print(e)
        # Create a Pose object with the same position
        pose = Pose()
        pose.position.x = point_world.point.x
        pose.position.y = point_world.point.y
        pose.position.z = point_world.point.z

        # Create a marker used for visualization
        self.marker_num += 1
        marker = Marker()
        marker.header.stamp = point_world.header.stamp
        marker.header.frame_id = point_world.header.frame_id
        marker.pose = pose
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.frame_locked = False
        marker.lifetime = rospy.Duration.from_sec(10)
        marker.id = self.marker_num
        marker.scale = Vector3(0.1, 0.1, 0.1)
        marker.color = ColorRGBA(0, 1, 0, 1)
    

        #print(pose.position.x, pose.position.z)
        if not self.marker_exists(pose.position.x, pose.position.z):
            self.marker_array.markers.append(marker)

        # for i in range(len(self.marker_array.markers)):
        #     if self.calc_dist(pose.position.x, pose.position.z, self.marker_array.markers[i].pose.position.x, self.marker_array.markers[i].pose.position.z) < 0.3:
        #         self.marker_array.markers[i].pose.position.x = (pose.position.x + self.marker_array.markers[i].pose.position.x)/2
        #         self.marker_array.markers[i].pose.position.z = (pose.position.z + self.marker_array.markers[i].pose.position.z)/2

        if self.color == "green":
            self.final_goal = [marker.pose.position.x, marker.pose.position.z]
            self.markers_pub.publish(self.marker_array)
            print("green ring: ", self.final_goal)
            self.soundclient.say("green ring")
            rospy.signal_shutdown("detected green")


            # self.park()




    def get_color(self, image, e1):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        center = (int(e1[0][1]), int(e1[0][0]))
        height = int(e1[1][1])
        for i in range(height):
            color = image_hsv[center[0] + i, center[1]]
            if (color != [0, 0, 255]).any():
                hue = color[0]
                sat = color[1]
                # print(sat)
                if hue == 0:
                    return "black"
                elif hue < 30 or hue > 160:
                    return "red"
                elif 40 < hue < 70 and 115 < sat < 120:
                    return "green"
                elif 90 < hue < 120:
                    return "blue"
                # else:
                    # raise Exception("Unknown color")

    def print_color(self, color_hsv):
        hue = color_hsv[0]
        if hue == 0:
            print("black ring")
            self.soundclient.say("black ring")
        elif hue < 30 or hue > 160:
            print("red ring")
            self.soundclient.say("red ring")
        elif 40 < hue < 70:
            print("green ring")
            self.soundclient.say("green ring")
        elif 90 < hue < 120:
            print("blue ring")
            self.soundclient.say("blue ring")
        





    def park(self):        

        arm_pub = rospy.Publisher("/arm_command", String, queue_size=1000)
        arm_pub.publish("extend")
        #self.final_goal = [3, -0.5]
        # Setup
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        client.wait_for_server()
        
        #print(self.final_goal)
        #angle = 2.8
        # Nardis goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = self.final_goal[0]
        goal.target_pose.pose.position.y = self.final_goal[1]
        orientation = Quaternion(*quaternion_from_euler(0, 0, 2.8, axes='sxyz'))
        goal.target_pose.pose.orientation = orientation
        # goal.target_pose.pose.orientation.w = 0.0
        # Posles goal
        client.send_goal(goal)

        # Cakas da pride na goal
        client.wait_for_result()
        
        # r = rospy.Rate(5)
        # while not rospy.is_shutdown():
        # move = Twist()
        # move.linear.x = 1.7
        # move.angular.z = 0
        # self.cmd_vel_pub.publish(move)
        
        # Nardis goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = self.final_goal[0] - 0.79
        goal.target_pose.pose.position.y = self.final_goal[1] + 0.1
        orientation = Quaternion(*quaternion_from_euler(0, 0, 0, axes='sxyz'))
        goal.target_pose.pose.orientation = orientation
        # goal.target_pose.pose.orientation.w = 0.0
        # Posles goal
        client.send_goal(goal)

        # Cakas da pride na goal
        client.wait_for_result()


        rospy.signal_shutdown("Done")


    def image_callback(self,data):
        # print('I got a new image!')

        #print(self.colors)
        self.final_goal = [3, -0.5]
        if len(self.colors) < 1:
            self.park()

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Set the dimensions of the image
        self.dims = cv_image.shape
        # print(self.dims)
        # Delete half of image
        cv_image_half = cv_image[0:self.dims[0]//2, :, :]
        #cv_image_half_down = cv_image[self.dims[0]//2:, :, :]

        cv_image_half_hsv = cv2.cvtColor(cv_image_half, cv2.COLOR_BGR2HSV)
        #cv_image_half_down_hsv = cv2.cvtColor(cv_image_half_down, cv2.COLOR_BGR2HSV)

        mask = ~cv2.inRange(cv_image_half_hsv, (0, 0, 150), (0, 100, 255))
        #mask_down = ~cv2.inRange(cv_image_half_down_hsv, (0, 0, 150), (0, 100, 255))
        output = cv2.bitwise_and(cv_image_half, cv_image_half, mask=mask)
        output[np.where((output == [0,0,0]).all(axis = -1))] = [255, 255, 255]
        #output_down = cv2.bitwise_and(cv_image_half_down, cv_image_half_down, mask=mask_down)
        #output_down[np.where((output_down == [0,0,0]).all(axis = -1))] = [255, 255, 255]
        # cv2.imshow("Image half", output)
        # cv2.waitKey(1)
        #cv2.imshow("Image half down", output_down)
        #cv2.waitKey(1)
        image = output
        #image_valji = output_down

        #KMEANS
        # Z = output.reshape((-1,3))
        # # convert to np.float32
        # Z = np.float32(Z)
        # # define criteria, number of clusters(K) and apply kmeans()
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # K = 2
        # ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # # Now convert back into uint8, and make original image
        # center = np.uint8(center)
        # res = center[label.flatten()]
        # cv_image_half_kmeans = res.reshape((cv_image_half.shape))
        # cv2.imshow("kmeans", cv_image_half_kmeans)
        # cv2.waitKey(1)

        # Tranform image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #gray_valji = cv2.cvtColor(image_valji, cv2.COLOR_BGR2GRAY)

        # Do histogram equlization
        img = cv2.equalizeHist(gray)
        #img_valji = cv2.equalizeHist(gray_valji)

        # Binarize the image, there are different ways to do it
        #ret, thresh = cv2.threshold(img, 50, 255, 0)
        #ret, thresh = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 25)
        #thresh_valji = cv2.adaptiveThreshold(img_valji, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 25)

        # Extract contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        #contours_valji, _ = cv2.findContours(thresh_valji, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # Example how to draw the contours, only for visualization purposes
        cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
        #cv2.drawContours(img_valji, contours_valji, -1, (255, 0, 0), 3)
        # cv2.imshow("Contour window",img)
        # cv2.waitKey(1)

        # Fit elipses to all extracted contours
        elps = []
        for cnt in contours:
            if cnt.shape[0] >= 10:
                ellipse = cv2.fitEllipse(cnt)
                if(image[int(ellipse[0][1]), int(ellipse[0][0])] == [255, 255, 255]).all():
                    elps.append(ellipse)


        try:
            depth_img = rospy.wait_for_message('/camera/depth/image_raw', Image)
        except Exception as e:
            print(e)

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
                depth_image = self.bridge.imgmsg_to_cv2(depth_img, "32FC1")
                if dist < 5 and (e1[1][0] + e1[1][1])/2 > 17 and (e1[1][0] + e1[1][1])/2 < 50:
                    # print(depth_image[center[0], center[1]])
                    candidates.append((e1,e2))


        try:
            depth_img = rospy.wait_for_message('/camera/depth/image_raw', Image)
        except Exception as e:
            print(e)


        # Extract the depth from the depth image
        for c in candidates:
            # the centers of the ellipses
            e1 = c[0]
            e2 = c[1]

            # drawing the ellipses on the image
            cv2.ellipse(cv_image, e1, (0, 255, 0), 2)
            cv2.ellipse(cv_image, e2, (0, 255, 0), 2)

            size = (e1[1][0] + e1[1][1])/2
            # print("SIZE" + str(size))
            center = (e1[0][1], e1[0][0])


            self.color = self.get_color(image, e1)
            # print(self.color)
            if self.color in self.colors:
                self.colors.remove(self.color)
                # print("JUST SAIDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD " + self.color + "ring")
                # self.soundclient.say(self.color)

            x1 = int(center[0] - size / 2)
            x2 = int(center[0] + size / 2)
            x_min = x1 if x1>0 else 0
            x_max = x2 if x2<cv_image.shape[0] else cv_image.shape[0]

            y1 = int(center[1] - size / 2)
            y2 = int(center[1] + size / 2)
            y_min = y1 if y1 > 0 else 0
            y_max = y2 if y2 < cv_image.shape[1] else cv_image.shape[1]
            

            depth_image = self.bridge.imgmsg_to_cv2(depth_img, "32FC1")
            #print(depth_image[x_min:x_max,y_min:y_max])
            self.get_pose(e1, float(np.nanmean(depth_image[x_min:x_max,y_min:y_max])))



        # if len(candidates)>0:
            # cv2.imshow("Image window", cv_image)
            # cv2.waitKey(1)
        

    def depth_callback(self,data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
        except CvBridgeError as e:
            print(e)

        # Do the necessairy conversion so we can visuzalize it in OpenCV
        image_1 = depth_image / 65536.0 * 255
        image_1 =image_1/np.max(image_1)*255

        image_viz = np.array(image_1, dtype= np.uint8)

        # cv2.imshow("Depth window", image_viz)
        # cv2.waitKey(1)


def main():

    ring_finder = The_Ring()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
