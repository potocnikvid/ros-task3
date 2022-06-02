#!/usr/bin/python3
import roslib
#roslib.load_manifest('exercise4')
import rospy
import sensor_msgs.msg
import message_filters
import tf2_ros # !!!
from std_msgs.msg import String, Bool, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3, Quaternion

# Node for face detection.
class Breadcrumbs():
	def __init__(self):
		self.rate = rospy.get_param('~rate', 1) # default value is 1

		markers_topic = rospy.get_param('~markers_topic', 'crumbs') # topic where the markers are published

		self.tf2_buffer = tf2_ros.Buffer() # tf2 buffer object (transform stream)
		self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)

		self.trail_pub = rospy.Publisher(markers_topic, MarkerArray, queue_size=5) # Publisher for the markers
		self.trail = []

		self.message_counter = 0

	def push_position(self):

		try:
			### Give me where is the 'base_link' frame relative to the 'map' frame at the latest available time
			trans = self.tf2_buffer.lookup_transform('map', 'base_link', rospy.Time(0))

			### Give me where is the 'base_link' frame relative to the 'map' frame (coordinate system) now, wait for up to 2 seconds for the transform to become available
			#trans = self.tf2_buffer.lookup_transform('map', 'base_link', rospy.Time.now(), rospy.Duration(2))

			### Give me where was the 'base_link' frame relative to the 'map' frame 5 seconds ago, wait for up to 2 seconds for the transform to become available
			#moment_in_past = rospy.Time.now() - rospy.Duration(5.)			
			#trans = self.tf2_buffer.lookup_transform('map', 'base_link', moment_in_past, rospy.Duration(1.))

			### Time travelling (sadly, only in the past):
			### Give me the transformation that describes how to
			### get to where 'base_link' was 2 seconds ago,
			### if I am starting from where 'map' was 1 second ago.
			### 'world' is a fixed frame in our transform tree
			### wait with a timeout of 1 second.
			#trans = self.tf2_buffer.lookup_transform_full(target_frame='base_link',
			#					      target_time=rospy.Time.now()-rospy.Duration(2.),
			#					      source_frame='map',
			#					      source_time=rospy.Time.now()-rospy.Duration(1.),
			#					      fixed_frame='world',
			#					      timeout=rospy.Duration(1.))

			self.trail.append(trans)
			
		except Exception as e:
			#(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException)
			print (e)

		markers = MarkerArray()

		i = 0
		for point in self.trail:
				#print point
				marker = Marker()
				marker.header.stamp = rospy.Time.now() # STAMP ... timestamp (when the data was recorded)
				marker.header.frame_id = 'map' # In which coordinate system is this position orientation.
				marker.pose.position = Point(point.transform.translation.x, point.transform.translation.y, point.transform.translation.z) # x, y, z
				marker.pose.orientation = Quaternion(0.5, 0.5, 0.5, 0.5) # Default values for orientation
				marker.type = Marker.CUBE # Marker shape
				marker.action = Marker.ADD # Replace, delete, etc.
				marker.frame_locked = False # If the frame moves, should the marker also move?
				marker.lifetime = rospy.Time(0) # The marker should live forever, rospy.Duration(10) - live for 10 seconds
				marker.id = i
				marker.scale = Vector3(0.1, 0.1, 0.1) # size of marker
				marker.color = ColorRGBA(1, 1, 0, 1)  # colour of marker
				markers.markers.append(marker)
				i = i + 1

		self.trail_pub.publish(markers)

		self.message_counter = self.message_counter + 1



if __name__ == '__main__':

		rospy.init_node('breadcrumbs')

		bc = Breadcrumbs()
		r = rospy.Rate(bc.rate)
		while not rospy.is_shutdown():
			bc.push_position()
			r.sleep()
