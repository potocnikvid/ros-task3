#include "ros/ros.h"

#include <nav_msgs/GetMap.h>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib_msgs/GoalStatusArray.h>
#include <actionlib/client/simple_action_client.h>
#include <tf2/LinearMath/Quaternion.h>
#include <sound_play/sound_play.h>
#include <unistd.h>

using namespace std;

#define STATUS_REPORT_INTERVAL 3.0
#define GLOAL_TIMEOUT 30.0

float map_resolution = 0;
bool moving = false;
geometry_msgs::TransformStamped map_transform;

// http://wiki.ros.org/navigation/Tutorials/SendingSimpleGoals
typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

void sleepok(int t, ros::NodeHandle &nh)
{
  if (nh.ok())
    sleep(t);
}

// Message location to the map
void mapCallback(const nav_msgs::OccupancyGridConstPtr &msg_map)
{
  int size_x = msg_map->info.width;
  int size_y = msg_map->info.height;

  if ((size_x < 3) || (size_y < 3))
  {
    ROS_INFO("Map size is only x: %d,  y: %d . Not running map to image conversion", size_x, size_y);
    return;
  }

  map_resolution = msg_map->info.resolution;
  map_transform.transform.translation.x = msg_map->info.origin.position.x;
  map_transform.transform.translation.y = msg_map->info.origin.position.y;
  map_transform.transform.translation.z = msg_map->info.origin.position.z;

  map_transform.transform.rotation = msg_map->info.origin.orientation;

  return;
}

void statusCallback(const actionlib_msgs::GoalStatusArray &goal_status)
{
  if (goal_status.header.seq % 10 == 0 && moving)
  {
    ROS_INFO("%s", (char *)&goal_status.status_list[0].text);
  }
}

int main(int argc, char **argv)
{

  ros::init(argc, argv, "homework4_map_goals");
  ros::NodeHandle node_handle;

  ros::Subscriber map_subscriber = node_handle.subscribe("map", 10, &mapCallback);
  ros::Subscriber goal_subscriber = node_handle.subscribe("/move_base/status", 10, &statusCallback);

  MoveBaseClient ac("move_base", true);
  sound_play::SoundClient sound_client;

  while (!ac.waitForServer(ros::Duration(5.0)))
  {
    ROS_INFO("Waiting for the \"move_base\" action server to come up.");
  }

  // 5 goal locations on the map
  // const float map_goals[5][2] = {{2.65, -1.10}, {0.55, -1.25}, {1.05, 1.40}, {-2.90, 0.90}, {0.95, -2.20}};
  // const float map_goals[26][2] = {{-0.60, -1.80},
  //                                 {-0.25, -2.60},
  //                                 {0.85, -2.10},
  //                                 {1.90, -2.15},
  //                                 {2.95, -1.35},
  //                                 {3.00, -1.10},
  //                                 {2.55, -0.90},
  //                                 {1.60, -1.20},
  //                                 {0.45, -1.10},
  //                                 {1.50, 0.15},
  //                                 {1.10, 1.45},
  //                                 {-0.05, 1.40},
  //                                 {0.00, 0.50},
  //                                 {0.95, -0.35},
  //                                 // {1.00, 0.95},
  //                                 {-1.50, 1.05},
  //                                 {-1.95, 0.15},
  //                                 {-2.80, 0.80},
  //                                 {-2.55, -0.20},
  //                                 {-1.55, 0.10},
  //                                 {-1.70, -0.80},
  //                                 {-0.60, -0.70},
  //                                 {-1.90, -1.80},
  //                                 {-2.20, -1.70},
  //                                 {-0.40, -1.55},
  //                                 {1.00, -2.50}};
  const float map_goals[14][2] = {{-0.5, 1.3}, {-0.5, 0.8}, {-1, 2}, {-1.5, 1.5}, {0, 2.5}, {2, 2.5}, {2.5, 1.5}, {1.1, 1.65}, {1.2, 0}, {3.5, -0.65}, {2.5, -1.2}, {1.5, -1.6}, {0.5, -1.3}, {0, -1.5}};
  // const float map_goals[7][4] = {{1.14, -0.57, 0.39, 0.92},
  //                                 {1.56, -0.17, 1, 0.013},
  //                                 {1.14, 0.1, 0.39, 0.92},
  //                                 {1.6, -1.2, -0.65, 0.75},
  //                                 {0.6, -1.3, 0.73, 0.68},
  //                                 {0, -0.2, 1, 0.06},
  //                                 {0.77, -1.12, 0.37, 0.93}};

  ros::Duration(6).sleep();
  // Move robot to every location
  int i = 0;
  while (i < (sizeof(map_goals) / sizeof(*map_goals)))
  {
    move_base_msgs::MoveBaseGoal map_goal;
    map_goal.target_pose.header.frame_id = "map";
    map_goal.target_pose.pose.orientation.w = 1;
    map_goal.target_pose.pose.position.x = map_goals[i][0];
    map_goal.target_pose.pose.position.y = map_goals[i][1];
    map_goal.target_pose.header.stamp = ros::Time::now();
    ROS_INFO("Moving to (x: %f, y: %f).", map_goals[i][0], map_goals[i][1]);

    ac.sendGoal(map_goal);

    moving = true;
    ac.waitForResult();
    moving = false;

    actionlib::SimpleClientGoalState goal_state = ac.getState();

    if (goal_state == actionlib::SimpleClientGoalState::SUCCEEDED)
    {
      ROS_INFO("Robot moved to (x: %f, y: %f).", map_goals[i][0], map_goals[i][1]);
      i++;
    }
    else if (goal_state == actionlib::SimpleClientGoalState::RECALLED)
    {
      ROS_INFO("recalled");
      ros::Duration(3).sleep();
    }
    else if (goal_state == actionlib::SimpleClientGoalState::PREEMPTED)
    {
      ROS_INFO("preempted");
      break;
      // ros::Duration(20).sleep();
      // i++;
    }
    else if (goal_state == actionlib::SimpleClientGoalState::ABORTED)
    {
      ROS_INFO("aborted");
      i++;
      ros::Duration(3).sleep();
    }
    else if (goal_state == actionlib::SimpleClientGoalState::REJECTED)
    {
      ROS_INFO("rejected");
      i++;
      ros::Duration(3).sleep();
    }
    // else if (goal_state == actionlib::SimpleClientGoalState::PREEMPTING)
    // {
    //   ROS_INFO("preempting");
    //   ros::Duration(3).sleep();
    // }
    // else if (goal_state == actionlib::SimpleClientGoalState::RECALLING)
    // {
    //   ROS_INFO("recalling");
    //   ros::Duration(3).sleep();
    // }
    // else if (goal_state == actionlib::SimpleClientGoalState::REJECTING)
    // {
    //   ROS_INFO("rejecting");
    //   ros::Duration(3).sleep();
    // }
    else
    {
      ROS_ERROR("The robot failed to move!");
      i++;
      // return 1; // Continue ...
    }
  }

  sleepok(30, node_handle);

  return 0;
}
