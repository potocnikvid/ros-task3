### Homework 4 - Usage instructions

To run amcl_simulation, gmapping_simulation, rins_world and face_localizer_dnn at the same time:

`roslaunch homework4 combined.launch`

To run map_goals:

`rosrun homework4 homework4_map_goals` 

To enable audio signals (allow robot to greet found faces):

`rosrun sound_play soundplay_node.py`