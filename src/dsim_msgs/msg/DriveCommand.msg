# One edit to the drive queue: the waypoints a person clicks in the viewer.
#
# The queue is POSITIONS AND HEADINGS ONLY. The drive node turns it into a
# trajectory: from wherever the reference is, straight to each waypoint in
# turn, stopping there and turning to that waypoint's heading before the next
# leg. Between waypoints the vehicle keeps the heading of the one it left.
uint8 OP_REPLACE = 0    # the queue becomes [this waypoint]
uint8 OP_APPEND = 1     # this waypoint goes on the end of the queue
uint8 OP_CLEAR = 2      # empty the queue and stop where the reference is

uint8 op
geometry_msgs/Point position    # world frame, m
float64 yaw                     # heading to face on arrival, rad, world frame
float64 v_max                   # speed cap for every leg from now on, m/s
