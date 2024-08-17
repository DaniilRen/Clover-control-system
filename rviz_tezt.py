import rospy
from visualization_msgs.msg import Marker, MarkerArray

once = True

def callback(msg):
    global once
    if once:
        once = False
        print(msg)

# публикация маркеров в rviz
def publish_markers(arr):
    pub = rospy.Publisher('/rviz_test', MarkerArray, queue_size=1)
    # sub = rospy.Subscriber('/rviz_test', MarkerArray, callback)
    result = []
    id = 0
    for point in arr:
        # На основе множества распознаваний одного пострадавшего формируем усредненные координаты

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "color_markers"
        marker.id = id
        marker.type =  Marker.CUBE
        marker.action = Marker.ADD

        # Позиция и ориентация
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Масштаб
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.5

        # Цвет
        marker.color.a = 0.8

        marker.color.r = 1
        marker.color.g = 0.1
        marker.color.b = 0

        result.append(marker)
        id += 1

    # Публикуем маркеры
    pub.publish(MarkerArray(markers=result))
    return None

rospy.init_node('rviz_test')
points = [(2, 2), (4, 4)]
delay = 10000000
i = 0
while True:
    if i % delay == 0:
        points = list(map(lambda x: (x[0]+0.1, x[1]+0.1), points))
        print(points)
        publish_markers(points)
    i += 1