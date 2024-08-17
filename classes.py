# подключаем библиотеки для работы с rospy
import rospy
from clover import srv
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image, Range
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray

# подключаем дополнительные библиотеки
from utils import *
from recognition import *
import math
import numpy as np
from datetime import datetime
import cv2

# функции получения информации с сервисов
get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
navigate_global = rospy.ServiceProxy('navigate_global', srv.NavigateGlobal)
set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
set_velocity = rospy.ServiceProxy('set_velocity', srv.SetVelocity)
set_attitude = rospy.ServiceProxy('set_attitude', srv.SetAttitude)
set_rates = rospy.ServiceProxy('set_rates', srv.SetRates)
land = rospy.ServiceProxy('land', Trigger)


class Copter():
    def __init__(self, stream) -> None:
        self.stream = stream

    # установка значений для полета
    def set_params(self, speed, flight_height, frame_id='aruco_map'):
        self.flight_height = flight_height
        self.stream.set_flight_height(flight_height)
        self.speed = speed
        self.frame_id = frame_id

    # функция генерации маршрута
    def set_route(self, func):
        self.route = func()

    # запись координат начальной позиции
    def set_start_point(self):
        telem = get_telemetry(frame_id='aruco_map')
        self.start_point = (telem.x, telem.y)
        print(f'start point set in x={telem.x}, y={telem.y}')

    # функция взлета
    def takeoff(self):
        print('takeoff')
        self.navigate_wait(z=1.75, frame_id='body', auto_arm=True)
        print('waiting for next action')
    
    # функция посадки
    def land_wait(self):
        land()
        while get_telemetry().armed:
            rospy.sleep(0.2)

    # функция для навигации по полю
    def navigate_wait(self, x=0, y=0, z=None, yaw=float('nan'),
                        speed=None, frame_id=None,
                        auto_arm=False, tolerance=0.2):
        if speed == None:
            speed = self.speed
        if frame_id == None:
            frame_id = self.frame_id
        if z == None:
            z = self.flight_height

        print(f'flying to x={x}, y={y}, z={z}')
        navigate(x=x, y=y, z=z, yaw=yaw, speed=speed,
                    frame_id=frame_id, auto_arm=auto_arm)

        while not rospy.is_shutdown():
            telem = get_telemetry(frame_id='navigate_target')
            if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
                break
            rospy.sleep(0.2)

    # полет к зданию особого итереса
    def navigate_target_building(self):
        print('navigating to special building')
        # берем координаты с сервера
        coords = get_server_coords(self.stream.buildings)
        if coords != None:
            self.navigate_wait(x=coords[0], y=coords[1], z=self.flight_height)
            rospy.sleep(0.3)
            # высота здания
            height = get_building_height(self.flight_height)
            print(f'building height: {height}')
            print(f'flying to x={coords[0]}, y={coords[1]}, z={height+0.5}')
            self.navigate_wait(x=coords[0], y=coords[1], z=height+0.5)
            # сохраняем фото
            rospy.sleep(0.3)
            self.stream.save_photo()
            rospy.sleep(0.3)
            print(f'flying to x={coords[0]}, y={coords[1]}, z={self.flight_height}')
            self.navigate_wait(x=coords[0], y=coords[1], z=self.flight_height)
            rospy.sleep(0.3)
            return 0
        print('Error while connecting to server')
        return 1    


class Stream():
    def __init__(self) -> None:
        self.img_sub = rospy.Subscriber('main_camera/image_raw', Image, self.callback)
        self.bridge = CvBridge()

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.cm, self.dc = camera_cfg_cvt(
            rospy.wait_for_message("/main_camera/camera_info", CameraInfo))
        self.search = BuildingSearch(cm=self.cm, dc=self.dc, tf_buffer=self.tf_buffer, cv_bridge=self.bridge)

    # функция для сохранения фотографии
    def save_photo(self):
        img = self.bridge.imgmsg_to_cv2(rospy.wait_for_message(self.main_topic, Image), 'bgr8')
        now = datetime.now()
        name = f'{now.strftime("%H-%M-%S")}.jpg'
        cv2.imwrite(name, img)
        print(f'Saved photo {name}')

    # callback для видеопотока
    def callback(self, data):
        try:
            img = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Создаем маску для пола площадки
        floor_mask = self.floor_mask(hsv)
        self.floor_mask_pub.publish(self.bridge.cv2_to_imgmsg(floor_mask, "mono8"))

        # Обрабатываем новый кадр из топика
        self.search.on_frame(img, mask_floor=floor_mask, hsv=hsv)

    # установка высоты полета для распознавания зданий
    def set_flight_height(self, flight_height):
        self.search.flight_height = flight_height

    # Метод, создающий маску для пола
    def floor_mask(self, hsv):
        hsv = cv2.blur(hsv, (10, 10))
        mask = cv2.inRange(hsv, self.floor_thr[0], self.floor_thr[1])

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        mask = np.zeros(mask.shape, dtype="uint8")
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True)
            
            area = cv2.contourArea(approx)
            if area < 600:
                continue
        
            mask = cv2.fillPoly(mask, pts = [approx], color=(255,255,255))
        
        return mask
