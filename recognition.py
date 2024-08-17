# подключаем библиотеки для работы с rospy
import rospy
from sensor_msgs.msg import Image
import tf2_ros
import tf2_geometry_msgs
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3Stamped, Vector3

# подключаем дополнительные библиотеки
from utils import *
from recognition import *
import numpy as np
import cv2


class BuildingSearch():
    # пороги для детектирования зданий в формате HSV
    thresholds = {
        'red': (np.array([0, 60, 60]), np.array([6, 255, 255])),
        'blue': (np.array([100, 80, 46]), np.array([124, 255, 255])),
        'green': (np.array([35, 43, 35]), np.array([90, 255, 255])),
        'yellow': (np.array([21, 88, 149]), np.array([35, 215, 255]))
    }

    # предполагаемая этажность в зависимости от цвета
    floors = {
        'red': 4,
        'blue': 3,
        'green': 2,
        'yellow': 1
    }

    # Параметры определения пожаров
    fraction = 0.005
    fire_radius = 1

    def __init__(self, cm: Optional[np.ndarray] = None, dc: Optional[np.ndarray] = None, tf_buffer = None, cv_bridge = None):
        # Параметры камеры для функции undistort
        self.cm = cm
        self.dc = dc

        # TF буффер и cv_bridge для сообщений типа Image
        self.tf_buffer = tf_buffer
        self.cv_bridge = cv_bridge

        # Массив для хранения координат найденных зданий
        self.buildings = []
        self.not_detected_colors = ['green', 'red', 'blue', 'yellow']

        # Топики для rviz и отладки программы
        self.debug_pub = rospy.Publisher("/a/buildings_debug", Image)
        self.mask_overlay_pub = rospy.Publisher("/a/buildings_mask_overlay_pub", Image)
        self.buildings_pub = rospy.Publisher("/a/buildings_pub", MarkerArray)
    
    # метод определений этажности здания
    def get_floors_info(self, floors):
        height = get_building_height(self.flight_height)
        if height >= 0.95:
            count = 4
        elif 0.7 <= height < 0.95:
            count = 3
        elif 0.45 <= height < 0.7:
            count = 2
        else:
            count = 1
        if floors == count:
            return count, True
        return count, False
    
    # создание маски по заданным пороговым значениям для определения здания
    def create_mask(self, frame, lower, upper):
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        mask = cv2.inRange(frame, lower, upper)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        return mask
    
    # создание общей маски для всех оьбъектов
    def create_mask_overlay(self, frame):
        masks = [self.create_mask(frame, lower, upper) for (lower, upper) in self.thresholds.values()]
        mask = np.zeros(frame.shape[:2], dtype="uint8")

        for mask_ in masks:
            mask = cv2.bitwise_or(mask_, mask)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        return mask
       
    # публикация маркеров в rviz
    def publish_markers(self):
        result = []
        id = 0
        for fs in self.buildings:
            # На основе множества распознаваний одного пострадавшего формируем усредненные координаты
            m = np.mean(fs, axis=0)

            marker = Marker()
            marker.header.frame_id = "aruco_map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "color_markers"
            marker.id = id
            marker.type =  Marker.CUBE
            marker.action = Marker.ADD

            # Позиция и ориентация
            marker.pose.position.x = m[0]
            marker.pose.position.y = m[1]
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
        self.buildings_pub.publish(MarkerArray(markers=result))
        return None
    

    # метод нахождения зданий
    def on_frame(self, frame, floor_mask):
         # переводим кадр в hsv формат
        hsv = to_hsv(frame)

        debug = frame.copy()
        mask_overlay = self.create_mask_overlay(hsv)

        conts = cv2.findContours(floor_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE)[-2]

        cont = sorted(conts , key=cv2.contourArea)[-1]

        mannualy_contour = []

        convex_floor = cv2.convexHull(cont, returnPoints=False)
        defects = cv2.convexityDefects(cont, convex_floor)

        if defects is not None:
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cont[s][0])
                end = tuple(cont[e][0])
                far = tuple(cont[f][0])

                dst = self.distance(start, end)

                mannualy_contour.append(start)
                if dst >= 40:
                    mannualy_contour.append(far)
                mannualy_contour.append(end)

        mannualy_contour = np.array(mannualy_contour).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(debug, [mannualy_contour], 0, (0,255,0), 3)

        floor_mask = np.zeros(floor_mask.shape, dtype="uint8")
        floor_mask = cv2.fillPoly(floor_mask, pts=[cont], color=(255,255,255))

        mask = cv2.bitwise_and(mask_overlay, mask_overlay, mask=floor_mask)
    
        contours = cv2.findContours(mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE)[-2]

        frame_vol = np.prod(frame.shape[0:2])

        # Фильтруем объекты по площади
        assert frame_vol != 0
        contours = list(filter(
                lambda c: (cv2.contourArea(c) / frame_vol) >= self.fraction and (cv2.contourArea(c) / frame_vol) < 0.2, 
                contours))

        # Находим центры объектов в кадре
        pnt_img = []
        for cnt in contours:
            M = cv2.moments(cnt)

            if M["m00"] == 0:
                continue

            pnt_img.append(
                    [int(M["m10"] / (M["m00"])),
                    int(M["m01"] / (M["m00"]))])

            cv2.circle(debug, tuple(pnt_img[-1]), 6, (255, 0, 0), 2)

            color = ((0,255,0))
            cv2.drawContours(debug, [cnt], 0, color, 3) 

        # Находим координаты объекта, относительно aruco_map
        if len(pnt_img) > 0:
            pnt_img = np.array(pnt_img).astype(np.float64)
            pnt_img_undist = cv2.undistortPoints(pnt_img.reshape(-1, 1, 2), self.cm, self.dc, None, None).reshape(-1, 2).T
            ray_v = np.ones((3, pnt_img_undist.shape[1]))
            ray_v[:2, :] = pnt_img_undist
            ray_v /= np.linalg.norm(ray_v, axis=0)

            if self.tf_buffer is not None:
                try:
                    transform = self.tf_buffer.lookup_transform("aruco_map", "main_camera_optical", rospy.Time())
                except tf2_ros.ConnectivityException:
                    print("LookupException")
                    return None
                except tf2_ros.LookupException:
                    print("LookupException")
                    return None

                t_wb = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])

                ray_v = np.array([unpack_vec(tf2_geometry_msgs.do_transform_vector3(Vector3Stamped(vector=Vector3(v[0], v[1], v[2])), transform)) for v in ray_v.T])
                ray_o = t_wb

                pnts = [intersect_ray_plane(v, ray_o) for v in ray_v]
                # [self.insert_building(p[:2], idx) for p in pnts if p is not None]
           
        
        # Публикуем маркеры rviz и изображения для отладки
        self.publish_markers()
        self.debug_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug, "bgr8"))
        self.mask_overlay_pub.publish(self.cv_bridge.cv2_to_imgmsg(floor_mask, "mono8"))

    def find_closest(self, point, tuple_obj):
        distances = []
        for fire in tuple_obj:
            distances.append((fire[0][0] - point[0]) ** 2 + (fire[0][1] - point[1]) ** 2)
        
        min_dist = min(distances)
        return distances.index(min_dist), min_dist

    def insert_building(self, point, idx):
        obj = self.buildings
        
        if len(obj) == 0:
            obj.append([point])
            return

        idx, distance = self.find_closest(point, obj)
        if distance <= self.fire_radius:
            obj[idx].append(point)
            return
        obj.append([point])