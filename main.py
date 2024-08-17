from classes import Copter, Stream
import rospy

def route_gen():
    route = []
    for i in range(0, 7, 1):
        for j in range(0, 7, 1):
            if i % 2 == 0:
                route.append([7 - (j / 2), 1 + (i / 2)])
            else:
                route.append([4 + (j / 2), 1 + (i / 2)])
    return route

def main():
    MIDDLE_POINT = (5.5, 2.5)

    stream = Stream()
    copter = Copter(stream)
    copter.set_params(speed=0.3, flight_height=1.75)
    copter.set_route(func=route_gen)

    copter.takeoff()
    copter.set_start_point()
    for point in copter.route:
        copter.navigate_wait(x=point[0], y=point[1])
        if point == MIDDLE_POINT:
            copter.stream.save_photo()
    copter.navigate_target_building()
    copter.navigate_wait(x=copter.start_point[0], y=copter.start_point[1])
    copter.land_wait()


if __name__ == "__main__":
    rospy.init_node('flight')
    main()