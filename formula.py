# formula.py
import numpy as np
from config import EARTH_RADIUS, SAT_HEIGHT, NUM_OF_ORB, NUM_OF_SPO, LIGHT_SPEED


def get_lat_lon(sat):
    """
    위성의 지역 인덱스를 바탕으로 위도와 경도 계산
    """
    lat_divs = np.linspace(-90, 90, NUM_OF_ORB + 1)
    lon_divs = np.linspace(-180, 180, NUM_OF_SPO + 1)
    lat_centers = (lat_divs[:-1] + lat_divs[1:]) / 2
    lon_centers = (lon_divs[:-1] + lon_divs[1:]) / 2
    return lat_centers[sat.region_x], lon_centers[sat.region_y]


def euclidean_distance_with_obstruction(lat1, lon1, lat2, lon2, radius=EARTH_RADIUS, altitude=SAT_HEIGHT):
    """
    두 위성 간의 직선 거리를 계산하며, 지구에 가려진 경우 NaN 반환
    """
    r = radius + altitude  # 위성 고도 반영
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    x1, y1, z1 = r * np.cos(lat1) * np.cos(lon1), r * np.cos(lat1) * np.sin(lon1), r * np.sin(lat1)
    x2, y2, z2 = r * np.cos(lat2) * np.cos(lon2), r * np.cos(lat2) * np.sin(lon2), r * np.sin(lat2)

    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    if distance == 0:
        return 0
    t_numerator = -(x1 * (x2 - x1) + y1 * (y2 - y1) + z1 * (z2 - z1))
    t_denominator = distance ** 2
    if t_denominator == 0:
        return distance
    t = t_numerator / t_denominator
    closest_x = x1 + t * (x2 - x1)
    closest_y = y1 + t * (y2 - y1)
    closest_z = z1 + t * (z2 - z1)
    min_distance_to_earth = np.sqrt(closest_x ** 2 + closest_y ** 2 + closest_z ** 2)

    return distance if min_distance_to_earth > radius else np.nan


def calculate_satellite_distance(sat1, sat2):
    """
    두 위성 간의 거리 계산
    """
    lat1, lon1 = get_lat_lon(sat1)
    lat2, lon2 = get_lat_lon(sat2)
    return euclidean_distance_with_obstruction(lat1, lon1, lat2, lon2)


def calculate_propagation_delay(sat1, sat2):
    """
    두 위성 간 전파 지연 (ms 단위) 계산
    """
    distance = calculate_satellite_distance(sat1, sat2)
    propagation_delay = distance / LIGHT_SPEED
    return propagation_delay
