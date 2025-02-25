from param import *
from packet import *
import numpy as np
import random
from satellite import *

# 구역을 8x16으로 나누었을 때의 위도, 경도 변환 함수
def get_lat_lon(sat):
    lat_divs = np.linspace(-90, 90, NUM_OF_ORB + 1)  # 8개 구간, 9개 경계점
    lon_divs = np.linspace(-180, 180, NUM_OF_SPO + 1)  # 16개 구간, 17개 경계점
    lat_centers = (lat_divs[:-1] + lat_divs[1:]) / 2
    lon_centers = (lon_divs[:-1] + lon_divs[1:]) / 2
    return lat_centers[sat.region_x], lon_centers[sat.region_y]


def euclidean_distance_with_obstruction(lat1, lon1, lat2, lon2, radius=EARTH_RADIUS, altitude=SAT_HEIGHT):
    r = radius + altitude  # 위성 고도 반영한 반지름
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    x1, y1, z1 = r * np.cos(lat1) * np.cos(lon1), r * np.cos(lat1) * np.sin(lon1), r * np.sin(lat1)
    x2, y2, z2 = r * np.cos(lat2) * np.cos(lon2), r * np.cos(lat2) * np.sin(lon2), r * np.sin(lat2)

    # 직선 거리 계산
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    if distance == 0:
        return 0  # 동일한 위치에 있는 경우 0 반환

    # 지구 중심에서 직선까지의 최단 거리 계산
    t_numerator = -(x1 * (x2 - x1) + y1 * (y2 - y1) + z1 * (z2 - z1))
    t_denominator = distance ** 2
    if t_denominator == 0:
        return distance

    t = t_numerator / t_denominator
    closest_x, closest_y, closest_z = x1 + t * (x2 - x1), y1 + t * (y2 - y1), z1 + t * (z2 - z1)
    min_distance_to_earth = np.sqrt(closest_x ** 2 + closest_y ** 2 + closest_z ** 2)

    # 지구에 의해 가려지면 NaN 반환
    return distance if min_distance_to_earth > radius else np.nan


def calculate_satellite_distance(sat1, sat2):
    lat1, lon1 = get_lat_lon(sat1)
    lat2, lon2 = get_lat_lon(sat2)
    return euclidean_distance_with_obstruction(lat1, lon1, lat2, lon2)

def calculate_propagation_delay(sat1, sat2): # [ms]

    distance = calculate_satellite_distance(sat1, sat2)
    propagation_delay = distance/LIGHT_SPEED

    return propagation_delay

# 각 VNF를 32개 위성에 랜덤 배치
def assign_vnfs(rows, cols, vnf_list, vnf_count):
    vnf_positions = {vnf: set() for vnf in vnf_list}

    all_positions = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(all_positions)

    for vnf, count in zip(vnf_list, [vnf_count] * len(vnf_list)):
        vnf_positions[vnf] = set(all_positions[:count])
        all_positions = all_positions[count:]

    matrix = [[[] for _ in range(cols)] for _ in range(rows)]

    for vnf, positions in vnf_positions.items():
        for r, c in positions:
            matrix[r][c].append(vnf)

    return matrix

def create_satellite_grid():
    """
    8x16 위성 그리드를 생성하여 2차원 리스트 형태로 반환합니다.
    각 위성은 FIFO 패킷 대기열을 갖는 Satellite 객체입니다.
    """
    grid = []
    for x in range(NUM_OF_ORB):
        row = []
        for y in range(NUM_OF_SPO):
            row.append(Satellite(x, y))
        grid.append(row)
    return grid


def print_queue_status(grid, time):
    """
    각 시간 슬롯마다 위성 그리드의 패킷 대기열 상태를 출력합니다.
    각 위성에 쌓여있는 패킷 수를 2차원 배열 형식으로 보여줍니다.
    """
    print(f"Time {time}:")
    for row in grid:
        # 각 위성의 대기열 길이를 한 줄에 출력
        print(" ".join(f"{len(sat.queue):2d}" for sat in row))
    print("-" * 40)


# =============================================================================
# 시뮬레이션 함수
# =============================================================================
def simulate(simulation_time, source_density_map, dest_density_map):
    """
    시뮬레이션 함수:
      - 매 타임슬롯마다 출발지 density_map의 각 구역에서 정해진 개수의 패킷 생성.
      - 패킷의 목적지는 dest_density_map의 가중치를 이용해 확률적으로 선택.
      - 생성된 패킷은 해당 위성의 대기열에 추가되고, 각 위성이 process_queue()를 호출하여 전달.
      - 매 타임슬롯마다 위성의 큐 상태와, 해당 타임슬롯에서 전달된 패킷의 세부 정보를 출력.

    :param simulation_time: 시뮬레이션 시간(초)
    :param source_density_map: 출발지용 density_map (패킷 생성률)
    :param dest_density_map: 목적지용 density_map (목적지 선택 시 가중치)
    :return: 최종 전달된 패킷 리스트
    """
    grid = create_satellite_grid()
    delivered_packets = []

    # 목적지 선택을 위한 population과 가중치 목록 준비
    destination_population = [(i, j) for i in range(NUM_OF_ORB) for j in range(NUM_OF_SPO)]
    destination_weights = [dest_density_map[i][j] for i in range(NUM_OF_ORB) for j in range(NUM_OF_SPO)]

    time = 0
    while time < simulation_time:
        # 매 타임슬롯마다 각 구역의 density 값만큼 패킷 생성
        for i in range(NUM_OF_ORB):
            for j in range(NUM_OF_SPO):
                count = source_density_map[i][j]
                for _ in range(count):
                    dest = random.choices(destination_population, weights=destination_weights, k=1)[0]
                    packet = Packet(source=(i, j), destination=dest)
                    grid[i][j].enqueue_packet(packet)

        tick_delivered = []
        # 각 위성이 보유한 패킷 처리 (인접 위성으로 전달 및 최종 전달)
        for row in grid:
            for sat in row:
                delivered = sat.process_queue(grid)
                tick_delivered.extend(delivered)
                delivered_packets.extend(delivered)

        # 타임슬롯 종료 후, 위성 큐 상태와 전달된 패킷들의 세부 정보 출력
        print_queue_status(grid, time)
        if tick_delivered:
            print(f"Tick {time} - Delivered {len(tick_delivered)} packets:")
            for p in tick_delivered:
                print(f"  Packet ID: {p.id}, Hops: {p.hops}, Total Propagation Delay: {p.travel_time:.3f} ms")
        else:
            print(f"Tick {time} - No packets delivered.")
        print("=" * 40)

        time += 1

    print(f"총 전달된 패킷 수: {len(delivered_packets)}")
    return delivered_packets



# def simulate(simulation_time, source_density_map, dest_density_map):
#     """
#     시뮬레이션 함수:
#       - 매 타임슬롯(초)마다 출발지 density_map의 각 구역에서 정해진 개수의 패킷을 생성합니다.
#       - 패킷의 목적지는 dest_density_map을 가중치로 하여 랜덤하게 선택합니다.
#       - 생성된 패킷은 해당 위성의 대기열에 추가되고, 각 위성이 process_queue()를 호출하여 인접 위성으로 전달합니다.
#       - 매 타임슬롯마다 위성의 큐 상태를 출력합니다.
#
#     :param simulation_time: 시뮬레이션 시간(초)
#     :param source_density_map: 출발지용 density_map (각 구역의 패킷 생성률)
#     :param dest_density_map: 목적지용 density_map (목적지 선택 시의 가중치)
#     :return: 최종 전달된 패킷 리스트
#     """
#     grid = create_satellite_grid()
#     delivered_packets = []
#
#     # 목적지 선택을 위한 population과 가중치 목록 준비
#     destination_population = [(i, j) for i in range(NUM_OF_ORB) for j in range(NUM_OF_SPO)]
#     destination_weights = [dest_density_map[i][j] for i in range(NUM_OF_ORB) for j in range(NUM_OF_SPO)]
#
#     time = 0
#     while time < simulation_time:
#         # 매 타임슬롯마다 각 구역의 density 값만큼 패킷 생성
#         for i in range(NUM_OF_ORB):
#             for j in range(NUM_OF_SPO):
#                 count = source_density_map[i][j]  # 해당 구역에서 매 초 생성되는 패킷 수
#                 for _ in range(count):
#                     dest = random.choices(destination_population, weights=destination_weights, k=1)[0]
#                     packet = Packet(source=(i, j), destination=dest)
#                     grid[i][j].enqueue_packet(packet)
#
#         # 생성된 패킷들 처리: 각 위성이 보유한 패킷을 인접 위성으로 전달
#         for row in grid:
#             for sat in row:
#                 delivered = sat.process_queue(grid)
#                 delivered_packets.extend(delivered)
#
#         # 매 타임슬롯이 끝난 후, 위성 큐 상태 출력
#         if PRINT_QUEUE:
#             print_queue_status(grid, time)
#
#         time += 1
#
#     print(f"총 전달된 패킷 수: {len(delivered_packets)}")
#     return delivered_packets
#


# =============================================================================
# 시뮬레이션 함수
# =============================================================================
def simulate(simulation_time, source_density_map, dest_density_map):
    """
    시뮬레이션 함수:
      - 매 타임슬롯마다 출발지 density_map의 각 구역에서 정해진 개수의 패킷을 생성.
      - 패킷의 목적지는 dest_density_map의 가중치를 이용해 확률적으로 선택하되,
        출발지와 동일하지 않도록 함.
      - 생성된 패킷은 해당 위성의 대기열에 추가되고, 각 위성이 process_queue()를 호출하여 전달.
      - 매 타임슬롯마다 위성의 큐 상태와, 해당 타임슬롯에서 전달된 패킷의 세부 정보를 출력.

    :param simulation_time: 시뮬레이션 시간(초)
    :param source_density_map: 출발지용 density_map (패킷 생성률)
    :param dest_density_map: 목적지용 density_map (목적지 선택 시 가중치)
    :return: 최종 전달된 패킷 리스트
    """
    grid = create_satellite_grid()
    delivered_packets = []

    # 목적지 선택을 위한 population과 가중치 목록 준비
    destination_population = [(i, j) for i in range(NUM_OF_ORB) for j in range(NUM_OF_SPO)]
    destination_weights = [dest_density_map[i][j] for i in range(NUM_OF_ORB) for j in range(NUM_OF_SPO)]

    time = 0
    while time < simulation_time:
        # 매 타임슬롯마다 각 구역의 density 값만큼 패킷 생성
        for i in range(NUM_OF_ORB):
            for j in range(NUM_OF_SPO):
                count = source_density_map[i][j]
                for _ in range(count):
                    # 출발지와 다른 목적지를 선택할 때까지 재선택
                    while True:
                        dest = random.choices(destination_population, weights=destination_weights, k=1)[0]
                        if dest != (i, j):
                            break
                    packet = Packet(source=(i, j), destination=dest)
                    grid[i][j].enqueue_packet(packet)

        tick_delivered = []
        # 각 위성이 보유한 패킷 처리 (인접 위성으로 전달 및 최종 전달)
        for row in grid:
            for sat in row:
                delivered = sat.process_queue(grid)
                tick_delivered.extend(delivered)
                delivered_packets.extend(delivered)

        # 타임슬롯 종료 후, 위성 큐 상태와 전달된 패킷들의 세부 정보 출력
        print_queue_status(grid, time)
        if tick_delivered:
            print(f"Tick {time} - Delivered {len(tick_delivered)} packets:")
            for p in tick_delivered:
                print(f"  Packet ID: {p.id}, Source: {p.source}, Destination: {p.destination}, "
                      f"Hops: {p.hops}, Total Propagation Delay: {p.travel_time:.3f} ms")
        else:
            print(f"Tick {time} - No packets delivered.")
        print("=" * 40)

        time += 1

    print(f"총 전달된 패킷 수: {len(delivered_packets)}")
    return delivered_packets
