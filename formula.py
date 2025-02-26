#formula.py
import csv
import os
from param import *
from packet import *
import numpy as np
import random
from satellite import *
from tqdm import tqdm
import time

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

# =============================================================================
# 위성 그리드 생성 함수 (VNF 할당 포함)
# =============================================================================
def create_satellite_grid():
    """
    8x16 위성 그리드를 생성하여 2차원 리스트로 반환.
    각 위성은 생성 시 VNF를 할당받으며, 총 128개 위성에 대해 4종류의 VNF가
    각각 32개씩 할당되도록 함.
    """
    grid = []
    total_sat = NUM_OF_ORB * NUM_OF_SPO  # 예: 128
    # 각 VNF가 동일 개수씩 할당되도록 리스트 생성
    vnf_assignments = []
    for vnf in VNF_LIST:
        vnf_assignments.extend([vnf] * (total_sat // len(VNF_LIST)))
    # 할당 순서를 무작위로 섞음
    random.shuffle(vnf_assignments)

    index = 0
    for x in range(NUM_OF_ORB):
        row = []
        for y in range(NUM_OF_SPO):
            assigned_vnf = vnf_assignments[index]
            index += 1
            sat = Satellite(x, y, vnf=assigned_vnf)
            row.append(sat)
        grid.append(row)
    return grid

def print_queue_status(grid, time):
    print(f"Time {time}:")
    for row in grid:
        print(" ".join(f"{len(sat.queue):2d}" for sat in row))
    print("-" * 40)

def print_vnf_assignments(grid):
    print("VNF Assignment for each Satellite:")
    for row in grid:
        for sat in row:
            print(f"{sat.sat_id:<5}: {sat.vnf:<5}", end=" | ")
        print()
    print("-" * 40)

def log_packets_to_csv(tick_delivered, time, file_path="packets_log.csv"):
    headers = [
        "Tick", "Time_Start(ms)", "Time_End(ms)", "Packet_ID", "Source", "Destination",
        "Routing_Path(Hops)", "Propagation_Delay(ms)", "Queueing_Delay(ms)",
        "Processing_Delay(ms)", "Transmission_Delay(ms)", "Total_Delay(ms)",
        "Creation_Time(ms)", "Arrival_Time(ms)"
    ]
    write_header = not os.path.exists(file_path)
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(headers)
        for p in tick_delivered:
            writer.writerow([
                time, f"{time:.3f}", f"{time + 1:.3f}", p.id, p.source, p.destination,
                " -> ".join(map(str, p.hops)), f"{p.propagation_delay:.3f}", f"{p.queueing_delay:.3f}",
                f"{p.processing_delay:.3f}", f"{p.transmission_delay:.3f}", f"{p.total_delay:.3f}",
                f"{p.creation_time:.3f}", f"{p.arrival_time:.3f}"
            ])

def log_satellites_to_csv(grid, file_path="satellites_log.csv"):
    """
    각 위성의 통계 정보를 CSV 파일로 기록합니다.
    """
    headers = [
        "Satellite_ID", "Region", "Latitude", "Longitude",
        "Processed_Count", "Total_Processing_Delay", "Avg_Processing_Delay", "Final_Queue_Length"
    ]
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for row in grid:
            for sat in row:
                avg_delay = sat.total_processing_delay / sat.processed_count if sat.processed_count > 0 else 0
                writer.writerow([
                    sat.sat_id,
                    f"({sat.region_x}, {sat.region_y})",
                    f"{sat.lat:.2f}",
                    f"{sat.lon:.2f}",
                    sat.processed_count,
                    f"{sat.total_processing_delay:.3f}",
                    f"{avg_delay:.3f}",
                    len(sat.queue)
                ])



# ==================================
# 시뮬레이션 함수
# =============================================================================
def simulate(simulation_time, source_density_map, dest_density_map, start_time, install_time):
    """
    simulation_time: 전체 시뮬레이션 시간(ms 단위)
    """
    grid = create_satellite_grid()
    if PRINT and PRINT_VNF:
        print_vnf_assignments(grid)
    delivered_packets = []

    destination_population = [(i, j) for i in range(NUM_OF_ORB) for j in range(NUM_OF_SPO)]
    destination_weights = [dest_density_map[i][j] for i in range(NUM_OF_ORB) for j in range(NUM_OF_SPO)]

    time = 0  # ms 단위 시작 시간
    for _ in tqdm(range(simulation_time)):
        # 각 tick마다 패킷 생성
        for i in range(NUM_OF_ORB):
            for j in range(NUM_OF_SPO):
                count = source_density_map[i][j]
                for _ in range(count):
                    while True:
                        dest = random.choices(destination_population, weights=destination_weights, k=1)[0]
                        if dest != (i, j):
                            break
                    sfc = random.choice(SFC_LIST)
                    packet = Packet(source=(i, j), destination=dest, sfc=sfc)
                    packet.creation_time = time
                    grid[i][j].enqueue_packet(packet, time)
        tick_delivered = []
        for row in grid:
            for sat in row:
                delivered = sat.process_queue(grid, time)
                tick_delivered.extend(delivered)
                delivered_packets.extend(delivered)

        if tick_delivered:
            if CSV:
                if SIMULATION_MODE == 1:
                    log_packets_to_csv(tick_delivered, time, file_path=f".\csv\packets_log_mode1_{SIMULATION_TIME}_{start_time.strftime('%m%d_%H%M')}.csv")
                elif SIMULATION_MODE == 2:
                    log_packets_to_csv(tick_delivered, time, file_path=f".\csv\packets_log_mode2[{install_time}ms]_{SIMULATION_TIME}_{start_time.strftime('%m%d_%H%M')}.csv")
            if PRINT and PRINT_PACKET:
                print(f"Tick {time} (Time {time:.3f}ms ~ {time + 1:.3f}ms):")
                for p in tick_delivered:
                    if PRINT and PRINT_PATH:
                        print(f"  Packet ID: {p.id}, Source: {p.source}, Destination: {p.destination}")
                        print(f"    Routing Path (Hops): {p.hops}")
                    if PRINT and PRINT_DELAY:
                        print(f"    Propagation Delay: {p.propagation_delay:.3f} ms, Queueing Delay: {p.queueing_delay:.3f} ms")
                        print(f"    Processing Delay: {p.processing_delay:.3f} ms, Transmission Delay: {p.transmission_delay:.3f} ms")
                        print(f"    Total Delay: {p.total_delay:.3f} ms")
                        print(f"    Created at: {p.creation_time:.3f} ms, Arrived at: {p.arrival_time:.3f} ms")
                    print("=" * 40)
            # else:
            #     print("  No packets delivered.")


        time += 1
    print(f"총 전달된 패킷 수: {len(delivered_packets)}")
    return delivered_packets, grid


