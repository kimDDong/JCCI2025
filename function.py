# function.py
import copy
import csv
import os
import random
from tqdm import tqdm
from datetime import datetime
import param
import config
import vnf
import sfc
from satellite import Satellite
from packet import Packet
import formula

# function.py
import random
from packet import Packet
import config
import sfc


def generate_packet_dataset(simulation_time):
    """
    DENSITY_MAP 기반으로 각 영역에서 시뮬레이션 전체 시간(tick) 동안
    패킷을 생성하여 리스트로 반환.

    simulation_time: 전체 시뮬레이션 시간 (tick 단위)
    """
    dataset = []
    destination_population = [(i, j) for i in range(config.NUM_OF_ORB) for j in range(config.NUM_OF_SPO)]
    destination_weights = [config.DENSITY_MAP[i][j] for i in range(config.NUM_OF_ORB) for j in range(config.NUM_OF_SPO)]
    for tick in tqdm(range(simulation_time)):
        for i in range(config.NUM_OF_ORB):
            for j in range(config.NUM_OF_SPO):
                count = config.DENSITY_MAP[i][j]
                for _ in range(count):
                    # 출발지와 목적지가 동일하지 않도록 선택
                    while True:
                        dest = random.choices(destination_population, weights=destination_weights, k=1)[0]
                        if dest != (i, j):
                            break
                    sfc_choice = random.choice(sfc.SFC_LIST)
                    packet = Packet(source=(i, j), destination=dest, sfc=sfc_choice)
                    packet.creation_time = tick  # 현재 tick을 생성 시각으로 기록
                    dataset.append(packet)
    return dataset


def create_satellite_grid(assign_vnf=False):
    """
    8x16 위성 그리드 생성.
    assign_vnf가 True이면 고정 VNF 모드용으로 위성마다 VNF를 할당.
    """
    grid = []
    total_sat = config.NUM_OF_ORB * config.NUM_OF_SPO
    if assign_vnf:
        vnf_assignments = []
        for v in vnf.VNF_LIST:
            vnf_assignments.extend([v] * (total_sat // len(vnf.VNF_LIST)))
        random.shuffle(vnf_assignments)
    index = 0
    for i in range(config.NUM_OF_ORB):
        row = []
        for j in range(config.NUM_OF_SPO):
            assigned_vnf = vnf_assignments[index] if assign_vnf else None
            index += 1
            sat = Satellite(i, j, vnf_assignment=assigned_vnf)
            row.append(sat)
        grid.append(row)
    return grid


def print_queue_status(grid, time):
    """
    각 타임슬롯에서 위성들의 큐 길이를 2차원 배열 형태로 출력
    """
    print(f"Time {time}:")
    for row in grid:
        print(" ".join(f"{len(sat.queue):2d}" for sat in row))
    print("-" * 40)


def print_vnf_assignments(grid):
    """
    위성별 VNF 할당 상태 출력
    """
    print("VNF Assignment for each Satellite:")
    for row in grid:
        for sat in row:
            print(f"{sat.sat_id:<10}: {sat.vnf}", end=" | ")
        print()
    print("-" * 40)

def log_packets_to_csv(packets, tick, start_time, file_path="packets_log.csv"):
    if param.SIMULATION_MODE == 0:
        headers = ["Tick", "Packet_ID", "Source", "Destination"]
        format_row = lambda p: [p.creation_time, p.id, p.source, p.destination]
    else:
        headers = [
            "Tick", "Time_Start(ms)", "Time_End(ms)", "Packet_ID", "Source", "Destination",
            "Routing_Path(Hops)", "Propagation_Delay(ms)", "Queueing_Delay(ms)",
            "Processing_Delay(ms)", "ransmission_Delay(ms)", "Total_Delay(ms)",
            "Creation_Time(ms)", "Arrival_Time(ms)"
        ]
        format_row = lambda p: [
            p.creation_time, f"{tick:.3f}", f"{(tick + 1):.3f}", p.id, p.source, p.destination,
            " -> ".join(map(str, p.hops)), f"{p.propagation_delay:.3f}", f"{p.queueing_delay:.3f}",
            f"{p.processing_delay:.3f}", f"{p.transmission_delay:.3f}", f"{p.total_delay:.3f}",
            f"{p.creation_time:.3f}", f"{p.arrival_time:.3f}"
        ]

    write_header = not os.path.exists(file_path)
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(headers)
        writer.writerows(format_row(p) for p in packets)

def log_satellites_to_csv(grid, start_time, file_path="satellites_log.csv"):
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


def simulate(simulation_time, start_time, install_time=0):
    """
    전체 시뮬레이션을 주어진 시간(밀리초 단위) 동안 실행.
    모드 1: 고정 VNF 할당, 모드 2: on-the-fly VNF 설치
    """
    if param.SIMULATION_MODE == 1:
        grid = create_satellite_grid(assign_vnf=True)
        if param.PRINT and param.PRINT_VNF:
            print_vnf_assignments(grid)
    else:
        grid = create_satellite_grid(assign_vnf=False)

    delivered_packets = []
    destination_population = [(i, j) for i in range(config.NUM_OF_ORB) for j in range(config.NUM_OF_SPO)]
    destination_weights = [config.DENSITY_MAP[i][j] for i in range(config.NUM_OF_ORB) for j in range(config.NUM_OF_SPO)]

    time_tick = 0  # 시뮬레이션 시간 (ms)
    for _ in tqdm(range(simulation_time)):
        # 각 위성 영역별 패킷 생성 (source density 반영)
        for i in range(config.NUM_OF_ORB):
            for j in range(config.NUM_OF_SPO):
                count = config.DENSITY_MAP[i][j]
                for _ in range(count):
                    while True:
                        dest = random.choices(destination_population, weights=destination_weights, k=1)[0]
                        if dest != (i, j):
                            break
                    sfc_choice = random.choice(sfc.SFC_LIST)
                    packet = Packet(source=(i, j), destination=dest, sfc=sfc_choice)
                    packet.creation_time = time_tick
                    grid[i][j].enqueue_packet(packet, time_tick)
        tick_delivered = []
        for row in grid:
            for sat in row:
                delivered = sat.process_queue(grid, time_tick)
                tick_delivered.extend(delivered)
                delivered_packets.extend(delivered)
        if param.PRINT and param.PRINT_QUEUE:
            print_queue_status(grid, time_tick)
        if tick_delivered and param.CSV:
            if param.SIMULATION_MODE == 1:
                log_packets_to_csv(tick_delivered, time_tick, start_time,
                                   file_path=f"./csv/packets_log_mode1_{simulation_time}_{start_time.strftime('%m%d_%H%M')}.csv")
            else:
                log_packets_to_csv(tick_delivered, time_tick, start_time,
                                   file_path=f"./csv/packets_log_mode2[{install_time}ms]_{simulation_time}_{start_time.strftime('%m%d_%H%M')}.csv")
        time_tick += 1
    print(f"총 전달된 패킷 수: {len(delivered_packets)}")
    return delivered_packets, grid
