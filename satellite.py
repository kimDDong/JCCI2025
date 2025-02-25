#satellite.py
import random
from collections import deque
import numpy as np
from param import *
from packet import *
import heapq
import formula

def get_lat_lon(sat):
    lat_divs = np.linspace(-90, 90, NUM_OF_ORB + 1)  # 8개 구간, 9개 경계점
    lon_divs = np.linspace(-180, 180, NUM_OF_SPO + 1)  # 16개 구간, 17개 경계점
    lat_centers = (lat_divs[:-1] + lat_divs[1:]) / 2
    lon_centers = (lon_divs[:-1] + lon_divs[1:]) / 2
    return lat_centers[sat.region_x], lon_centers[sat.region_y]


# [추가: 다중 목적지 Dijkstra 함수]
def dijkstra_path_to_targets(grid, start, targets):
    rows, cols = len(grid), len(grid[0])
    dist = { (i, j): float('inf') for i in range(rows) for j in range(cols) }
    prev = { (i, j): None for i in range(rows) for j in range(cols) }
    dist[start] = 0
    pq = [(0, start)]
    reached = None
    while pq:
        d, node = heapq.heappop(pq)
        if node in targets:
            reached = node
            break
        if d > dist[node]:
            continue
        x, y = node
        neighbors = []
        if x > 0: neighbors.append((x-1, y))
        if x < rows - 1: neighbors.append((x+1, y))
        if y > 0: neighbors.append((x, y-1))
        if y < cols - 1: neighbors.append((x, y+1))
        for n in neighbors:
            sat_current = grid[x][y]
            sat_neighbor = grid[n[0]][n[1]]
            delay = formula.calculate_propagation_delay(sat_current, sat_neighbor)
            weight = delay if not np.isnan(delay) else float('inf')
            if d + weight < dist[n]:
                dist[n] = d + weight
                prev[n] = node
                heapq.heappush(pq, (dist[n], n))
    if reached is None:
        return []
    path = []
    node = reached
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path

def dijkstra_next_hop_extended(grid, current, targets):
    path = dijkstra_path_to_targets(grid, current, targets)
    return path[1] if len(path) >= 2 else None





# =============================================================================
# Dijkstra 알고리즘 기반 경로 탐색 함수
# =============================================================================
def dijkstra_path(grid, start, destination):
    rows, cols = len(grid), len(grid[0])
    dist = {(i, j): float('inf') for i in range(rows) for j in range(cols)}
    prev = {(i, j): None for i in range(rows) for j in range(cols)}
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        d, node = heapq.heappop(pq)
        if node == destination:
            break
        if d > dist[node]:
            continue
        x, y = node
        # 인접 노드: 상하좌우
        neighbors = []
        if x > 0:
            neighbors.append((x - 1, y))
        if x < rows - 1:
            neighbors.append((x + 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if y < cols - 1:
            neighbors.append((x, y + 1))

        for n in neighbors:
            sat_current = grid[x][y]
            sat_neighbor = grid[n[0]][n[1]]
            delay = formula.calculate_propagation_delay(sat_current, sat_neighbor)
            if np.isnan(delay):
                weight = float('inf')
            else:
                weight = delay
            if d + weight < dist[n]:
                dist[n] = d + weight
                prev[n] = node
                heapq.heappush(pq, (dist[n], n))

    # 경로 복원 (start부터 destination까지)
    path = []
    node = destination
    if prev[node] is None and node != start:
        return []  # 경로 없음
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path


def dijkstra_next_hop(grid, current, destination):
    path = dijkstra_path(grid, current, destination)
    if len(path) < 2:
        return None
    return path[1]


# =============================================================================
# Satellite 클래스 (Dijkstra 기반 next hop 적용, FIFO 대기열 포함)
# =============================================================================
class Satellite:
    def __init__(self, region_x, region_y, sat_id=None, vnf=None):
        self.region_x = region_x
        self.region_y = region_y
        self.region_index = (region_x, region_y)
        self.sat_id = sat_id if sat_id is not None else f"Sat_{region_x}_{region_y}"
        self.lat, self.lon = formula.get_lat_lon(self)
        self.vnf = vnf  # 모드 2에서는 사용하지 않음 (on-the-fly 설치)
        self.status = "active"
        self.vnf_status = {}
        self.queue = deque()

    def enqueue_packet(self, packet, scheduled_time):
        packet.last_enqueue_time = scheduled_time
        packet.available_time = scheduled_time
        self.queue.append(packet)

    def process_queue(self, grid, current_time):
        delivered_packets = []
        num_packets = len(self.queue)
        for _ in range(num_packets):
            packet = self.queue.popleft()
            # 아직 처리 가능 시각이 아니면 다시 대기열에 넣음
            if current_time < packet.available_time:
                self.queue.append(packet)
                continue

            # 공통: 큐잉 지연 계산
            queue_delay = current_time - packet.last_enqueue_time
            packet.queueing_delay += queue_delay

            if SIMULATION_MODE == 1:
                # [모드 1] 기존 알고리즘: 위성이 미리 할당받은 VNF 사용
                if packet.sfc_index < len(packet.sfc) and self.vnf == packet.sfc[packet.sfc_index]:
                    proc_delay = VNF_CPU_REQUIREMENTS[self.vnf] / SATELLITE_CPU_CAPACITY
                    trans_delay = VNF_BW_REQUIREMENTS[self.vnf] / SATELLITE_BW_CAPACITY
                    packet.processing_delay += proc_delay
                    packet.transmission_delay += trans_delay
                    packet.hops.append(f"{self.sat_id}(processed {self.vnf})")
                    packet.sfc_index += 1
                    packet.available_time = current_time + proc_delay + trans_delay

                packet.hops.append(self.sat_id)

                if packet.sfc_index < len(packet.sfc):
                    target_vnf = packet.sfc[packet.sfc_index]
                    targets = {(i, j) for i in range(NUM_OF_ORB) for j in range(NUM_OF_SPO)
                               if grid[i][j].vnf == target_vnf}
                else:
                    targets = {packet.destination}

                if (self.region_x, self.region_y) in targets:
                    if (self.region_x, self.region_y) == packet.destination:
                        packet.arrival_time = packet.creation_time + packet.total_delay
                        delivered_packets.append(packet)
                    else:
                        packet.last_enqueue_time = current_time
                        packet.available_time = current_time
                        self.queue.append(packet)
                    continue

                current_coord = (self.region_x, self.region_y)
                next_coord = formula.dijkstra_next_hop_extended(grid, current_coord, targets)
                if next_coord is not None:
                    next_sat = grid[next_coord[0]][next_coord[1]]
                    link_delay = formula.calculate_propagation_delay(self, next_sat)
                    if np.isnan(link_delay):
                        link_delay = float('inf')
                    packet.propagation_delay += link_delay
                    next_sat.enqueue_packet(packet, current_time + link_delay)
                else:
                    packet.last_enqueue_time = current_time
                    packet.available_time = current_time
                    self.queue.append(packet)
            elif SIMULATION_MODE == 2:
                # [모드 2] on-the-fly VNF 설치 알고리즘
                if packet.sfc_index < len(packet.sfc):
                    required_vnf = packet.sfc[packet.sfc_index]
                    # VNF 설치 지연 (실험 변수)
                    installation_delay = CURRENT_VNF_INSTALLATION_TIME
                    proc_delay = VNF_CPU_REQUIREMENTS[required_vnf] / SATELLITE_CPU_CAPACITY
                    trans_delay = VNF_BW_REQUIREMENTS[required_vnf] / SATELLITE_BW_CAPACITY
                    total_proc = installation_delay + proc_delay + trans_delay

                    # 처리 지연에 설치 지연 포함
                    packet.processing_delay += installation_delay + proc_delay
                    packet.transmission_delay += trans_delay

                    packet.hops.append(f"{self.sat_id}(installed {required_vnf})")
                    packet.sfc_index += 1

                    # 설치 및 처리 후, 패킷은 total_proc 만큼의 시간이 지난 후에 처리 가능
                    packet.available_time = current_time + total_proc

                    # **즉시 추가 처리를 하지 않고 재대기**:
                    packet.last_enqueue_time = current_time
                    self.queue.append(packet)
                    continue  # 이번 tick에서는 여기서 멈추고 다음 tick에서 다시 처리

                # 이후의 처리 (VNF 설치가 완료된 패킷에 대해)
                packet.hops.append(self.sat_id)

                # 목적지 도달 확인
                if (self.region_x, self.region_y) == packet.destination:
                    # SFC 처리 완료 시 패킷 도착
                    if packet.sfc_index == len(packet.sfc):
                        # 모드 1과 동일하게, arrival_time을 creation_time + total_delay로 계산
                        packet.arrival_time = packet.creation_time + packet.total_delay
                        delivered_packets.append(packet)
                    else:
                        packet.last_enqueue_time = current_time
                        packet.available_time = current_time
                        self.queue.append(packet)
                    continue

                # 라우팅: 단순히 최종 목적지로 진행
                current_coord = (self.region_x, self.region_y)
                next_coord = formula.dijkstra_next_hop(grid, current_coord, packet.destination)
                if next_coord is not None:
                    next_sat = grid[next_coord[0]][next_coord[1]]
                    link_delay = formula.calculate_propagation_delay(self, next_sat)
                    if np.isnan(link_delay):
                        link_delay = float('inf')
                    packet.propagation_delay += link_delay
                    next_sat.enqueue_packet(packet, current_time + link_delay)
                else:
                    packet.last_enqueue_time = current_time
                    packet.available_time = current_time
                    self.queue.append(packet)
        return delivered_packets

    def __repr__(self):
        return (f"<Satellite id={self.sat_id}, region_index={self.region_index}, "
                f"lat={self.lat:.2f}, lon={self.lon:.2f}, vnf={self.vnf}>")


# =============================================================================
# next_hop 함수: 현재 위성에서 목적지까지 맨해튼 거리를 줄일 수 있는 인접 위성을 임의 선택
# =============================================================================
def next_hop(current, destination):
    cur_x, cur_y = current
    dest_x, dest_y = destination
    dx = dest_x - cur_x
    dy = dest_y - cur_y
    options = []

    # X축 이동 옵션
    if dx > 0:
        options.append((cur_x + 1, cur_y))
    elif dx < 0:
        options.append((cur_x - 1, cur_y))

    # Y축 이동 옵션
    if dy > 0:
        options.append((cur_x, cur_y + 1))
    elif dy < 0:
        options.append((cur_x, cur_y - 1))

    valid_options = [(x, y) for x, y in options if 0 <= x < NUM_OF_ORB and 0 <= y < NUM_OF_SPO]
    if not valid_options:
        return None
    return random.choice(valid_options)


# =============================================================================
# 위성 그리드 생성 및 상태 출력 함수들
# =============================================================================
def create_satellite_grid():
    """
    8x16 위성 그리드를 생성하여 2차원 리스트로 반환.
    각 위성은 FIFO 대기열을 갖는 Satellite 객체.
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
    매 타임슬롯마다 각 위성의 대기열에 쌓인 패킷 수를 2차원 배열 형식으로 출력.
    """
    print(f"Time {time}:")
    for row in grid:
        print(" ".join(f"{len(sat.queue):2d}" for sat in row))
    print("-" * 40)

