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


# =============================================================================
# Dijkstra 알고리즘 기반 경로 탐색 함수
# =============================================================================
def dijkstra_path(grid, start, destination):
    """
    grid: 2차원 위성 배열
    start, destination: (x, y) 좌표
    각 인접 위성 간의 비용은 calculate_propagation_delay()로 구함.
    """
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
            # 만약 link가 지구에 의해 가려져 np.nan이 나오면 매우 큰 비용으로 취급
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
    """
    grid: 2차원 위성 배열
    current: (x, y) 현재 위성 좌표
    destination: (x, y) 패킷의 최종 목적지 좌표
    Dijkstra 알고리즘을 통해 얻은 최단 경로에서 바로 다음 위성 좌표 반환.
    """
    path = dijkstra_path(grid, current, destination)
    if len(path) < 2:
        return None
    return path[1]


# =============================================================================
# Satellite 클래스 (Dijkstra 기반 next hop 적용, FIFO 대기열 포함)
# =============================================================================
class Satellite:
    def __init__(self, region_x, region_y, sat_id=None, vnf=None):
        """
        위성 객체 생성

        :param region_x: 위성의 위도 구역 인덱스 (0 ~ 7)
        :param region_y: 위성의 경도 구역 인덱스 (0 ~ 15)
        :param sat_id: 위성의 식별자 (지정하지 않으면 자동 생성됨)
        :param vnf: 할당된 VNF (예: ['FW'])
        """
        self.region_x = region_x
        self.region_y = region_y
        self.region_index = (region_x, region_y)
        self.sat_id = sat_id if sat_id is not None else f"Sat_{region_x}_{region_y}"
        self.lat, self.lon = get_lat_lon(self)
        self.vnf = vnf
        self.status = "active"
        self.vnf_status = {}
        self.queue = deque()  # FIFO 대기열

    def assign_vnf(self, vnf):
        self.vnf = vnf

    def update_vnf_status(self, vnf_name, status):
        self.vnf_status[vnf_name] = status

    def enqueue_packet(self, packet):
        """패킷을 대기열에 추가"""
        self.queue.append(packet)

    def process_queue(self, grid):
        """
        위성의 대기열에 있는 패킷을 처리합니다.
          - 패킷이 최종 목적지에 도달하면 전달 완료.
          - 아니라면 Dijkstra 알고리즘으로 결정한 인접 위성으로 전달.

        현재 위성에서 다음 홉으로 이동 시, 해당 링크의 propagation delay를
        계산하여 packet.travel_time에 누적합니다.
        또한 현재 위성의 ID를 packet.hops에 추가합니다.

        :param grid: 전체 위성 그리드 (2차원 리스트)
        :return: 현재 위성이 최종 목적지에 도달시킨 패킷 리스트
        """
        delivered_packets = []
        num_packets = len(self.queue)
        for _ in range(num_packets):
            packet = self.queue.popleft()
            # 현재 위성을 거친 것으로 기록
            packet.hops.append(self.sat_id)

            if (self.region_x, self.region_y) == packet.destination:
                delivered_packets.append(packet)
            else:
                # Dijkstra 기반으로 다음 홉 결정
                current_coord = (self.region_x, self.region_y)
                next_coord = dijkstra_next_hop(grid, current_coord, packet.destination)
                if next_coord is not None:
                    next_sat = grid[next_coord[0]][next_coord[1]]
                    # 현재 위성에서 다음 위성까지의 propagation delay 계산
                    link_delay = formula.calculate_propagation_delay(self, next_sat)
                    if np.isnan(link_delay):
                        link_delay = float('inf')
                    packet.travel_time += link_delay
                    next_sat.enqueue_packet(packet)
                else:
                    # 다음 홉 결정 실패 시 (경로 없음), 다시 대기열에 넣음
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

