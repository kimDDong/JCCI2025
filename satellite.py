# satellite.py
import random
from collections import deque
import numpy as np
from config import NUM_OF_ORB, NUM_OF_SPO
import routing
import param
import vnf
import formula

class Satellite:
    def __init__(self, region_x, region_y, sat_id=None, vnf_assignment=None):
        self.region_x = region_x
        self.region_y = region_y
        self.region_index = (region_x, region_y)
        self.sat_id = sat_id if sat_id is not None else f"Sat_{region_x}_{region_y}"
        self.lat, self.lon = formula.get_lat_lon(self)
        self.vnf = vnf_assignment  # 고정 VNF 모드 (SIMULATION_MODE 1) 시 사용
        self.status = "active"
        self.queue = deque()
        # 위성 처리 통계
        self.processed_count = 0
        self.total_processing_delay = 0

    def enqueue_packet(self, packet, scheduled_time):
        packet.last_enqueue_time = scheduled_time
        packet.available_time = scheduled_time
        self.queue.append(packet)

    def process_queue(self, grid, current_time):
        delivered_packets = []
        num_packets = len(self.queue)
        for _ in range(num_packets):
            packet = self.queue.popleft()
            if current_time < packet.available_time:
                self.queue.append(packet)
                continue

            # 큐잉 지연 계산
            queue_delay = current_time - packet.last_enqueue_time
            packet.queueing_delay += queue_delay

            if param.SIMULATION_MODE == 1:
                # [모드 1] 고정 VNF 할당
                if packet.sfc_index < len(packet.sfc) and self.vnf == packet.sfc[packet.sfc_index]:
                    proc_delay = vnf.VNF_CPU_REQUIREMENTS[self.vnf] / vnf.SATELLITE_CPU_CAPACITY
                    trans_delay = vnf.VNF_BW_REQUIREMENTS[self.vnf] / vnf.SATELLITE_BW_CAPACITY
                    packet.processing_delay += proc_delay
                    packet.transmission_delay += trans_delay
                    packet.hops.append(f"{self.sat_id}(processed {self.vnf})")
                    packet.sfc_index += 1
                    packet.available_time = current_time + proc_delay + trans_delay

                    self.processed_count += 1
                    self.total_processing_delay += proc_delay
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
                next_coord = routing.dijkstra_next_hop_extended(grid, current_coord, targets)
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

            elif param.SIMULATION_MODE == 2:
                # [모드 2] on-the-fly VNF 설치
                if packet.sfc_index < len(packet.sfc):
                    required_vnf = packet.sfc[packet.sfc_index]
                    installation_delay = param.CURRENT_VNF_INSTALLATION_TIME
                    proc_delay = vnf.VNF_CPU_REQUIREMENTS[required_vnf] / vnf.SATELLITE_CPU_CAPACITY
                    trans_delay = vnf.VNF_BW_REQUIREMENTS[required_vnf] / vnf.SATELLITE_BW_CAPACITY
                    total_proc = installation_delay + proc_delay + trans_delay

                    packet.processing_delay += installation_delay + proc_delay
                    packet.transmission_delay += trans_delay

                    packet.hops.append(f"{self.sat_id}(installed {required_vnf})")
                    packet.sfc_index += 1

                    self.processed_count += 1
                    self.total_processing_delay += (installation_delay + proc_delay)

                    packet.available_time = current_time + total_proc
                    packet.last_enqueue_time = current_time
                    self.queue.append(packet)
                    continue

                packet.hops.append(self.sat_id)

                if (self.region_x, self.region_y) == packet.destination:
                    if packet.sfc_index == len(packet.sfc):
                        packet.arrival_time = packet.creation_time + packet.total_delay
                        delivered_packets.append(packet)
                    else:
                        packet.last_enqueue_time = current_time
                        packet.available_time = current_time
                        self.queue.append(packet)
                    continue

                current_coord = (self.region_x, self.region_y)
                next_coord = routing.dijkstra_next_hop(grid, current_coord, packet.destination)
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
        return f"<Satellite id={self.sat_id}, region={self.region_index}, lat={self.lat:.2f}, lon={self.lon:.2f}, vnf={self.vnf}>"
