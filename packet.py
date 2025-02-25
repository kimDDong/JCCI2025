# =============================================================================
# Packet 클래스: 각 패킷에 고유 ID, 생성 시점, 경로(hops), 누적 지연(travel_time) 기록
# =============================================================================
class Packet:
    id_counter = 0  # 모든 패킷에 대해 고유 ID 부여
    def __init__(self, source, destination, sfc):
        """
        :param source: (x, y) 출발지 좌표
        :param destination: (x, y) 목적지 좌표 (출발지와 달라야 함)
        :param sfc: 선택된 SFC (예: ["FW", "IDS", "NAT"])
        """
        self.id = Packet.id_counter
        Packet.id_counter += 1
        self.source = source
        self.destination = destination
        self.sfc = sfc            # SFC 체인 (VNF 목록)
        self.sfc_index = 0        # 다음에 처리해야 할 VNF의 인덱스
        self.travel_time = 0      # 누적 지연 (ms 단위): propagation, queueing, processing, transmission delay 모두 포함
        self.last_enqueue_time = 0  # 패킷이 큐에 들어간 시각 (queueing delay 계산용)
        self.hops = []            # 패킷이 거친 위성들의 ID (처리 및 포워딩 정보 포함)
        self.creation_time = 0   # 패킷 생성 시점 (simulation 시작 후 ms)
        self.arrival_time = 0    # 패킷 도착 시점 (simulation 시작 후 ms)
        self.propagation_delay = 0
        self.queueing_delay = 0
        self.processing_delay = 0
        self.transmission_delay = 0
        self.last_enqueue_time = 0  # 패킷이 큐에 들어간 시각 (queueing delay 계산용)
        # 패킷이 다음 위성에서 처리 가능해지는 시각
        self.available_time = 0
        self.last_enqueue_time = 0  # 패킷이 큐에 들어간 시각 (queueing delay 계산용)

    @property
    def total_delay(self):
        return (self.propagation_delay +
                self.queueing_delay +
                self.processing_delay +
                self.transmission_delay)

    def __repr__(self):
        return (f"Packet(id={self.id}, source={self.source}, dest={self.destination}, "
                f"sfc={self.sfc}, current_sfc_index={self.sfc_index}, "
                f"hops={self.hops}, total_delay={self.total_delay:.3f} ms)")
