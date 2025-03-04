# packet.py

class Packet:
    id_counter = 0

    def __init__(self, source, destination, sfc):
        """
        :param source: (x, y) 출발 위성 영역
        :param destination: (x, y) 목적지 위성 영역
        :param sfc: VNF 체인 (리스트)
        """
        self.id = Packet.id_counter
        Packet.id_counter += 1
        self.source = source
        self.destination = destination
        self.sfc = sfc
        self.sfc_index = 0  # 다음 처리할 VNF 인덱스
        self.travel_time = 0  # 누적 지연 (ms)
        self.hops = []  # 거친 위성 ID 기록
        self.creation_time = 0  # 패킷 생성 시각
        self.arrival_time = 0   # 도착 시각
        self.propagation_delay = 0
        self.queueing_delay = 0
        self.processing_delay = 0
        self.transmission_delay = 0
        self.available_time = 0  # 다음 처리 가능 시각
        self.last_enqueue_time = 0  # 마지막 큐 인입 시각

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
