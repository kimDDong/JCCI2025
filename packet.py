# =============================================================================
# Packet 클래스: 각 패킷에 고유 ID, 생성 시점, 경로(hops), 누적 전파 지연(travel_time) 기록
# =============================================================================
class Packet:
    id_counter = 0  # 모든 패킷에 대해 고유 ID 부여

    def __init__(self, source, destination):
        """
        :param source: (x, y) 출발지 좌표
        :param destination: (x, y) 목적지 좌표
        """
        self.id = Packet.id_counter
        Packet.id_counter += 1
        self.source = source
        self.destination = destination
        self.hops = []         # 패킷이 거쳐간 위성의 ID 리스트
        self.travel_time = 0   # 누적 propagation delay (ms 단위)

    def __repr__(self):
        return (f"Packet(id={self.id}, source={self.source}, dest={self.destination}, "
                f"hops={self.hops}, travel_time={self.travel_time:.3f} ms)")
