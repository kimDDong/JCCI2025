# =============================================================================
# Packet 클래스: 각 패킷에 고유 ID, 생성 시점, 경로(hops), 누적 전파 지연(travel_time) 기록
# =============================================================================
# [수정된 Packet 클래스]
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
        self.hops = []            # 패킷이 거친 위성들의 ID (처리 및 포워딩 정보 포함)
        self.travel_time = 0      # 누적 propagation delay (ms 단위)

    def __repr__(self):
        return (f"Packet(id={self.id}, source={self.source}, dest={self.destination}, "
                f"sfc={self.sfc}, current_sfc_index={self.sfc_index}, "
                f"hops={self.hops}, travel_time={self.travel_time:.3f} ms)")

