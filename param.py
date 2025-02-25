# param.py
# PRINT FORMAT
PRINT_VNF = 1
PRINT_PATH = 1
PRINT_DELAY = 1
PRINT_PACKET = 1
PRINT = 1


# Simulation Parameter
SIMULATION_TIME = 60  # [ms]

# 시뮬레이션 모드 선택: 1이면 기존(고정 VNF 배치), 2이면 "on‐the‐fly VNF 설치" 알고리즘
SIMULATION_MODE = 2  # 사용자가 1 또는 2로 설정

# VNF 설치 시간 실험: 각 값(ms)마다 실험 진행
VNF_INSTALLATION_TIMES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
CURRENT_VNF_INSTALLATION_TIME = VNF_INSTALLATION_TIMES[0]  # 시뮬레이션 실행 시 main.py에서 업데이트됨

# Network Parameter
EARTH_RADIUS = 6371          # [km]
LIGHT_SPEED = 299.792458     # [km/ms]

# Constellation param
SAT_HEIGHT = 550        # [km]
NUM_OF_ORB = 8
NUM_OF_SPO = 16
NUM_OF_SAT = NUM_OF_ORB * NUM_OF_SPO

# VNF 및 SFC 관련 파라미터
VNF_LIST = ["FW", "IDS", "NAT", "LB"]
SFC_LIST = [
    ["FW", "IDS", "NAT"],  # 보안 중심 SFC
    ["LB", "NAT", "FW"]    # 부하 분산 중심 SFC
]

# CPU 및 Bandwidth 파라미터
VNF_CPU_REQUIREMENTS = {
    "FW": 10,    # 예: 10 ms CPU 요구량
    "IDS": 20,
    "NAT": 15,
    "LB": 5
}
SATELLITE_CPU_CAPACITY = 100

VNF_BW_REQUIREMENTS = {
    "FW": 2,
    "IDS": 3,
    "NAT": 2.5,
    "LB": 1
}
SATELLITE_BW_CAPACITY = 10

# (기타 기존 파라미터 그대로 유지)

# 패킷 생성률 등
DENSITY_MAP = [
    [9,1,1,1,1,1,1,1,14,14,8,8,6,6,5,3],
    [8,14,32,44,48,26,1,44,82,38,30,26,41,48,29,6],
    [2,2,55,100,76,14,1,36,42,23,15,30,88,98,65,1],
    [3,2,2,18,12,3,1,6,6,9,14,38,29,30,14,6],
    [6,2,1,1,9,24,23,1,1,6,3,2,18,23,17,5],
    [1,1,1,1,11,29,5,1,9,12,2,1,2,14,9,6],
    [1,1,1,1,5,3,1,1,1,1,1,1,1,2,1,5],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
]
