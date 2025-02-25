# PRINT FORMAT
PRINT_QUEUE = 0
PRINT_PACKET = 0
PRINT_VNF = 0

# Simulation Parameter
SIMULATION_TIME = 1 # [min]

# Network Parameter
EARTH_RADIUS = 6371          # [km]
LIGHT_SPEED = 299.792458     # [km/ms] 빛의 속도
    # PACKET_SIZE = 1         # [KB]
    # LINK_CAPACITY = 260      # [Mbps]
    # PACKET_PER_MS = 33
    # PACKET_MAX_TTL = 1000

# Constellation param
SAT_HEIGHT = 550        # [km]
NUM_OF_ORB = 8
NUM_OF_SPO = 16
NUM_OF_SAT = NUM_OF_ORB * NUM_OF_SPO
    # INIT_LATITUDE = 101.25
    # INCLINATION = 90
    # CONSTELLATION_PARAM_F = 0
    # POLAR_LATITUDE = 88

# Satellite spec
# BW_CAPACITY
# CPU_CAPACITY

VNF_LIST = ["FW", "IDS", "NAT", "LB"]
SFC_LIST = [
    ["FW", "IDS", "NAT"], # 보안 중심 SFC
    ["LB", "NAT", "FW"]   # 부하 분산 중심 SFC
            ]

VNF_installation_time = 10       # ms

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
