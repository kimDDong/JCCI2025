# param.py

# 출력 및 CSV 관련 설정
PRINT_VNF = True
PRINT_PATH = True
PRINT_DELAY = True
PRINT_PACKET = True
PRINT_QUEUE = True
PRINT = True
CSV = True

# 시뮬레이션 실행 시간 (ms 단위)
SIMULATION_TIME = 1 * 60 * 1000
# SIMULATION_TIME = 15

# # 시뮬레이션 모드: 0 = 데이터셋 생성 모드, 1 = 고정 VNF 배치, 2 = on‐the‐fly VNF 설치
SIMULATION_MODE = 0

# VNF 설치 시간 실험 (ms)
VNF_INSTALLATION_TIMES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
CURRENT_VNF_INSTALLATION_TIME = VNF_INSTALLATION_TIMES[0]
