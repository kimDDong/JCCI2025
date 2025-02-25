#main.py
from formula import *
import copy
import time

if __name__ == '__main__':
    # 예시 density_map (출발지 및 목적지용)
    source_map = copy.deepcopy(DENSITY_MAP)
    dest_map = copy.deepcopy(DENSITY_MAP)

    start_time = time.time()
    delivered = simulate(simulation_time= 2*SIMULATION_TIME, source_density_map=source_map,
                         dest_density_map=dest_map)
    finish_time = time.time()
    print(f"시뮬레이션 실행 시간: {finish_time - start_time:.2f} sec")
