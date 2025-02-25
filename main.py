from formula import *
import copy


if __name__ == '__main__':
    # 예시 density_map (출발지 및 목적지용)
    source_map = copy.deepcopy(DENSITY_MAP)
    dest_map = copy.deepcopy(DENSITY_MAP)

    # 예를 들어 50초 동안 시뮬레이션 실행
    delivered = simulate(simulation_time= 1*SIMULATION_TIME, source_density_map=source_map,
                         dest_density_map=dest_map)
