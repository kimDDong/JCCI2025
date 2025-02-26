# main.py
from formula import *
import copy
from datetime import datetime
import param

if __name__ == '__main__':
    source_map = copy.deepcopy(DENSITY_MAP)
    dest_map = copy.deepcopy(DENSITY_MAP)

    if param.SIMULATION_MODE == 1:
        start_time = datetime.now()
        delivered, grid = simulate(simulation_time= SIMULATION_TIME,
                                   source_density_map=source_map,
                                   dest_density_map=dest_map,
                                   start_time=start_time,
                                   install_time=0)
        finish_time = datetime.now()
        print(f"[모드 1] 실행 시간: {finish_time - start_time}, 전달된 패킷 수: {len(delivered)}")
        # 모드 1에서도 위성 로그 CSV 파일 생성
        if CSV:
            log_satellites_to_csv(grid, file_path=f".\csv\satellites_log_mode1_{SIMULATION_TIME}_{start_time.strftime('%m%d_%H%M')}.csv")

    elif param.SIMULATION_MODE == 2:
        start_time = datetime.now()
        for install_time in param.VNF_INSTALLATION_TIMES:
            param.CURRENT_VNF_INSTALLATION_TIME = install_time
            print(f"\n[모드 2] VNF 설치 시간 {install_time} ms 로 시뮬레이션 시작")
            source_map = copy.deepcopy(DENSITY_MAP)
            dest_map = copy.deepcopy(DENSITY_MAP)
            # 각 실험마다 새로운 grid 생성
            delivered, grid = simulate(simulation_time= SIMULATION_TIME,
                                       source_density_map=source_map,
                                       dest_density_map=dest_map,
                                       start_time=start_time,
                                       install_time=install_time)
            finish_time = datetime.now()
            print(f"VNF 설치 시간 {install_time} ms, 실행 시간: {finish_time - start_time}, 전달된 패킷 수: {len(delivered)}")
            if CSV:
                log_satellites_to_csv(grid, file_path=f".\csv\satellites_log_mode2[{install_time}ms]_{SIMULATION_TIME}_{start_time.strftime('%m%d_%H%M')}.csv")
