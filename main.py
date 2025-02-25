# main.py
from formula import *
import copy
import time
import param  # param 모듈을 임포트하여 파라미터에 접근

if __name__ == '__main__':
    source_map = copy.deepcopy(DENSITY_MAP)
    dest_map = copy.deepcopy(DENSITY_MAP)

    if param.SIMULATION_MODE == 1:
        # 기존 알고리즘 실행 (고정 VNF 배치 + 다익스트라 기반 라우팅)
        start_time = time.time()
        delivered = simulate(simulation_time= 2 * SIMULATION_TIME,
                             source_density_map=source_map,
                             dest_density_map=dest_map)
        finish_time = time.time()
        print(f"[모드 1] 시뮬레이션 실행 시간: {finish_time - start_time:.2f} sec, 전달된 패킷 수: {len(delivered)}")
    elif param.SIMULATION_MODE == 2:
        # "On-the-fly VNF 설치" 알고리즘:
        # VNF_INSTALLATION_TIMES의 각 값에 대해 실험 실행
        for install_time in param.VNF_INSTALLATION_TIMES:
            # 현재 사용할 설치 시간을 업데이트 (전역 변수처럼 사용)
            param.CURRENT_VNF_INSTALLATION_TIME = install_time
            print(f"\n[모드 2] VNF 설치 시간 {install_time} ms 로 시뮬레이션 시작")
            # 실험마다 새 지도(깊은 복사)를 생성
            source_map = copy.deepcopy(DENSITY_MAP)
            dest_map = copy.deepcopy(DENSITY_MAP)
            start_time = time.time()
            delivered = simulate(simulation_time= 2 * SIMULATION_TIME,
                                 source_density_map=source_map,
                                 dest_density_map=dest_map)
            finish_time = time.time()
            print(f"VNF 설치 시간 {install_time} ms, 시뮬레이션 실행 시간: {finish_time - start_time:.2f} sec, 전달된 패킷 수: {len(delivered)}")
