# main.py
from datetime import datetime
import param
import function

if __name__ == '__main__':
    start_time = datetime.now()

    if param.SIMULATION_MODE == 0:
        print(f"데이터셋 생성 모드: {param.SIMULATION_TIME} 동안 패킷 생성만 수행합니다.")
        dataset = function.generate_packet_dataset(param.SIMULATION_TIME)
        print(f"생성된 패킷 수: {len(dataset)}")
        # 생성된 데이터셋을 CSV 파일로 저장
        function.log_packets_to_csv(dataset, tick=0, start_time=start_time,
                                    file_path=f"./csv/packets_dataset_{param.SIMULATION_TIME}_{start_time.strftime('%m%d_%H%M')}.csv")

    if param.SIMULATION_MODE == 1:
        delivered, grid = function.simulate(simulation_time=param.SIMULATION_TIME,
                                            start_time=start_time,
                                            install_time=0)
        finish_time = datetime.now()
        print(f"[모드 1] 실행 시간: {finish_time - start_time}, 전달된 패킷 수: {len(delivered)}")
        if param.CSV:
            function.log_satellites_to_csv(grid, start_time,
                                           file_path=f"./csv/satellites_log_mode1_{param.SIMULATION_TIME}_{start_time.strftime('%m%d_%H%M')}.csv")

    elif param.SIMULATION_MODE == 2:
        for install_time in param.VNF_INSTALLATION_TIMES:
            param.CURRENT_VNF_INSTALLATION_TIME = install_time
            print(f"\n[모드 2] VNF 설치 시간 {install_time} ms 로 시뮬레이션 시작")
            delivered, grid = function.simulate(simulation_time=param.SIMULATION_TIME,
                                                start_time=start_time,
                                                install_time=install_time)
            finish_time = datetime.now()
            print(f"VNF 설치 시간 {install_time} ms, 실행 시간: {finish_time - start_time}, 전달된 패킷 수: {len(delivered)}")
            if param.CSV:
                function.log_satellites_to_csv(grid, start_time,
                                               file_path=f"./csv/satellites_log_mode2[{install_time}ms]_{param.SIMULATION_TIME}_{start_time.strftime('%m%d_%H%M')}.csv")
