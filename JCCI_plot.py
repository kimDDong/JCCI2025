import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 폰트 설정 (한글 사용 시)
plt.rcParams["font.family"] = "Malgun Gothic"  # Windows 예시


def extract_legend(csv_file):
    """
    파일명에서 레전드(예: mode1, mode2[1ms])를 추출합니다.
    파일명 구조: packets_log_mode1_60_0226_0907.csv 또는 packets_log_mode2[1ms]_60_0225_2202.csv
    """
    base = os.path.basename(csv_file)
    name_no_ext = os.path.splitext(base)[0]
    parts = name_no_ext.split('_')
    return parts[2] if len(parts) > 2 else "Unknown"

def plot_cdfs_for_column(csv_files, column, save_path=None):
    plt.figure(figsize=(8, 6))

    tmp = 1
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if column not in df.columns:
            continue
        data = np.sort(df[column].dropna())
        cdf = np.arange(1, len(data) + 1) / len(data)
        legend = extract_legend(csv_file)
        if legend == 'mode1':
            legend = 'VNF 사전 배치'
        else:
            legend = 'VNF 실시간 배치 [' + str(tmp) + 'ms]'
            tmp += 1
        plt.plot(data, cdf, linewidth=2, label=legend)
    if column == 'Total_Delay(ms)':
        # plt.title("종단간 지연시간 확률밀도함수 비교")
        plt.xlabel("종단간 지연시간 [ms]", fontsize=16)
        plt.ylabel("확률밀도함수", fontsize=16)
        plt.legend()
        if save_path:
            plt.savefig(save_path)
    elif column == 'Propagation_Delay(ms)':
        plt.xlabel("전파 지연시간 [ms]", fontsize=16)
        plt.ylabel("확률밀도함수", fontsize=16)
        plt.legend()
        if save_path:
            plt.savefig(save_path)


def plot_subplots_vertical(csv_files, col1, col2, save_path=None):
    # 각 컬럼 데이터의 전체 최소/최대값 계산
    col1_x_min, col1_x_max = np.inf, -np.inf
    col1_y_min, col1_y_max = np.inf, -np.inf
    col2_x_min, col2_x_max = np.inf, -np.inf
    col2_y_min, col2_y_max = np.inf, -np.inf

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if col1 in df.columns:
            data = np.sort(df[col1].dropna())
            if len(data) > 0:
                cdf = np.arange(1, len(data) + 1) / len(data)
                col1_x_min = min(col1_x_min, data[0])
                col1_x_max = max(col1_x_max, data[-1])
                col1_y_min = min(col1_y_min, cdf[0])
                col1_y_max = max(col1_y_max, cdf[-1])
        if col2 in df.columns:
            data = np.sort(df[col2].dropna())
            if len(data) > 0:
                cdf = np.arange(1, len(data) + 1) / len(data)
                col2_x_min = min(col2_x_min, data[0])
                col2_x_max = max(col2_x_max, data[-1])
                col2_y_min = min(col2_y_min, cdf[0])
                col2_y_max = max(col2_y_max, cdf[-1])

    # 원점(0)을 포함하도록 최소값을 조정 (만약 데이터 최소값이 0보다 크다면)
    col1_x_min = min(0, col1_x_min)
    col1_y_min = min(0, col1_y_min)
    col2_x_min = min(0, col2_x_min)
    col2_y_min = min(0, col2_y_min)

    # 세로 2개의 서브플롯 생성
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    # 스타일 설정
    pre_batch_style = {'color': 'black', 'linestyle': '-'}  # 사전 배치는 검은 실선
    mode2_colors = ['red', 'orange', 'gold', 'green', 'blue']  # 실시간 배치는 점선 + 색상
    mode2_counter = 0
    mode2_counter2 = -1

    # 각 CSV 파일에 대해 그래프 그리기
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        legend_extracted = extract_legend(csv_file)  # 예: 'mode1' 또는 'mode2[1ms]'
        if 'mode1' in legend_extracted:
            style = pre_batch_style
            legend_name = '사전 배치'
        else:
            style = {
                'color': mode2_colors[mode2_counter % len(mode2_colors)],
                'linestyle': '--'
            }
            mode2_counter += 1
            mode2_counter2 += 2
            legend_name = f"실시간 배치 [{mode2_counter2}ms]"

        # 첫 번째 서브플롯: col1 (전파 지연시간)
        if col1 in df.columns:
            data1 = np.sort(df[col1].dropna())
            cdf1 = np.arange(1, len(data1) + 1) / len(data1)
            axes[0].plot(data1, cdf1, linewidth=2,
                         color=style['color'],
                         linestyle=style['linestyle'],
                         label=legend_name)

        # 두 번째 서브플롯: col2 (종단간 지연시간)
        if col2 in df.columns:
            data2 = np.sort(df[col2].dropna())
            cdf2 = np.arange(1, len(data2) + 1) / len(data2)
            axes[1].plot(data2, cdf2, linewidth=2,
                         color=style['color'],
                         linestyle=style['linestyle'],
                         label=legend_name)

    # 서브플롯 (a): 전파 지연시간
    axes[0].set_xlabel("전파 지연시간 [ms]", fontsize=14)
    axes[0].set_ylabel("누적 확률", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].set_xlim(col1_x_min, col1_x_max)
    axes[0].set_ylim(col1_y_min, col1_y_max)

    # 서브플롯 (b): 종단간 지연시간
    axes[1].set_xlabel("종단간 지연시간 [ms]", fontsize=14)
    axes[1].set_ylabel("누적 확률", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].set_xlim(col2_x_min, col2_x_max)
    axes[1].set_ylim(col2_y_min, col2_y_max)

    # 서브플롯 구분 라벨 (a), (b)
    axes[0].text(0.5, -0.25, '(a)', transform=axes[0].transAxes,
                 fontsize=16, ha='center', va='top')
    axes[1].text(0.5, -0.25, '(b)', transform=axes[1].transAxes,
                 fontsize=16, ha='center', va='top')

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    csv_files = glob.glob("./csv/새 폴더 (4)/packets_log_mode*.csv")
    if not csv_files:
        print("CSV 파일을 찾을 수 없습니다.")
        return

    # 저장 폴더 생성
    os.makedirs("./graph", exist_ok=True)
    save_path = os.path.join("./graph/JCCI2025", "CDF_Subplots_Vertical_Propagation_Total.png")

    # 두 컬럼에 대해 서브플롯 생성: (a) Propagation_Delay(ms), (b) Total_Delay(ms)
    plot_subplots_vertical(csv_files, "Propagation_Delay(ms)", "Total_Delay(ms)", save_path=save_path)


if __name__ == '__main__':
    main()
