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


def plot_histograms_for_column(csv_files, column, bins=50, save_path=None):
    plt.figure(figsize=(8, 6))

    # 전체 데이터 범위를 이용해 동일한 bin edge를 생성
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if column in df.columns:
            all_data.extend(df[column].dropna().tolist())
    if not all_data:
        print(f"No data found for column: {column}")
        return

    bin_edges = np.linspace(min(all_data), max(all_data), bins + 1)

    # 각 파일별 히스토그램을 step 형태로 그리기 (alpha는 자동 적용)
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if column not in df.columns:
            continue
        data = df[column].dropna()
        legend = extract_legend(csv_file)
        plt.hist(data, bins=bin_edges, histtype='step', linewidth=2, label=legend)

    plt.title(f"{column} Histogram Comparison")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_cdfs_for_column(csv_files, column, save_path=None):
    plt.figure(figsize=(8, 6))

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if column not in df.columns:
            continue
        data = np.sort(df[column].dropna())
        cdf = np.arange(1, len(data) + 1) / len(data)
        legend = extract_legend(csv_file)
        plt.plot(data, cdf, linewidth=2, label=legend)

    plt.title(f"{column} CDF Comparison")
    plt.xlabel(column)
    plt.ylabel("CDF")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    # CSV 파일 경로: ./csv/ 폴더 내 packets_log_mode로 시작하는 모든 CSV 파일
    csv_files = glob.glob("./csv/packets_log_mode*.csv")
    if not csv_files:
        print("CSV 파일을 찾을 수 없습니다.")
        return

    # 분석할 지연 시간 컬럼들
    delay_columns = [
        "Propagation_Delay(ms)",
        "Queueing_Delay(ms)",
        "Processing_Delay(ms)",
        "Transmission_Delay(ms)",
        "Total_Delay(ms)"
    ]

    # 저장 폴더 생성
    os.makedirs("./graph", exist_ok=True)

    for col in delay_columns:
        # 히스토그램 플롯
        hist_save_path = os.path.join("./graph",
                                      f"Histogram_{col.replace('(', '').replace(')', '').replace(' ', '_')}.png")
        plot_histograms_for_column(csv_files, col, bins=50, save_path=hist_save_path)

        # CDF 플롯
        cdf_save_path = os.path.join("./graph", f"CDF_{col.replace('(', '').replace(')', '').replace(' ', '_')}.png")
        plot_cdfs_for_column(csv_files, col, save_path=cdf_save_path)


if __name__ == '__main__':
    main()
