import pandas as pd
import matplotlib.pyplot as plt

def plot_csv(csv_path):
    
    df = pd.read_csv(csv_path)

    # Chuyển cột thời gian về kiểu datetime
    df['ds'] = pd.to_datetime(df['ds'])

    # Sắp xếp theo thời gian nếu chưa được sắp xếp
    df = df.sort_values('ds')

    # Vẽ biểu đồ
    plt.figure(figsize=(36, 6))

    # Vẽ giá trị thực tế
    if 'y' in df.columns:
        plt.plot(df['ds'], df['y'], label='Giá trị thực tế')

    # Vẽ các cột dự báo nếu tồn tại
    if "TimesNet" in df.columns:
        plt.plot(df['ds'], df["TimesNet"], label='Dự báo TimesNet', linestyle='--')

    if "TimesNet-median" in df.columns:
        plt.plot(df['ds'], df["TimesNet-median"], label='Dự báo median', linestyle='--')
    if "NHITS" in df.columns:
        plt.plot(df['ds'], df["NHITS"], label='Dự báo NHITS', linestyle='--')
    if "NHITS-median" in df.columns:
        plt.plot(df['ds'], df["NHITS-median"], label='Dự báo NHITS median', linestyle='--')
    if "PatchTST" in df.columns:
        plt.plot(df['ds'], df["PatchTST"], label='Dự báo PatchTST', linestyle='--')
    if "PatchTST-median" in df.columns:
        plt.plot(df['ds'], df["PatchTST-median"], label='Dự báo PatchTST median', linestyle='--')
    # if 'TimesNet-lo-100' in df.columns and 'TimesNet-hi-100' in df.columns:
    #     plt.fill_between(df['ds'], df['TimesNet-lo-100'], df['TimesNet-hi-100'],
    #                      color='gray', alpha=0.3, label='Khoảng dự báo 100%')

    # Cài đặt biểu đồ
    plt.title('Biểu đồ giá trị thực tế và dự báo')
    plt.xlabel('Thời gian')
    plt.ylabel('Giá trị')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Lưu hình trước khi hiển thị
    plt.savefig('results/forecast_plot_nhits_7.7M.png')
    plt.show()

def main():
    csv_path = 'results/prediction_nhits_7.71.csv'
    try:
        plot_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {csv_path}")
    except Exception as e:
        print(f"❌ Lỗi khi xử lý file: {e}")

if __name__ == "__main__":
    main()
