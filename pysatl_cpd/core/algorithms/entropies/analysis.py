import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from approximate_entropy import ApproximateEntropyAlgorithm
from bubble_entropy import BubbleEntropyAlgorithm
from sample_entropy import SampleEntropyAlgorithm
from shannon_entropy import ShannonEntropyAlgorithm
from slope_entropy import SlopeEntropyAlgorithm

tickers = {"KO": "Coca-Cola (KO)", "^GSPC": "S&P 500 (^GSPC)"}

t = np.linspace(0, 10, 500)
ecg_series = np.sin(2 * np.pi * 1.5 * t) + 0.1 * np.random.randn(len(t))
health_series = 100 + 5 * np.sin(0.5 * t) + np.random.randn(len(t))
router_series = np.random.choice([0, 1], size=500, p=[0.3, 0.7])

time_series_list = {}
for symbol, full_name in tickers.items():
    stock_data = yf.download(symbol, start="2023-01-01", end="2025-07-01")
    stock_series = stock_data["Close"].ffill().values
    time_series_list[full_name] = stock_series

time_series_list["ECG"] = ecg_series
time_series_list["Health"] = health_series
time_series_list["Router Signal"] = router_series

algorithms = [
    BubbleEntropyAlgorithm(window_size=30, embedding_dimension=3, threshold=0.05),
    ApproximateEntropyAlgorithm(window_size=30, m=2, r=0.2, threshold=0.05),
    SampleEntropyAlgorithm(window_size=30, m=2, threshold=0.05),
    ShannonEntropyAlgorithm(window_size=30, threshold=0.05),
    SlopeEntropyAlgorithm(window_size=20, threshold=0.05),
]


def analyze_series(name, series):
    print(f"\n=== Анализ серии: {name} ===")
    for algo in algorithms:
        if hasattr(algo, "reset"):
            algo.reset()

        change_points = []
        for i, value in enumerate(series):
            if algo.detect(value):
                cp = algo.localize(value)
                if cp is not None and cp < len(series):
                    change_points.append(cp)

        print(f"{algo.__class__.__name__}: Обнаруженные change points: {change_points}")
        plot_series_with_cps(name, series, change_points, algo.__class__.__name__)


def plot_series_with_cps(series_name, series, cps, algo_name):
    plt.figure(figsize=(12, 3))
    plt.plot(series, label=f"{series_name}")
    if cps:
        plt.scatter(cps, series[cps], color="red", marker="x", s=50, label="Change points")
    plt.title(f"{series_name} - {algo_name}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


for name, series_data in time_series_list.items():
    analyze_series(name, series_data)
