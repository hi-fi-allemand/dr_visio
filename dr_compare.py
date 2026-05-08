"""
DR Compare — визуализация динамического диапазона двух WAV файлов
с одинаковым масштабом по оси дБ.

Использование:
    python dr_compare.py file1.wav file2.wav
    python dr_compare.py file1.wav file2.wav --window 100   # окно RMS в мс
    python dr_compare.py file1.wav file2.wav --out result.png

Зависимости: pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.io import wavfile
import sys
import os
import argparse

SILENCE_FLOOR = 1e-9


def rms_to_db(rms):
    return 20.0 * np.log10(max(float(rms), SILENCE_FLOOR))


def read_wav(filepath):
    """Читает WAV любой битности, возвращает (sr, float64 [-1..1])."""
    import wave as wavemodule
    # Определяем реальную битность через стандартный модуль wave
    with wavemodule.open(filepath, "rb") as wf:
        sampwidth = wf.getsampwidth()  # байт на семпл: 2=16bit, 3=24bit, 4=32bit

    sr, data = wavfile.read(filepath)
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Нормализация с учётом реальной битности
    if sampwidth == 1:          # uint8
        data = (data.astype(np.float64) - 128.0) / 128.0
    elif sampwidth == 2:        # int16
        data = data.astype(np.float64) / 32768.0
    elif sampwidth == 3:        # 24-bit: scipy читает как int32 со сдвигом влево на 8 бит
        data = data.astype(np.float64) / 2147483648.0
    elif sampwidth == 4:        # int32
        data = data.astype(np.float64) / 2147483648.0
    else:
        data = data.astype(np.float64)

    return sr, data



def compute_rms_curve(data, sr, window_ms=100):
    """RMS-огибающая с заданным окном в мс."""
    win  = max(1, int(sr * window_ms / 1000))
    hop  = max(1, win // 4)
    times, rms_db = [], []
    for start in range(0, len(data) - win, hop):
        chunk = data[start : start + win]
        rms   = np.sqrt(np.mean(chunk ** 2))
        times.append((start + win / 2) / sr)
        rms_db.append(rms_to_db(rms))
    return np.array(times), np.array(rms_db)


def compute_peak_curve(data, sr, window_ms=100):
    """Пиковая огибающая."""
    win  = max(1, int(sr * window_ms / 1000))
    hop  = max(1, win // 4)
    times, peaks = [], []
    for start in range(0, len(data) - win, hop):
        chunk = data[start : start + win]
        peak  = np.max(np.abs(chunk))
        times.append((start + win / 2) / sr)
        peaks.append(rms_to_db(peak))
    return np.array(times), np.array(peaks)


def compute_dr(data, sr):
    """
    DR по стандарту Pleasurize Music Foundation:
    делим на блоки 3 сек, считаем Peak и RMS каждого,
    DR = среднее(top 20% пиков) - среднее(top 20% RMS)
    """
    block = sr * 3
    peaks_db, rms_db = [], []
    for start in range(0, len(data) - block, block):
        chunk = data[start : start + block]
        peaks_db.append(rms_to_db(np.max(np.abs(chunk))))
        rms_db.append(rms_to_db(np.sqrt(np.mean(chunk ** 2))))

    if not peaks_db:
        return 0.0

    n_top = max(1, len(peaks_db) // 5)
    top_peaks = sorted(peaks_db, reverse=True)[:n_top]
    top_rms   = sorted(rms_db,   reverse=True)[:n_top]
    return np.mean(top_peaks) - np.mean(top_rms)


def plot_comparison(files, window_ms=100, out_path=None):
    colors_rms  = ["#cc0000", "#0055aa"]   # красный / синий — RMS
    colors_peak = ["#888888", "#aaaaaa"]    # светлее — пик
    colors_fill = ["#cc0000", "#0055aa"]

    fig, axes = plt.subplots(len(files), 1,
                             figsize=(14, 4 * len(files)),
                             facecolor="white")
    if len(files) == 1:
        axes = [axes]

    # Собираем все dB значения для единого масштаба
    all_rms_vals = []
    datasets = []
    for path in files:
        sr, data = read_wav(path)
        t_rms,  rms  = compute_rms_curve(data, sr, window_ms)
        t_peak, peak = compute_peak_curve(data, sr, window_ms)
        dr = compute_dr(data, sr)
        duration = len(data) / sr
        datasets.append((path, sr, t_rms, rms, t_peak, peak, dr, duration))
        all_rms_vals.extend(rms)
        all_rms_vals.extend(peak)

    # Единый диапазон оси Y — используем перцентили чтобы выбросы не сжимали график
    arr_all = np.array(all_rms_vals)
    arr_all = arr_all[np.isfinite(arr_all)]
    y_max = min(3.0,  np.percentile(arr_all, 99) + 3)
    y_min = max(-96,  np.percentile(arr_all, 2)  - 6)

    for idx, (path, sr, t_rms, rms, t_peak, peak, dr, duration) in enumerate(datasets):
        ax  = axes[idx]
        col_rms  = colors_rms[idx % len(colors_rms)]
        col_peak = colors_peak[idx % len(colors_peak)]
        col_fill = colors_fill[idx % len(colors_fill)]

        ax.set_facecolor("white")

        # Заливка между peak и rms — это и есть визуальный DR
        ax.fill_between(t_peak, peak, rms,
                        alpha=0.35, color=col_fill, label="Динамический диапазон")

        # Пиковая огибающая
        ax.plot(t_peak, peak,
                color=col_peak, linewidth=0.8, alpha=0.7, label="Peak")

        # RMS-огибающая
        ax.plot(t_rms, rms,
                color=col_rms, linewidth=1.2, label="RMS")

        # 0 dBFS линия
        ax.axhline(0, color="#666666", linewidth=0.5, linestyle="--", alpha=0.4)

        # Заголовок с DR
        name = os.path.basename(path)
        ax.set_title(f"{name}   |   DR ≈ {dr:.0f}",
                     color="black", fontsize=13, pad=8, fontweight="bold")

        # Единый масштаб Y
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, duration)

        # Оси
        ax.set_ylabel("дБFS", color="black", fontsize=10)
        ax.set_xlabel("Время (сек)", color="black", fontsize=10)
        ax.tick_params(colors="#aaaaaa")
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.grid(True, which="major", color="#dddddd", linewidth=0.5)
        ax.grid(True, which="minor", color="#eeeeee", linewidth=0.3)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

        # Легенда
        ax.legend(loc="lower right", fontsize=9,
                  facecolor="white", edgecolor="black",
                  labelcolor="black")

        # Аннотация DR
        dr_text = f"DR = {dr:.0f}\n{'█' * min(int(dr), 20)}"
        ax.text(0.01, 0.05, dr_text,
                transform=ax.transAxes,
                color=col_rms, fontsize=10, fontfamily="monospace",
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="black", alpha=0.8))

    fig.suptitle("Сравнение динамического диапазона  (единый масштаб)",
                 color="black", fontsize=14, y=1.01, fontweight="bold")

    plt.tight_layout()

    if out_path is None:
        out_path = "dr_comparison.png"

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Картинка сохранена: {out_path}")
    return out_path


# ─── ТОЧКА ВХОДА ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="WAV файлы (1 или 2)")
    parser.add_argument("--window", type=int, default=100,
                        help="Окно RMS в мс (default: 100)")
    parser.add_argument("--out", default=None, help="Путь к выходному PNG")
    args = parser.parse_args()

    for f in args.files:
        if not os.path.isfile(f):
            print(f"Файл не найден: {f}")
            sys.exit(1)

    plot_comparison(args.files, window_ms=args.window, out_path=args.out)
