import csv
from collections import defaultdict

# Читаем результаты
results = defaultdict(lambda: defaultdict(list))

try:
    with open("experiments_cuda.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 5:
                size = int(row[0])
                config = row[1]
                time_ms = float(row[2])
                gflops = float(row[3])
                correct = int(row[4])
                
                if correct == 1:
                    results[size][config].append(time_ms)
except FileNotFoundError:
    print("File experiments_cuda.csv not found")
    exit(1)

# Вычисляем средние
averages = defaultdict(dict)
all_configs = set()

for size, configs in results.items():
    for config, times in configs.items():
        if times:
            averages[size][config] = sum(times) / len(times)
            all_configs.add(config)

all_configs = sorted(list(all_configs))

# Создаем отчет
with open("results_table_cuda.txt", "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("РЕЗУЛЬТАТЫ ЛАБОРАТОРНОЙ РАБОТЫ (CUDA)\n")
    f.write("Умножение матриц с различными конфигурациями блоков\n")
    f.write("=" * 90 + "\n\n")
    
    # Таблица
    header = f"{'Размер':<12}"
    for config in all_configs:
        header += f"{config + ' (ms)':<15}"
    header += f"{'Ускорение':<12}"
    f.write(header + "\n")
    f.write("-" * 90 + "\n")
    
    sizes_sorted = sorted(averages.keys())
    
    for size in sizes_sorted:
        best_time = min(averages[size].values()) if averages[size] else 0
        
        row = f"{size:<12}"
        for config in all_configs:
            if config in averages[size]:
                row += f"{averages[size][config]:<15.2f}"
            else:
                row += f"{'N/A':<15}"
        
        if "8x8" in averages[size] and best_time > 0:
            speedup = averages[size]["8x8"] / best_time
            row += f"{speedup:<12.2f}x"
        else:
            row += f"{'N/A':<12}"
        
        f.write(row + "\n")
    
    f.write("=" * 90 + "\n\n")
    
    # Анализ
    f.write("АНАЛИЗ РЕЗУЛЬТАТОВ\n")
    f.write("-" * 90 + "\n")
    f.write("\n1. Влияние размера блоков:\n")
    f.write("   - 8x8:   Много накладных расходов, низкая загрузка GPU\n")
    f.write("   - 16x16: Хороший баланс, рекомендуется для средних матриц\n")
    f.write("   - 32x32: Максимальная загрузка, оптимально для больших матриц\n")
    
    f.write("\n2. Рекомендации по выбору конфигурации:\n")
    for size in sizes_sorted:
        if size in averages:
            best = min(averages[size].items(), key=lambda x: x[1])
            f.write(f"   - {size}x{size}: лучше всего {best[0]} ({best[1]:.2f} мс)\n")

print("\nReport saved to results_table_cuda.txt")
