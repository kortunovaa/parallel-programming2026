import subprocess
import time
import os

sizes = [200, 400, 800, 1200, 1600, 2000]

# Конфигурации блоков (вместо количества процессов MPI)
block_configs = [
    (8, 8),    # 64 потока
    (16, 16),  # 256 потоков
    (32, 32),  # 1024 потока (максимум)
    (16, 32),  # 512 потоков (прямоугольный)
    (32, 16),  # 512 потоков (прямоугольный)
]

repeats = 3

exe_name = "lab3_cuda.exe" if os.name == 'nt' else "./lab3_cuda"

# Создаем CSV
with open("experiments_cuda.csv", "w") as f:
    f.write("Размер матрицы,Конфигурация блоков,Время(мс),GFLOPS,Верно\n")

total_tests = len(sizes) * len(block_configs) * repeats
current_test = 0

print("=" * 70)
print("CUDA MATRIX MULTIPLICATION EXPERIMENTS")
print("=" * 70)

for size in sizes:
    print(f"\n=== Matrix size: {size}x{size} ===")
    
    for bx, by in block_configs:
        print(f"\n--- Block config: {bx}x{by} ({bx*by} threads/block) ---")
        
        for rep in range(1, repeats + 1):
            current_test += 1
            print(f"  Run {rep}/3 [{current_test}/{total_tests}]")
            
            cmd = [exe_name, str(size), str(bx), str(by), str(rep)]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.stdout:
                    # Выводим только последние строки
                    lines = result.stdout.strip().split('\n')
                    for line in lines[-5:]:
                        print(f"    {line}")
            except Exception as e:
                print(f"    Error: {e}")
            
            time.sleep(0.5)

print("\n" + "=" * 70)
print("EXPERIMENTS COMPLETED!")
print("Results saved to experiments_cuda.csv")
print("=" * 70)
