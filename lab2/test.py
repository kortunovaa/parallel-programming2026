import subprocess
import time

sizes = [200, 400, 800, 1200, 1600, 2000]
threads = [1, 2, 4, 8]
repeats = 3

with open("experiments.csv", "w") as f:
    f.write("Размер матрицы,Потоки,Время(мс),Верно\n")

total_tests = len(sizes) * len(threads) * repeats
current_test = 0

for size in sizes:
    for thread in threads:
        for rep in range(1, repeats + 1):
            current_test += 1
            print(f"\nТест {current_test}/{total_tests} ")
            print(f"Размер: {size}, Потоки: {thread}, Повтор: {rep}")
            
            result = subprocess.run(["./laba2.exe", str(size), str(thread), str(rep)])
            
            time.sleep(1)

print("\nВсе эксперименты завершены!")
print("Результаты сохранены в experiments.csv")
