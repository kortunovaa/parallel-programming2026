import subprocess
import time

sizes = [200, 400, 800, 1200, 1600, 2000]
processes = [1, 2, 4, 8]  # Количество процессов MPI
repeats = 3

# Определяем команду для запуска MPI
# Для Windows с MS-MPI
mpi_command = "mpiexec"

# Для Linux/OpenMPI:
# mpi_command = "mpirun"

with open("experiments_mpi.csv", "w") as f:
    f.write("Размер матрицы,Процессы,Время(мс),Верно\n")

total_tests = len(sizes) * len(processes) * repeats
current_test = 0

for size in sizes:
    for proc in processes:
        for rep in range(1, repeats + 1):
            current_test += 1
            print(f"\n--- Тест {current_test}/{total_tests} ---")
            print(f"Размер: {size}, Процессы MPI: {proc}, Повтор: {rep}")
            
            # Запускаем MPI программу
            # Для Windows:
            cmd = [mpi_command, "-n", str(proc), "laba2_mpi.exe", str(size), str(proc), str(rep)]
            
            # Для Linux:
            # cmd = [mpi_command, "-np", str(proc), "./laba2_mpi", str(size), str(proc), str(rep)]
            
            result = subprocess.run(cmd)
            
            # Небольшая пауза между запусками
            time.sleep(1)

print("\nВсе эксперименты завершены!")
print("Результаты сохранены в experiments_mpi.csv")
