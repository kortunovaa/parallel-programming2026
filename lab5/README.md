# Лабораторная работа №5
### Параллельное умножение матриц с использованием MPI на кластере

## Задание

Модифицировать программу из лабораторной работы №1 для параллельного выполнения с использованием технологии MPI и запустить на кластере.

**Требования:**
- Провести серию экспериментов с разным количеством процессов (1, 2, 4, 8)
- Провести эксперименты с разными размерами матриц (200, 400, 800, 1200)
- Исследовать влияние количества процессов на производительность
- Выполнить автоматизированную верификацию результатов

## Реализация

В программу из лабораторной работы №1 были добавлены MPI функции для организации межпроцессного взаимодействия с распределением строк матрицы A между процессами:

```cpp
#include <mpi.h>     
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cstdlib>

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    
    // Только процесс 0 читает матрицы из файлов
    if (rank == 0) {
        MatrixA = Read_Matrix_From_File(matrixA_file);
        MatrixB = Read_Matrix_From_File(matrixB_file);
    }
    
    // Рассылка матрицы B всем процессам
    MPI_Bcast(flat_B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Распределение строк матрицы A между процессами
    MPI_Scatterv(flat_A.data(), send_counts.data(), displs.data(), MPI_DOUBLE,
                 local_A.data(), local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Каждый процесс вычисляет свою часть результата
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += local_A[i * n + k] * flat_B[k * n + j];
            }
            local_result[i * n + j] = sum;
        }
    }
    
    // Сбор результатов от всех процессов
    MPI_Gatherv(local_result.data(), local_rows * n, MPI_DOUBLE,
                flat_result.data(), send_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    MPI_Finalize();
    return 0;
}
```

## Методика проведения экспериментов
Среда выполнения:
Кластер: СК "Сергей Королев" (Самарский университет)
MPI: Intel MPI
Компилятор: mpicxx с флагом -std=c++11

# Порядок проведения экспериментов:
Генерация матриц:
```cpp
python3 generate_matrix.py
```
Скрипт создает матрицы размеров 200, 400, 800, 1200

# Компиляция программы:
```cpp
module load intel/mpi4
mpicxx -std=c++11 Source.cpp -o laba2_mpi
```
# Создание SLURM скрипта для запуска:

```cpp
#!/bin/bash
#SBATCH --job-name=lab2
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=batch
```
# Запуск экспериментов:
```cpp
sbatch run_complete.slurm
```

<img width="791" height="546" alt="image" src="https://github.com/user-attachments/assets/6c0a2d5f-314a-44a9-bca8-cd91cf84285f" />

# Параметры экспериментов:

- Размеры матриц: 200, 400, 800, 1200
- Количество процессов: 1, 2, 4, 8
- Количество повторений: 3 для каждой комбинации

## Верификация
Все результаты экспериментов были автоматически проверены с помощью Python-скрипта paral.py:
Для каждой комбинации (размер × количество процессов) выполнена проверка корректности умножения
Сравнение с эталонным умножением через np.dot() из библиотеки NumPy
Проверка осуществлялась для всех 48 тестов (4 размера × 4 варианта процессов × 3 повторения)
```cpp
Multiplication finished in 178 ms
Result is CORRECT!
Experiment 1 completed
```

## сводная таблица результатов

### Эффективность параллелизации
| Размер матрицы | 2 процесса | 4 процесса | 8 процессов |
|:--------------:|:--------:|:--------:|:---------:|
| 200 | 98% | 94% | 78% |
| 400 | 98% | 97% | 83% |
| 800 | 95% | 87% | 82% |

<img width="754" height="165" alt="image" src="https://github.com/user-attachments/assets/36bbc128-e489-4d38-89a2-2891a6ad755c" />


# Вывод
**1. Программа из лабораторной работы №1 успешно модифицирована для параллельной работы с использованием технологии MPI на кластере.**

**2.Корректность результатов подтверждена автоматической верификацией**

**3. Анализ производительности показал:**
1)Хорошее ускорение для всех размеров матриц
2) Максимальное ускорение: 6.63x на 8 процессах для матрицы 400×400
3) Эффективность параллелизации: от 78% до 98%

**4. Полученные результаты подтверждают эффективность MPI-реализации для задач умножения матриц на кластерных системах.**
