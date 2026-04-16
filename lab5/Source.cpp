#include <mpi.h>     
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cstdlib>  

using namespace std;

vector<vector<double>> Read_Matrix_From_File(const string& filename) {
    ifstream file(filename.c_str());  // используем c_str() для C++98 совместимости

    if (!file.is_open()) {
        cout << "Error opening file: " << filename << "\n";
        exit(1);
    }

    int size;
    file >> size;

    vector<vector<double> > matrix(size, vector<double>(size));  // пробел между >>

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            file >> matrix[i][j];
        }
    }

    file.close();
    return matrix;
}

void Write_Matrix_To_File(const string& filename, const vector<vector<double> >& matrix) {
    ofstream file(filename.c_str());  // используем c_str()

    if (!file.is_open()) {
        cout << "Error opening file: " << filename << "\n";
        return;
    }

    int n = matrix.size();
    file << n << "\n";

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << matrix[i][j];
            if (j < n - 1) file << " ";
        }
        file << "\n";
    }

    file.close();
}

void Write_Experiment_Results(const string& filename,
    int matrix_size,
    int num_processes,
    chrono::milliseconds duration,
    bool correctly) {
    ofstream file(filename.c_str(), ios::app);  // используем c_str()

    if (!file.is_open()) {
        cout << "Error opening file: " << filename << "\n";
        return;
    }

    file << matrix_size << "," << num_processes << ","
        << duration.count() << "," << (correctly ? "1" : "0") << "\n";

    file.close();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    if (rank == 0) {
        cout << "=== PARALLEL VERSION (with Scatterv/Gatherv) ===\n";
    }

    if (argc != 4) {
        if (rank == 0) {
            cout << "Usage: mpiexec -n <num_processes> " << argv[0]
                << " <matrix_size> <num_processes> <experiment_number>\n";
            cout << "Example: mpiexec -n 4 " << argv[0] << " 200 4 1\n";
        }
        MPI_Finalize();
        return 1;
    }

    int matrix_size = atoi(argv[1]);
    int num_processes_arg = atoi(argv[2]);
    int exp_num = atoi(argv[3]);

    // Проверка соответствия количества процессов
    if (num_processes != num_processes_arg && rank == 0) {
        cout << "Warning: Number of MPI processes (" << num_processes 
             << ") doesn't match argument (" << num_processes_arg << ")\n";
    }

    string matrixA_file = "MatrixA_" + to_string(matrix_size) + "_" + to_string(exp_num) + ".txt";
    string matrixB_file = "MatrixB_" + to_string(matrix_size) + "_" + to_string(exp_num) + ".txt";
    string result_file = "result_" + to_string(matrix_size) + "_" + to_string(exp_num) + ".txt";

    int n = matrix_size;

    // Все процессы выделяют память для матриц
    vector<double> flat_A;
    vector<double> flat_B(n * n);
    vector<double> flat_result;

    // Только процесс 0 читает матрицы из файлов
    if (rank == 0) {
        vector<vector<double> > MatrixA = Read_Matrix_From_File(matrixA_file);
        vector<vector<double> > MatrixB = Read_Matrix_From_File(matrixB_file);

        cout << "\n=== MPI Experiment " << exp_num << " ===\n";
        cout << "Matrix size: " << matrix_size << "x" << matrix_size << "\n";
        cout << "Number of processes: " << num_processes << "\n";

        // Преобразуем A в плоский массив
        flat_A.resize(n * n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                flat_A[i * n + j] = MatrixA[i][j];
            }
        }

        // Преобразуем B в плоский массив
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                flat_B[i * n + j] = MatrixB[i][j];
            }
        }
    }

    // Синхронизация перед замером времени
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = chrono::high_resolution_clock::now();

    // РАССЫЛКА МАТРИЦЫ B ВСЕМ ПРОЦЕССАМ
    MPI_Bcast(flat_B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Распределение строк матрицы A
    int rows_per_process = n / num_processes;
    int remainder = n % num_processes;

    vector<int> send_counts(num_processes);
    vector<int> displs(num_processes);

    int current_row = 0;
    for (int i = 0; i < num_processes; i++) {
        send_counts[i] = (rows_per_process + (i < remainder ? 1 : 0)) * n;
        displs[i] = current_row * n;
        current_row += rows_per_process + (i < remainder ? 1 : 0);
    }

    int local_rows = send_counts[rank] / n;
    vector<double> local_A(local_rows * n);

    // Рассылка строк матрицы A
    MPI_Scatterv(flat_A.data(), send_counts.data(), displs.data(), MPI_DOUBLE,
        local_A.data(), local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Каждый процесс вычисляет свою часть
    vector<double> local_result(local_rows * n, 0.0);

    // Параллельное умножение
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += local_A[i * n + k] * flat_B[k * n + j];
            }
            local_result[i * n + j] = sum;
        }
    }

    // Сбор результатов
    if (rank == 0) {
        flat_result.resize(n * n);
    }

    MPI_Gatherv(local_result.data(), local_rows * n, MPI_DOUBLE,
        flat_result.data(), send_counts.data(), displs.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto stop = chrono::high_resolution_clock::now();

    // Процесс 0 сохраняет результаты
    if (rank == 0) {
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        cout << "Multiplication finished in " << duration.count() << " ms\n";

        // Восстанавливаем матрицу результата
        vector<vector<double> > result(n, vector<double>(n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = flat_result[i * n + j];
            }
        }

        // Проверка результата
        Write_Matrix_To_File("tmp.txt", result);

        string command = "python3 paral.py " + matrixA_file + " " + matrixB_file + " tmp.txt";
        int ret = system(command.c_str());

        bool correctly = (ret == 0);

        if (correctly) {
            cout << "Result is CORRECT!\n";
        } else {
            cout << "Result is INCORRECT!\n";
        }

        Write_Matrix_To_File(result_file, result);
        Write_Experiment_Results("experiments_mpi.csv", matrix_size, num_processes, duration, correctly);

        cout << "Experiment " << exp_num << " completed\n";
    }

    MPI_Finalize();
    return 0;
}
