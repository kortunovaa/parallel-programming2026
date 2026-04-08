#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <mpi.h>

using namespace std;

vector<vector<double>> Read_Matrix_From_File(const string& filename) {
    ifstream file(filename);

    if (!file.is_open()) {
        cout << "Error opening file: " << filename << "\n";
        exit(1);
    }

    int size;
    file >> size;

    vector<vector<double>> matrix(size, vector<double>(size));

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            file >> matrix[i][j];
        }
    }

    file.close();
    return matrix;
}

// Параллельное умножение матриц с распределением строк
vector<vector<double>> Multiply_Matrix_Parallel(const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    int rank, int num_processes) {
    int n = A.size();

    // Рассчитываем, сколько строк обработает каждый процесс
    int rows_per_process = n / num_processes;
    int remainder = n % num_processes;

    // Определяем диапазон строк для текущего процесса
    int start_row = rank * rows_per_process + min(rank, remainder);
    int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);
    int local_rows = end_row - start_row;

    // Каждый процесс вычисляет свою часть результата
    vector<vector<double>> local_result(local_rows, vector<double>(n, 0.0));

    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[start_row + i][k] * B[k][j];
            }
            local_result[i][j] = sum;
        }
    }

    return local_result;
}

void Write_Matrix_To_File(const string& filename, const vector<vector<double>>& matrix) {
    ofstream file(filename);

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
    ofstream file(filename, ios::app);

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

    string matrixA_file = "MatrixA_" + to_string(matrix_size) + "_" + to_string(exp_num) + ".txt";
    string matrixB_file = "MatrixB_" + to_string(matrix_size) + "_" + to_string(exp_num) + ".txt";
    string result_file = "result_" + to_string(matrix_size) + "_" + to_string(exp_num) + ".txt";

    vector<vector<double>> MatrixA, MatrixB, result;

    // Процесс 0 читает матрицы и рассылает их всем
    if (rank == 0) {
        cout << "\n=== MPI Experiment " << exp_num << " ===\n";
        cout << "Matrix size: " << matrix_size << "x" << matrix_size << "\n";
        cout << "Number of processes: " << num_processes << "\n";

        MatrixA = Read_Matrix_From_File(matrixA_file);
        cout << "Matrix A loaded. SIZE: " << MatrixA.size() << "x" << MatrixA.size() << "\n";

        MatrixB = Read_Matrix_From_File(matrixB_file);
        cout << "Matrix B loaded. SIZE: " << MatrixB.size() << "x" << MatrixB.size() << "\n";

        if (MatrixA.size() != MatrixB.size()) {
            cout << "Error!!! Matrices sizes don't match!\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Синхронизация перед замером времени
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = chrono::high_resolution_clock::now();

    // Рассылаем матрицу B всем процессам
    int n = matrix_size;
    vector<double> flat_B;

    if (rank == 0) {
        // Преобразуем B в плоский массив
        flat_B.resize(n * n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                flat_B[i * n + j] = MatrixB[i][j];
            }
        }
    }

    // Выделяем память для B на всех процессах
    vector<double> local_B(n * n);

    // Рассылаем B всем процессам
    MPI_Bcast(local_B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Если не процесс 0, восстанавливаем матрицу B из плоского массива
    if (rank != 0) {
        MatrixB.resize(n, vector<double>(n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                MatrixB[i][j] = local_B[i * n + j];
            }
        }
    }

    // Рассылаем строки матрицы A процессам
    // Рассчитываем распределение строк
    int rows_per_process = n / num_processes;
    int remainder = n % num_processes;

    // Определяем сколько строк получит каждый процесс
    vector<int> send_counts(num_processes);
    vector<int> displs(num_processes);

    int current_row = 0;
    for (int i = 0; i < num_processes; i++) {
        send_counts[i] = rows_per_process + (i < remainder ? 1 : 0);
        displs[i] = current_row;
        current_row += send_counts[i];
    }

    // Подготавливаем плоскую матрицу A для рассылки
    vector<double> flat_A;
    if (rank == 0) {
        flat_A.resize(n * n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                flat_A[i * n + j] = MatrixA[i][j];
            }
        }
    }

    // Каждый процесс получает свои строки
    int local_rows = send_counts[rank];
    vector<double> local_A(local_rows * n);

    MPI_Scatterv(flat_A.data(), send_counts.data() * n, displs.data() * n,
        MPI_DOUBLE, local_A.data(), local_rows * n, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // Преобразуем полученные строки в матрицу
    vector<vector<double>> local_MatrixA(local_rows, vector<double>(n));
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            local_MatrixA[i][j] = local_A[i * n + j];
        }
    }

    // КАЖДЫЙ ПРОЦЕСС вычисляет свою часть результата ПАРАЛЛЕЛЬНО!
    vector<vector<double>> local_result = Multiply_Matrix_Parallel(
        local_MatrixA, MatrixB, rank, num_processes);

    // Собираем результаты от всех процессов
    vector<double> flat_result;
    if (rank == 0) {
        flat_result.resize(n * n);
    }

    // Преобразуем локальный результат в плоский массив
    vector<double> local_flat(local_rows * n);
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            local_flat[i * n + j] = local_result[i][j];
        }
    }

    // Собираем все части результата на процессе 0
    MPI_Gatherv(local_flat.data(), local_rows * n, MPI_DOUBLE,
        flat_result.data(), send_counts.data() * n, displs.data() * n,
        MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Синхронизация после умножения
    MPI_Barrier(MPI_COMM_WORLD);
    auto stop = chrono::high_resolution_clock::now();

    // Только процесс 0 собирает и проверяет результат
    if (rank == 0) {
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        cout << "Multiplication finished in " << duration.count() << " ms\n";

        // Восстанавливаем матрицу результата из плоского массива
        result.resize(n, vector<double>(n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = flat_result[i * n + j];
            }
        }

        // Проверка результата
        bool correctly = false;

        string command = "python paral.py " + matrixA_file + " " + matrixB_file + " " + result_file;
        int ret = system(command.c_str());

        // Сохраняем результат во временный файл для проверки
        Write_Matrix_To_File("tmp.txt", result);
        command = "python paral.py " + matrixA_file + " " + matrixB_file + " tmp.txt";
        ret = system(command.c_str());

        if (ret == 0) {
            correctly = true;
            cout << "Result is CORRECT!\n";
        }
        else if (ret == 1) {
            correctly = false;
            cout << "Result is INCORRECT!\n";
        }
        else {
            cout << "Error comparing matrices\n";
        }

        Write_Experiment_Results("experiments_mpi.csv", matrix_size, num_processes, duration, correctly);

        cout << "Experiment " << exp_num << " completed\n";
        cout << "Total operations: " << matrix_size * matrix_size * matrix_size << "\n";
        cout << "Parallel efficiency: " << (duration.count() > 0 ?
            (double)(matrix_size * matrix_size * matrix_size) / (duration.count() * num_processes * 1e6) : 0) << " MFlops/process\n";
    }

    MPI_Finalize();
    return 0;
}
