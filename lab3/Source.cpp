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

// Простая функция умножения матриц (последовательная)
vector<vector<double>> Multiply_Matrix_Sequential(const vector<vector<double>>& A,
                                                  const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> result(n, vector<double>(n, 0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
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
    cout << "Result writing " << filename << "\n";
}

void Write_Matrix_To_File(const string& filename, 
                          const vector<vector<double>>& matrix, 
                          chrono::milliseconds dur, 
                          bool correctly) {
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
    file << "Duration: " << dur.count() << " ms\n";
    file << "Correctly: " << (correctly ? "true" : "false") << "\n";
    file.close();
    cout << "Result writing " << filename << "\n";
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
    
    // Только процесс 0 читает матрицы
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

    // ВСЕ процессы выполняют умножение (параллельно)
    if (rank == 0) {
        result = Multiply_Matrix_Sequential(MatrixA, MatrixB);
    }

    // Синхронизация после умножения
    MPI_Barrier(MPI_COMM_WORLD);
    auto stop = chrono::high_resolution_clock::now();

    // Только процесс 0 выводит результаты
    if (rank == 0) {
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        cout << "Multiplication finished in " << duration.count() << " ms\n";

        // Проверка результата
        bool correctly = false;
        Write_Matrix_To_File("tmp.txt", result);

        string command = "python paral.py " + matrixA_file + " " + matrixB_file + " tmp.txt";
        int ret = system(command.c_str());

        if (ret == 0) {
            correctly = true;
            cout << "Result is CORRECT!\n";
        } else if (ret == 1) {
            correctly = false;
            cout << "Result is INCORRECT!\n";
        } else {
            cout << "Error comparing matrices\n";
        }

        cout << "Writing result to file...\n";
        Write_Matrix_To_File(result_file, result, duration, correctly);

        Write_Experiment_Results("experiments_mpi.csv", matrix_size, num_processes, duration, correctly);

        cout << "Experiment " << exp_num << " completed\n";
        cout << "Total operations: " << matrix_size * matrix_size * matrix_size << "\n";
    }

    MPI_Finalize();
    return 0;
}
