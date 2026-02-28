#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>     
#include <omp.h>

using namespace std;

vector<vector<double>> Read_Matrix_From_File(const string& filename) {
    ifstream file(filename);

    if (!file.is_open()) {
        cout << "Error1" << filename << "\n";
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

vector<vector<double>> Multiply_Matrix(const vector<vector<double>>& A,
    const vector<vector<double>>& B,
    int num_threads) {

    int n = A.size();
    vector<vector<double>> result(n, vector<double>(n, 0));

    omp_set_num_threads(num_threads);

#pragma omp parallel for schedule(dynamic)
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
        std::cout << "Error " << filename << "\n";
        return;
    }

    int n = matrix.size();
    file << n << "\n";

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << matrix[i][j];
            if (j < n - 1) {
                file << " ";
            }
        }
        file << "\n";
    }

    file.close();
    cout << "Result writing " << filename << "\n";
}

void Write_Matrix_To_File(const string& filename, const vector<vector<double>>& matrix, chrono::milliseconds dur, bool correctly) {
    ofstream file(filename);

    if (!file.is_open()) {
        std::cout << "Error " << filename << "\n";
        return;
    }

    int n = matrix.size();
    file << n << "\n";

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << matrix[i][j];
            if (j < n - 1) {
                file << " ";
            }
        }
        file << "\n";
    }
    file << "Duration: " << dur.count() << " ms\n";
    file << "Correctly: " << (correctly ? "true" : "false") << "\n";
    file.close();
    cout << "Result writing " << filename << "\n";
}

void Write_Experiment_Results(const string& filename, int matrix_size, int num_threads, chrono::milliseconds duration,bool correctly) {
    ofstream file(filename, ios::app);

    if (!file.is_open()) {
        std::cout << "Error " << filename << "\n";
        return;
    }

    file << matrix_size << "," << num_threads << ","
        << duration.count() << "," << (correctly ? "1" : "0") << "\n";

    file.close();
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " <matrix_size> <num_threads> <experiment_number>\n";
        cout << "Example: " << argv[0] << " 200 2 1\n";
        return 1;
    }

    int matrix_size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    int exp_num = atoi(argv[3]);

    string matrixA_file = "MatrixA_" + to_string(matrix_size) + "_" + to_string(exp_num) + ".txt";
    string matrixB_file = "MatrixB_" + to_string(matrix_size) + "_" + to_string(exp_num) + ".txt";
    string result_file = "result_" + to_string(matrix_size) + "_" + to_string(exp_num) + ".txt";

    cout << "\nЭксперимент " << exp_num << "\n";
    cout << "Размер матрицы: " << matrix_size << "x" << matrix_size << "\n";
    cout << "Количество потоков: " << num_threads << "\n";

    vector<vector<double>> MatrixA = Read_Matrix_From_File(matrixA_file);
    cout << "Matrix A loaded. SIZE: " << MatrixA.size() << "x" << MatrixA.size() << "\n";

    vector<vector<double>> MatrixB = Read_Matrix_From_File(matrixB_file);
    cout << "Matrix B loaded. SIZE: " << MatrixB.size() << "x" << MatrixB.size() << "\n";

    if (MatrixA.size() != MatrixB.size()) {
        cout << "Error!!! Matrices sizes don't match!\n";
        return 1;
    }

    auto start = chrono::high_resolution_clock::now();
    vector<vector<double>> result = Multiply_Matrix(MatrixA, MatrixB, num_threads);
    auto stop = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

    cout << "Multiplication finished in " << duration.count() << " ms\n";

    bool correctly = false;
    Write_Matrix_To_File("tmp.txt", result);

    string command = "python paral.py " + matrixA_file + " " + matrixB_file + " tmp.txt";
    int ret = system(command.c_str());

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

    cout << "Writing result to file...\n";
    Write_Matrix_To_File(result_file, result, duration, correctly);

    Write_Experiment_Results("experiments.csv", matrix_size, num_threads, duration, correctly);

    cout << "Experiment " << exp_num << " completed\n";
    cout << "Total operations: " << MatrixA.size() * MatrixA.size() * MatrixA.size() << "\n";

    return 0;
}
