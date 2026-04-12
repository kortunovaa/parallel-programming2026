#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cuda_runtime.h>
#include <iomanip>

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Простое CUDA ядро (каждый поток - один элемент результата)
__global__ void matrixMulSimple(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Оптимизированное ядро с shared memory (для блоков 32x32)
__global__ void matrixMulOptimized(const double* A, const double* B, double* C, int n) {
    __shared__ double sharedA[32][32];
    __shared__ double sharedB[32][32];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    double sum = 0.0;
    
    for (int tile = 0; tile < (n + blockDim.x - 1) / blockDim.x; tile++) {
        if (row < n && (tile * blockDim.x + tx) < n) {
            sharedA[ty][tx] = A[row * n + tile * blockDim.x + tx];
        } else {
            sharedA[ty][tx] = 0.0;
        }
        
        if ((tile * blockDim.y + ty) < n && col < n) {
            sharedB[ty][tx] = B[(tile * blockDim.y + ty) * n + col];
        } else {
            sharedB[ty][tx] = 0.0;
        }
        
        __syncthreads();
        
        for (int k = 0; k < blockDim.x; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

vector<vector<double>> ReadMatrix(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }
    
    int n;
    file >> n;
    vector<vector<double>> matrix(n, vector<double>(n));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file >> matrix[i][j];
        }
    }
    file.close();
    return matrix;
}

void WriteMatrix(const string& filename, const vector<vector<double>>& matrix) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
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

void WriteResults(const string& filename, int matrix_size, const string& block_config,
                  double time_ms, bool correctly, double gflops) {
    ofstream file(filename, ios::app);
    if (!file.is_open()) return;
    
    file << matrix_size << "," << block_config << ","
         << fixed << setprecision(3) << time_ms << ","
         << fixed << setprecision(2) << gflops << ","
         << (correctly ? "1" : "0") << "\n";
    file.close();
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cout << "Usage: " << argv[0] 
             << " <matrix_size> <block_x> <block_y> <experiment_number>\n";
        cout << "Example: " << argv[0] << " 200 16 16 1\n";
        return 1;
    }
    
    int n = atoi(argv[1]);
    int block_x = atoi(argv[2]);
    int block_y = atoi(argv[3]);
    int exp_num = atoi(argv[4]);
    
    string matrixA_file = "MatrixA_" + to_string(n) + "_" + to_string(exp_num) + ".txt";
    string matrixB_file = "MatrixB_" + to_string(n) + "_" + to_string(exp_num) + ".txt";
    string result_file = "result_cuda_" + to_string(n) + "_" + 
                         to_string(block_x) + "x" + to_string(block_y) + "_" + 
                         to_string(exp_num) + ".txt";
    
    cout << "\n=== CUDA Experiment " << exp_num << " ===\n";
    cout << "Matrix size: " << n << "x" << n << "\n";
    cout << "Block config: " << block_x << "x" << block_y << "\n";
    
    // Читаем матрицы
    vector<vector<double>> A = ReadMatrix(matrixA_file);
    vector<vector<double>> B = ReadMatrix(matrixB_file);
    
    cout << "Matrices loaded\n";
    
    // Подготовка данных для GPU
    size_t bytes = n * n * sizeof(double);
    vector<double> flatA(n * n), flatB(n * n), flatC(n * n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            flatA[i * n + j] = A[i][j];
            flatB[i * n + j] = B[i][j];
        }
    }
    
    // Выделение памяти на GPU
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    
    // Копирование данных на GPU
    CUDA_CHECK(cudaMemcpy(d_A, flatA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, flatB.data(), bytes, cudaMemcpyHostToDevice));
    
    // Конфигурация запуска
    dim3 blockSize(block_x, block_y);
    dim3 gridSize((n + block_x - 1) / block_x, (n + block_y - 1) / block_y);
    
    cout << "Grid size: " << gridSize.x << "x" << gridSize.y << "\n";
    cout << "Total threads: " << gridSize.x * gridSize.y * block_x * block_y << "\n";
    
    // Замер времени
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    // Запуск ядра
    if (block_x == 32 && block_y == 32 && n % 32 == 0) {
        matrixMulOptimized<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    } else {
        matrixMulSimple<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
    
    // Копирование результата обратно
    CUDA_CHECK(cudaMemcpy(flatC.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    
    // Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Восстановление матрицы результата
    vector<vector<double>> C(n, vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = flatC[i * n + j];
        }
    }
    
    // Проверка результата
    WriteMatrix("tmp.txt", C);
    string cmd = "python paral.py " + matrixA_file + " " + matrixB_file + " tmp.txt";
    int ret = system(cmd.c_str());
    bool correct = (ret == 0);
    
    // Расчет производительности
    double gflops = (2.0 * n * n * n) / (time_ms / 1000.0) / 1e9;
    
    cout << "\nExecution time: " << fixed << setprecision(3) << time_ms << " ms\n";
    cout << "Performance: " << fixed << setprecision(2) << gflops << " GFLOPS\n";
    cout << "Result: " << (correct ? "CORRECT" : "INCORRECT") << "\n";
    
    // Сохранение результата
    ofstream resultFile(result_file);
    resultFile << n << "\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            resultFile << C[i][j];
            if (j < n - 1) resultFile << " ";
        }
        resultFile << "\n";
    }
    resultFile << "# Time: " << time_ms << " ms\n";
    resultFile << "# Block config: " << block_x << "x" << block_y << "\n";
    resultFile << "# GFLOPS: " << gflops << "\n";
    resultFile.close();
    
    // Запись в CSV
    WriteResults("experiments_cuda.csv", n, to_string(block_x) + "x" + to_string(block_y),
                 time_ms, correct, gflops);
    
    cout << "\nExperiment completed!\n";
    
    return 0;
}
