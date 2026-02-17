#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>//для определения времени на выполнение задачи

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

vector<vector<double>> Multiply_Matrix(const vector<vector<double>>& A,const vector<vector<double>>& B) {

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
        std::cout << "Error " << filename << "\n";
        return;
    }

    int n = matrix.size();

    file << n << "\n";

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << matrix[i][j];
            if (j < n - 1) {
                file << " "; //пробел между числами(если не последний столбец)
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
    file << "Duration: " << dur;
    file << "\nCorrectly: " << correctly;
    file.close();
    cout << "Result writing " << filename << "\n";
}


int main() {

    vector<vector<double>> MatrixA = Read_Matrix_From_File("MatrixA.txt");
    cout << "Matrix A loading. SIZE: " << MatrixA.size() << "x" << MatrixA.size() << "\n";

    vector<vector<double>> MatrixB = Read_Matrix_From_File("MatrixB.txt");
    cout << "Matrix B loading. SIZE: " << MatrixB.size() << "x" << MatrixB.size() << "\n";

    if (MatrixA.size() != MatrixB.size()) {
        cout << "Error!!!" << "\n";
        return 1;
    }

    auto start = chrono::high_resolution_clock::now();
    vector<vector<double>> result = Multiply_Matrix(MatrixA, MatrixB);
    auto stop = chrono::high_resolution_clock::now();

    auto dur = chrono::duration_cast<chrono::milliseconds>(stop - start);

    cout << "Finish!" << "\n";

    bool correctly = false;

    Write_Matrix_To_File("tmp.txt", result);

    int ret = system("python paral.py MatrixA.txt MatrixB.txt tmp.txt");

    if (ret == 0)
    {
        correctly = true;
    }
    else if (ret == 1)
    {
        correctly = false;
    }
    else 
    {
        cout << "Error compare matrices\n";
    }

    cout << "Writing in file result.txt..." << "\n";
    Write_Matrix_To_File("result.txt", result, dur, correctly);

    cout << "Sum operations multiplication: "<< MatrixA.size() * MatrixA.size() * MatrixA.size() << "\n";
   


    return 0;
}
