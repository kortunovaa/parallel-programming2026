import numpy as np

def generate_matrices_for_experiments():
    np.random.seed(42)
    sizes = [200, 400, 800, 1200, 1600, 2000]
    repeats = 3  #Количество повторений для каждого размера
    
    for n in sizes:
        for exp_num in range(1, repeats + 1):
            # Генерируем матрицы
            A = np.random.randint(-100, 100, size=(n, n))
            B = np.random.randint(-100, 100, size=(n, n))
            
            with open(f"MatrixA_{n}_{exp_num}.txt", "w") as f:
                f.write(f"{n}\n")
                for i in range(n):
                    f.write(" ".join(str(x) for x in A[i]) + "\n")
            
            with open(f"MatrixB_{n}_{exp_num}.txt", "w") as f:
                f.write(f"{n}\n")
                for i in range(n):
                    f.write(" ".join(str(x) for x in B[i]) + "\n")
            
            print(f"Сгенерированы матрицы {n}x{n}, эксперимент #{exp_num}")

if __name__ == "__main__":
    generate_matrices_for_experiments()
    print("Готово!")
