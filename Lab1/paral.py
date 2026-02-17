import numpy as np
import sys


def read_matrix_from_file(filenameA, filenameB, filenameRes):
    with open(filenameA, "r") as f:
       
        n = int(f.readline().strip())

        matrix1 = []
        for _ in range(n):
            row = list(map(int, f.readline().strip().split()))
            matrix1.append(row)
    with open(filenameB, "r") as f:
        n = int(f.readline().strip())

        matrix2 = []
        for _ in range(n):
            row = list(map(int, f.readline().strip().split()))
            matrix2.append(row)
    with open(filenameRes, "r") as f:
        n = int(f.readline().strip())

        matrix3 = []
        for _ in range(n):
            row = list(map(int, f.readline().strip().split()))
            matrix3.append(row)

    return (
        np.array(matrix1, dtype=np.int64),
        np.array(matrix2, dtype=np.int64),
        np.array(matrix3, dtype=np.int64),
    )


def main():
    if len(sys.argv) != 4:
        sys.exit(-1)

    filenameA = sys.argv[1]
    filenameB = sys.argv[2]
    filenameRes = sys.argv[3]

    try:
        matrix1, matrix2, matrix3 = read_matrix_from_file(filenameA, filenameB, filenameRes)

        n = matrix1.shape[0]

        if (
            matrix1.shape != (n, n)
            or matrix2.shape != (n, n)
            or matrix3.shape != (n, n)
        ):
            sys.exit(-1)

        product = np.dot(matrix1, matrix2)

        if np.array_equal(product, matrix3):
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        sys.exit(-1)


if __name__ == "__main__":
    main()
