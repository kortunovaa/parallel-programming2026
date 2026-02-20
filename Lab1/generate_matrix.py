import numpy as np


def generate_matrix():
    sizes = [100]
    np.random.seed(42)

    for n in sizes:
        A = np.random.randint(-21 ,210, size=(n, n))
        B = np.random.randint(-21 ,210,  size=(n, n))

        with open(f"input.txt", "a") as f:
            f.write(f"{n}\n")
            for i in range(n):
                f.write(" ".join(str(x) for x in A[i]) + "\n")
            f.write("\n")
            for i in range(n):
                f.write(" ".join(str(x) for x in B[i]) + "\n")
            f.write("\n")


if __name__ == "__main__":
    open(f"input.txt", "w")
    generate_matrix()
