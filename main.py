from itertools import permutations


class SimpleIterationSolver:

    def __init__(self, A: list[list[float]], b: list[float],
                 epsilon: float, max_iterations: int = 1000) -> None:
        self.n: int = len(A)
        self.A: list[list[float]] = [row[:] for row in A]
        self.b: list[float] = b[:]
        self.epsilon: float = epsilon
        self.max_iterations: int = max_iterations
        self.C: list[list[float]] = []
        self.d: list[float] = []
        self.norm_C1: float = 0.0
        self.norm_C2: float = 0.0
        self.norm_C3: float = 0.0
        self.solution: list[float] = []
        self.iterations: int = 0
        self.errors: list[float] = []

    def check_diagonal_dominance(self, A: list[list[float]]) -> bool:
        n: int = len(A)
        strict_count: int = 0
        for i in range(n):
            diag: float = abs(A[i][i])
            off_diag_sum: float = sum(abs(A[i][j]) for j in range(n) if j != i)
            if diag < off_diag_sum:
                return False
            if diag > off_diag_sum:
                strict_count += 1
        return strict_count >= 1

    def achieve_diagonal_dominance(self) -> bool:
        if self.check_diagonal_dominance(self.A):
            return True

        n: int = self.n

        rows: list[int] = list(range(n))
        if n <= 10:
            for perm in permutations(rows):
                A_perm: list[list[float]] = [self.A[perm[i]] for i in range(n)]
                b_perm: list[float] = [self.b[perm[i]] for i in range(n)]
                if self.check_diagonal_dominance(A_perm):
                    self.A = A_perm
                    self.b = b_perm
                    return True
        else:
            used: list[bool] = [False] * n
            order: list[int] = [0] * n
            for j in range(n):
                best_row: int = -1
                best_val: float = -1.0
                for i in range(n):
                    if not used[i] and abs(self.A[i][j]) > best_val:
                        best_val = abs(self.A[i][j])
                        best_row = i
                if best_row == -1:
                    return False
                order[j] = best_row
                used[best_row] = True

            A_new: list[list[float]] = [self.A[order[j]] for j in range(n)]
            b_new: list[float] = [self.b[order[j]] for j in range(n)]
            if self.check_diagonal_dominance(A_new):
                self.A = A_new
                self.b = b_new
                return True

            count: int = 0
            for perm in permutations(rows):
                count += 1
                if count > 1_000_000:
                    break
                A_perm = [self.A[perm[i]] for i in range(n)]
                b_perm = [self.b[perm[i]] for i in range(n)]
                if self.check_diagonal_dominance(A_perm):
                    self.A = A_perm
                    self.b = b_perm
                    return True

        return False

    def build_iteration_matrix(self) -> None:
        n: int = self.n
        self.C = [[0.0] * n for _ in range(n)]
        self.d = [0.0] * n

        for i in range(n):
            if abs(self.A[i][i]) < 1e-15:
                raise ValueError(
                    f"Диагональный элемент a[{i + 1}][{i + 1}] равен нулю. "
                    "Решение невозможно."
                )
            self.d[i] = self.b[i] / self.A[i][i]
            for j in range(n):
                if i != j:
                    self.C[i][j] = -self.A[i][j] / self.A[i][i]

    def compute_norms(self) -> tuple[float, float, float]:
        n: int = self.n

        self.norm_C1 = max(
            sum(abs(self.C[i][j]) for j in range(n))
            for i in range(n)
        )

        self.norm_C2 = max(
            sum(abs(self.C[i][j]) for i in range(n))
            for j in range(n)
        )

        self.norm_C3 = sum(
            self.C[i][j] ** 2
            for i in range(n) for j in range(n)
        ) ** 0.5

        return self.norm_C1, self.norm_C2, self.norm_C3

    def solve(self) -> bool:
        n: int = self.n
        x_prev: list[float] = self.d[:]
        x_curr: list[float] = [0.0] * n

        for k in range(1, self.max_iterations + 1):
            for i in range(n):
                s: float = 0.0
                for j in range(n):
                    s += self.C[i][j] * x_prev[j]
                x_curr[i] = s + self.d[i]

            self.errors = [abs(x_curr[i] - x_prev[i]) for i in range(n)]
            max_error: float = max(self.errors)

            if max_error <= self.epsilon:
                self.solution = x_curr[:]
                self.iterations = k
                return True

            x_prev = x_curr[:]

        self.solution = x_curr[:]
        self.iterations = self.max_iterations
        return False


def read_from_keyboard() -> tuple[int, list[list[float]], list[float], float]:
    while True:
        try:
            n: int = int(input("Введите размерность матрицы n (n <= 20): "))
            if 1 <= n <= 20:
                break
            print("Ошибка: n должно быть от 1 до 20.")
        except ValueError:
            print("Ошибка: введите целое число.")

    print(f"Введите расширенную матрицу ({n} строк, {n + 1} столбцов):")
    print("Каждая строка: a_i1 a_i2 ... a_in b_i")

    A: list[list[float]] = []
    b: list[float] = []
    for i in range(n):
        while True:
            try:
                line: list[str] = input(f"  Строка {i + 1}: ").split()
                if len(line) != n + 1:
                    print(f"  Ошибка: ожидается {n + 1} чисел.")
                    continue
                vals: list[float] = [float(x) for x in line]
                A.append(vals[:n])
                b.append(vals[n])
                break
            except ValueError:
                print("  Ошибка: введите числа.")

    while True:
        try:
            epsilon: float = float(input("Введите точность (epsilon): "))
            if epsilon > 0:
                break
            print("Ошибка: точность должна быть > 0.")
        except ValueError:
            print("Ошибка: введите число.")

    return n, A, b, epsilon


def read_from_file(filename: str) -> tuple[int, list[list[float]], list[float], float]:
    """
    Формат файла:
        n
        a_11 a_12 ... a_1n b_1
        a_21 a_22 ... a_2n b_2
        ...
        a_n1 a_n2 ... a_nn b_n
        epsilon
    """
    with open(filename, 'r') as f:
        lines: list[str] = [line.strip() for line in f if line.strip()]

    n: int = int(lines[0])
    if n < 1 or n > 20:
        raise ValueError(f"Размерность n={n} вне допустимого диапазона [1, 20].")

    A: list[list[float]] = []
    b: list[float] = []
    for i in range(1, n + 1):
        vals: list[float] = [float(x) for x in lines[i].split()]
        if len(vals) != n + 1:
            raise ValueError(
                f"Строка {i}: ожидается {n + 1} чисел, получено {len(vals)}."
            )
        A.append(vals[:n])
        b.append(vals[n])

    epsilon: float = float(lines[n + 1])
    if epsilon <= 0:
        raise ValueError("Точность должна быть > 0.")

    return n, A, b, epsilon


def print_matrix(label: str, matrix: list[list[float]],
                 vec: list[float] | None = None) -> None:
    print(f"\n{label}:")
    n: int = len(matrix)
    for i in range(n):
        row_str: str = "  ".join(f"{matrix[i][j]:>10.4f}" for j in range(n))
        if vec is not None:
            row_str += f"  | {vec[i]:>10.4f}"
        print(f"  {row_str}")


def print_iteration_table(solver: SimpleIterationSolver) -> None:
    n: int = solver.n
    x_prev: list[float] = solver.d[:]

    header: str = f"{'k':>3}"
    for i in range(n):
        header += f"{'x' + str(i + 1):>12}"
    header += f"{'max|dx|':>12}"
    print(header)
    print("-" * len(header))

    row: str = f"{0:>3}"
    for i in range(n):
        row += f"{x_prev[i]:>12.6f}"
    row += f"{'-':>12}"
    print(row)

    for k in range(1, solver.iterations + 1):
        x_curr: list[float] = [0.0] * n
        for i in range(n):
            s: float = sum(solver.C[i][j] * x_prev[j] for j in range(n))
            x_curr[i] = s + solver.d[i]

        errors: list[float] = [abs(x_curr[i] - x_prev[i]) for i in range(n)]
        max_err: float = max(errors)

        row = f"{k:>3}"
        for i in range(n):
            row += f"{x_curr[i]:>12.6f}"
        row += f"{max_err:>12.6f}"
        print(row)

        x_prev = x_curr[:]


def main() -> None:
    while True:
        choice: str = input("\nВыберите способ ввода данных:\n"
                       "  1 — с клавиатуры\n"
                       "  2 — из файла\n"
                       "Ваш выбор: ").strip()
        if choice in ('1', '2'):
            break
        print("Ошибка: введите 1 или 2.")

    try:
        if choice == '1':
            n, A, b, epsilon = read_from_keyboard()
        else:
            filename = input("Введите имя файла: ").strip()
            n, A, b, epsilon = read_from_file(filename)
    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"Ошибка при чтении данных: {e}")
        return

    solver = SimpleIterationSolver(A, b, epsilon)

    print_matrix("Исходная матрица [A|b]", solver.A, solver.b)

    if solver.check_diagonal_dominance(solver.A):
        print("Диагональное преобладание выполняется.")
    else:
        print("Диагональное преобладание НЕ выполняется. Выполняется перестановка строк.")
        if solver.achieve_diagonal_dominance():
            print("Диагональное преобладание достигнуто после перестановки.")
            print_matrix("Матрица после перестановки [A|b]", solver.A, solver.b)
        else:
            print("Невозможно достичь диагонального преобладания перестановкой строк.")
            ans = input("Продолжить всё равно? (да/нет): ").strip().lower()
            if ans not in ('да', 'yes', 'y', 'д'):
                print("Вычисление прервано.")
                return

    try:
        solver.build_iteration_matrix()
    except ValueError as e:
        print(f"Ошибка: {e}")
        return

    print_matrix("Матрица C", solver.C)
    print(f"\nВектор d: [{', '.join(f'{x:.4f}' for x in solver.d)}]")

    norm1, norm2, norm3 = solver.compute_norms()
    print(f"\nНормы матрицы C:")
    print(f"  ||C||_1 = {norm1:.6f}  (макс. строчная сумма)")
    print(f"  ||C||_2 = {norm2:.6f}  (макс. столбцовая сумма)")
    print(f"  ||C||_3 = {norm3:.6f}  (евклидова норма)")

    convergence = any(n < 1.0 for n in (norm1, norm2, norm3))
    if convergence:
        satisfied = []
        if norm1 < 1.0:
            satisfied.append("||C||_1")
        if norm2 < 1.0:
            satisfied.append("||C||_2")
        if norm3 < 1.0:
            satisfied.append("||C||_3")
        print(f"Условие сходимости выполнено ({', '.join(satisfied)} < 1).")
    else:
        print("ВНИМАНИЕ: ни одна норма не меньше 1, достаточное условие сходимости не выполнено")
        ans = input("Продолжить всё равно? (да/нет): ").strip().lower()
        if ans not in ('да', 'yes', 'y', 'д'):
            print("Вычисление прервано.")
            return

    print(f"\nНачальное приближение x^(0) = d = [{', '.join(f'{x:.4f}' for x in solver.d)}]")
    print(f"Точность: {epsilon}")
    print()

    converged = solver.solve()

    print("Таблица итераций:")
    print_iteration_table(solver)

    print()
    if converged:
        print(f"Решение найдено за {solver.iterations} итераций.")
    else:
        print(f"Решение не сошлось за {solver.iterations} итераций.")

    print("\nВектор неизвестных:")
    for i in range(n):
        print(f"  x{i + 1} = {solver.solution[i]:.6f}")

    print("\nВектор погрешностей |x_i^(k) - x_i^(k-1)|:")
    for i in range(n):
        print(f"  |dx{i + 1}| = {solver.errors[i]:.10f}")


if __name__ == "__main__":
    main()
