class GaussSeidelSolver:
    def __init__(self):
        self.n = 0
        self.matrix = []
        self.b_vector = []
        self.epsilon = 0.0001
        self.max_iterations = 1000
        self.column_moves = []
        self.show_iterations = False

    def  get_valid_int(self, prompt, min_val=None, max_val=None):
        while True:
            try:
                value = int(input(prompt))
                if min_val is not None and value < min_val:
                    print(f"Значение должно быть >= {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"Значение должно быть <= {max_val}")
                    continue
                return value
            except ValueError:
                print("Ошибка: введите целое число")

    def get_valid_float(self, prompt, min_val=None, max_val=None, allow_zero=False):
        while True:
            try:
                value = float(input(prompt))
                if not allow_zero and value == 0:
                    print("Значение не должно быть равно 0")
                    continue
                if min_val is not None and value < min_val:
                    print(f"Значение должно быть >= {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"Значение должно быть <= {max_val}")
                    continue
                return value
            except ValueError:
                print("Ошибка: введите число")

    def read_input(self):
        while True:
            print("Выберите источник ввода (1 - клавиатура, 2 - файл): ")
            source = input().strip()
            if source in ['1', '2']:
                break
            print("Ошибка: введите 1 или 2")

        if source == '2':
            self._read_from_file()
        else:
            self._manual_input()

        while True:
            show = input("Показывать значения на каждой итерации? (y/n): ").strip().lower()
            if show in ['y', 'n']:
                self.show_iterations = (show == 'y')
                break
            print("Ошибка: введите y или n")

    def _read_from_file(self):
        while True:
            filename = input("Введите имя файла: ").strip()
            if not filename:
                print("Ошибка: имя файла не может быть пустым :(")
                continue
            try:
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    if len(lines) < 2:
                        print("Ошибка: недостаточно строк в файле :(")
                        continue

                    self.n = int(lines[0].strip())
                    if self.n <= 0 or self.n > 20:
                        print("Ошибка: размерность должна быть от 1 до 20")
                        continue

                    self.epsilon = float(lines[1].strip())
                    if self.epsilon <= 0:
                        print("Ошибка: точность должна быть положительной")
                        continue

                    if len(lines) < 2 + self.n:
                        print("Ошибка: недостаточно строк с данными матрицы")
                        continue

                    self.matrix = []
                    self.b_vector = []
                    for i in range(2, 2 + self.n):
                        row = list(map(float, lines[i].strip().split()))
                        if len(row) != self.n + 1:
                            print(f"Ошибка: строка {i - 1} должна содержать {self.n + 1} чисел")
                            self.matrix = []
                            self.b_vector = []
                            continue
                        self.b_vector.append(row.pop())
                        self.matrix.append(row)

                    if len(self.matrix) == self.n:
                        return
            except FileNotFoundError:
                print("Ошибка: файл не найден. Введите другое имя или выберите ввод с клавиатуры")
            except ValueError:
                print("Ошибка: некорректные данные в файле")
            except Exception as e:
                print(f"Ошибка при чтении файла: {e}")

    def _manual_input(self):
        self.n = self.get_valid_int("Введите размерность матрицы n (1-20): ", 1, 20)
        self.epsilon = self.get_valid_float("Введите точность (epsilon > 0): ", min_val=0.0000001)

        print(f"Введите матрицу коэффициентов и свободные члены ({self.n} строк по {self.n + 1} чисел):")
        self.matrix = []
        self.b_vector = []
        for i in range(self.n):
            while True:
                try:
                    row = list(map(float, input(f"Строка {i + 1}: ").split()))
                    if len(row) != self.n + 1:
                        print(f"Ошибка: должно быть {self.n + 1} чисел")
                        continue
                    self.b_vector.append(row.pop())
                    self.matrix.append(row)
                    break
                except ValueError:
                    print("Ошибка: введите числа через пробел")

    def check_diagonal_dominance(self):
        self.column_moves = list(range(self.n))

        for iter_num in range(self.n):
            dominated = True
            for i in range(self.n):
                diag = abs(self.matrix[i][i])
                row_sum = sum(abs(self.matrix[i][j]) for j in range(self.n) if j != i)
                if diag <= row_sum:
                    dominated = False
                    break

            if dominated:
                return True

            for i in range(self.n):
                max_val = abs(self.matrix[i][i])
                max_row = i
                max_col = i

                for k in range(i, self.n):
                    for j in range(i, self.n):
                        if abs(self.matrix[k][j]) > max_val:
                            max_val = abs(self.matrix[k][j])
                            max_row = k
                            max_col = j

                if max_row != i:
                    self.matrix[i], self.matrix[max_row] = self.matrix[max_row], self.matrix[i]
                    self.b_vector[i], self.b_vector[max_row] = self.b_vector[max_row], self.b_vector[i]

                if max_col != i:
                    for k in range(self.n):
                        self.matrix[k][i], self.matrix[k][max_col] = self.matrix[k][max_col], self.matrix[k][i]
                    self.column_moves[i], self.column_moves[max_col] = self.column_moves[max_col], \
                    self.column_moves[i]

        for i in range(self.n):
            diag = abs(self.matrix[i][i])
            row_sum = sum(abs(self.matrix[i][j]) for j in range(self.n) if j != i)
            if diag <= row_sum:
                print("Внимание: диагональное преобладание не достигнуто. Сходимость не гарантирована.")
                return False
        return True

    def get_matrix_norm(self):
        max_norm = 0
        for i in range(self.n):
            row_sum = sum(abs(self.matrix[i][j]) for j in range(self.n))
            if row_sum > max_norm:
                max_norm = row_sum
        return max_norm

    def solve(self):
        x = [0.0] * self.n
        iterations = 0
        errors = [0.0] * self.n

        if self.show_iterations:
            print("\n" + "=" * 60)
            print("Наши любимые итерации")
            print("=" * 60)
            header = "Итерация  | "
            for i in range(self.n):
                header += f"x[{i + 1}]         | "
            header += "Макс. погрешность"
            print(header)
            print("-" * 60)

        while iterations < self.max_iterations:
            x_prev = x[:]
            iterations += 1
            max_error = 0.0

            for i in range(self.n):
                sum_val = self.b_vector[i]
                for j in range(self.n):
                    if i != j:
                        sum_val -= self.matrix[i][j] * x[j]

                x[i] = sum_val / self.matrix[i][i]

            for i in range(self.n):
                errors[i] = abs(x[i] - x_prev[i])
                if errors[i] > max_error:
                    max_error = errors[i]

            if self.show_iterations:
                row = f"{iterations:^9} | "
                for i in range(self.n):
                    row += f"{x[i]:^12.6f} | "
                row += f"{max_error:.6e}"
                print(row)

            if max_error < self.epsilon:
                if self.show_iterations:
                    print("-" * 60)
                    print(f"Сходимость достигнута на итерации {iterations}")
                return x, iterations, errors

        print("Достигнут лимит итераций")
        return x, iterations, errors

    def restore_original_order(self, x):
        result = [0.0] * self.n
        for i in range(self.n):
            result[self.column_moves[i]] = x[i]
        return result

    def run(self):
        self.read_input()
        self.check_diagonal_dominance()

        print(f"\nНорма матрицы: {self.get_matrix_norm()}")

        solution, iterations, errors = self.solve()

        solution = self.restore_original_order(solution)
        errors = self.restore_original_order(errors)

        print(f"\nКоличество итераций: {iterations}")
        print("Вектор неизвестных:")
        for i, val in enumerate(solution):
            print(f"x[{i + 1}] = {val:.6f}")

        print("\nВектор погрешностей:")
        for i, val in enumerate(errors):
            print(f"|dx[{i + 1}]| = {val:.6f}")


if __name__ == "__main__":
    solver = GaussSeidelSolver()
    solver.run()