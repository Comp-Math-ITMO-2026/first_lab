import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import TextIO
import sys

type FloatArray = NDArray[np.float64]

@dataclass
class SeidelResult:
    x: FloatArray
    iters: int
    err: FloatArray
    a_norm: float


def row_match(cands: list[list[int]]) -> list[int] | None:
    n = len(cands)
    col_to_row = [-1] * n

    def try_row(row: int, seen_cols: list[bool]) -> bool:
        for col in cands[row]:
            if seen_cols[col]:
                continue
            seen_cols[col] = True
            if col_to_row[col] == -1 or try_row(col_to_row[col], seen_cols):
                col_to_row[col] = row
                return True
        return False

    for row in range(n):
        seen_cols = [False] * n
        if not try_row(row, seen_cols):
            return None

    return col_to_row


def is_diag_dom(a: FloatArray) -> bool:
    a_abs = np.abs(a)
    diag_abs = np.diag(a_abs)
    off_diag_sum = np.sum(a_abs, axis=1) - diag_abs
    weak_dom = diag_abs >= off_diag_sum
    strict_dom = diag_abs > off_diag_sum
    return bool(np.all(weak_dom) and np.any(strict_dom))


def reorder_diag_dom(
    a: FloatArray, b: FloatArray
) -> tuple[FloatArray, FloatArray] | None:
    n = a.shape[0]

    row_cands: list[list[int]] = []
    for i in range(n):
        row = a[i]
        row_sum = np.sum(np.abs(row))
        row_cands.append(
            [j for j in range(n) if abs(row[j]) >= row_sum - abs(row[j])]
        )

    row_order = row_match(row_cands)
    if row_order is None:
        return None

    a_reord = a[row_order, :]
    b_reord = b[row_order]

    if not is_diag_dom(a_reord):
        return None

    return a_reord, b_reord


def gauss_seidel(
    a: FloatArray,
    b: FloatArray,
    eps: float = 1e-5,
    max_iter: int = 1000,
) -> SeidelResult:
    n = a.shape[0]
    x: FloatArray = np.zeros(n, dtype=np.float64)

    for k in range(1, max_iter + 1):
        x_prev = x.copy()
        for i in range(n):
            left_sum = np.dot(a[i, :i], x[:i])
            right_sum = np.dot(a[i, i + 1 :], x_prev[i + 1 :])
            x[i] = (b[i] - left_sum - right_sum) / a[i, i]

        err: FloatArray = np.abs(x - x_prev)
        if np.all(err <= eps):
            a_norm = float(np.linalg.norm(a, ord=np.inf))
            res = SeidelResult(
                x=x,
                iters=k,
                err=err,
                a_norm=a_norm,
            )
            return res

    raise ValueError(
        f"Нет сходимости за {max_iter} итераций"
    )


def ask_yes(question: str, default: bool | None = None) -> bool:
    while True:
        if default is True:
            prompt = " [Y/n]: "
        elif default is False:
            prompt = " [y/N]: "
        else:
            prompt = " [y/n]: "
        ans = input(question + "?" + prompt).strip()
        if ans in ("y", "Y"):
            return True
        if ans in ("n", "N"):
            return False
        if ans == "" and default is not None:
            return default


def read_line(src: TextIO, prompt: str, interactive: bool) -> str:
    if interactive:
        return input(prompt)
    return src.readline().strip()


def read_pos_int(
    src: TextIO, prompt: str, err_msg: str, interactive: bool
) -> int:
    raw = read_line(src, prompt, interactive)
    try:
        val = int(raw)
        if val <= 0:
            raise ValueError
        return val
    except ValueError as exc:
        raise ValueError(err_msg) from exc


def read_pos_float(
    src: TextIO, prompt: str, err_msg: str, interactive: bool
) -> float:
    raw = read_line(src, prompt, interactive)
    try:
        val = float(raw)
        if val <= 0:
            raise ValueError
        return val
    except ValueError as exc:
        raise ValueError(err_msg) from exc


def read_system(
    n: int, src: TextIO, interactive: bool
) -> tuple[FloatArray, FloatArray]:
    a: FloatArray = np.empty((n, n), dtype=np.float64)
    b: FloatArray = np.empty(n, dtype=np.float64)
    row_count = 0

    for ln, line in enumerate(src, start=1):
        text = line.strip()
        if not text:
            continue
        if row_count >= n:
            raise ValueError(
                f"Лишняя строка: ожидалось {n}, получено {row_count + 1}"
            )

        vals = text.split()
        if len(vals) != n + 1:
            raise ValueError(
                f"Строка {ln}: ожидалось {n+1} чисел, получено {len(vals)}"
            )

        for col_n, val in enumerate(vals, start=1):
            try:
                num = float(val)
            except ValueError:
                raise ValueError(
                    f"Строка {ln}, столбец {col_n}: '{val}' не число"
                )
            if col_n <= n:
                a[row_count, col_n - 1] = num
            else:
                b[row_count] = num

        row_count += 1

        if interactive and row_count >= n:
            break

    if row_count != n:
        raise ValueError(
            f"Строк: ожидалось {n}, получено {row_count}"
        )

    return a, b


def show_system(a: FloatArray, b: FloatArray) -> None:
    n, m = a.shape
    for i in range(n):
        row_str = " ".join(f"{a[i, j]:8.3f}" for j in range(m))
        print(f"[{row_str}] | {b[i]:8.3f}")


def solve_from(src: TextIO, interactive: bool) -> None:
    try:
        n = read_pos_int(
            src,
            "Введите размер матрицы n: ",
            "Размер матрицы: нужно положительное целое.",
            interactive,
        )
        max_iter = read_pos_int(
            src,
            "Введите максимальное число итераций: ",
            "Итерации: нужно положительное целое.",
            interactive,
        )
        eps = read_pos_float(
            src,
            "Введите точность ε: ",
            "Точность: нужно положительное число.",
            interactive,
        )
    except ValueError as e:
        print(e)
        return

    if interactive:
        print(
            "Введите расширенную матрицу (A с последним столбцом b), строки построчно:"
        )
    try:
        a, b = read_system(n, src, interactive)
    except ValueError as e:
        print(f"Ошибка чтения: {e}")
        return

    print("Получена система:")
    show_system(a, b)

    if not is_diag_dom(a):
        print(
            "Нет диагонального преобладания, пробуем перестановку строк"
        )
        reordered_sys = reorder_diag_dom(a, b)
        if reordered_sys is not None:
            a, b = reordered_sys
            print("Cистема успешно преобразована:")
            show_system(a, b)
        else:
            print(
                "Не удалось получить диагональное преобладание. Сходимость не гарантирована.",
            )
            if (not interactive) or (
                interactive and not ask_yes("Продолжить", False)
            ):
                return

    try:
        res = gauss_seidel(a, b, eps=eps, max_iter=max_iter)
    except ValueError as e:
        print(f"Ошибка решения: {e}")
        return

    print("\n--- Результаты ---")
    print(f"Норма матрицы: {res.a_norm:.6f}")
    print("Вектор неизвестных x:")
    for i, val in enumerate(res.x, start=1):
        print(f"x{i} = {val:.6f}")
    print(f"Количество итераций: {res.iters}")
    print("Вектор погрешностей последней итерации:")
    print(np.array2string(res.err, precision=6, separator=", "))


def main() -> None:
    interactive = len(sys.argv) < 2
    if interactive:
        solve_from(sys.stdin, True)
        return

    path = sys.argv[1]
    try:
        with open(path, "r", encoding="utf-8") as src:
            solve_from(src, False)
    except OSError as e:
        print(f"Ошибка файла: {e}")


if __name__ == "__main__":
    main()
