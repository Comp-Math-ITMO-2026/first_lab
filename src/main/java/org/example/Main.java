package org.example;

import java.io.*;
import java.math.*;
import java.util.*;

import static org.example.Dominance.checkDominance;
import static org.example.Dominance.makeDiagonalDominance;
import static org.example.Reader.*;

public class Main {

    public static final MathContext MC = new MathContext(20, RoundingMode.HALF_UP);

    public static int[] columnOrder;

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.println("1 - ввод с клавиатуры");
        System.out.println("2 - ввод из файла");
        int mode = readInt(sc, "Выбор: ", 1, 2);

        InputData inputData = (mode == 1) ? inputKeyboard(sc) : inputFile(sc);

        columnOrder = new int[inputData.A.length];
        for (int i = 0; i < columnOrder.length; i++) columnOrder[i] = i;

        if (!checkDominance(inputData.A)) {
            if (!makeDiagonalDominance(inputData.A, inputData.b)) {
                System.out.println("Невозможно добиться диагонального преобладания");
                return;
            }
        }

        System.out.println("Норма матрицы = " + matrixNorm(inputData.A));

        Result result = gaussSeidel(inputData.A, inputData.b, inputData.eps);

        BigDecimal[] realX = new BigDecimal[result.x.length];
        for (int i = 0; i < realX.length; i++)
            realX[columnOrder[i]] = result.x[i];

        System.out.println("\nРешение:");
        for (int i = 0; i < realX.length; i++)
            System.out.println("x" + (i + 1) + " = " + realX[i]);

        System.out.println("\nИтераций: " + result.iterations);
        System.out.println("\nПогрешности:");
        for (int i = 0; i < result.errors.length; i++)
            System.out.println("|delta x" + (i + 1) + "| = " + result.errors[i]);
    }

    public static Result gaussSeidel(BigDecimal[][] A, BigDecimal[] b, BigDecimal eps) {
        int n = b.length;
        BigDecimal[] x = new BigDecimal[n];
        BigDecimal[] prev = new BigDecimal[n];
        BigDecimal[] errors = new BigDecimal[n];
        Arrays.fill(x, BigDecimal.ZERO);

        int iterations = 0;
        boolean stop;
        do {
            iterations++;
            System.arraycopy(x, 0, prev, 0, n);
            for (int i = 0; i < n; i++) {
                BigDecimal sum = b[i];
                for (int j = 0; j < n; j++) {
                    if (j != i) {
                        sum = sum.subtract(A[i][j].multiply(x[j], MC), MC);
                    }
                }
                x[i] = sum.divide(A[i][i], MC);
            }
            stop = true;
            for (int i = 0; i < n; i++) {
                errors[i] = x[i].subtract(prev[i], MC).abs();
                if (errors[i].compareTo(eps) > 0) stop = false;
            }
        } while (!stop && iterations < 10000);

        Result r = new Result();
        r.x = x;
        r.errors = errors;
        r.iterations = iterations;
        return r;
    }


    public static BigDecimal matrixNorm(BigDecimal[][] A) {
        BigDecimal max = BigDecimal.ZERO;
        for (BigDecimal[] row : A) {
            BigDecimal sum = BigDecimal.ZERO;
            for (BigDecimal element : row) sum = sum.add(element.abs());
            if (sum.compareTo(max) > 0) max = sum;
        }
        return max;
    }
}