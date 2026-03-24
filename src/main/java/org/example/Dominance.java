package org.example;

import java.math.BigDecimal;

import static org.example.Main.columnOrder;

public class Dominance {
    public static boolean makeDiagonalDominance(BigDecimal[][] A, BigDecimal[] b) {
        return permRows(0, A, b, new boolean[A.length]) || permCols(0, A);
    }

    public static boolean permRows(int r, BigDecimal[][] A, BigDecimal[] b, boolean[] used) {
        int n = A.length;
        if (r == n) return checkDominance(A);
        for (int i = 0; i < n; i++)
            if (!used[i]) {
                used[i] = true;
                swapRows(A, b, r, i);
                if (permRows(r + 1, A, b, used)) return true;
                swapRows(A, b, r, i);
                used[i] = false;
            }
        return false;
    }

    static void swapRows(BigDecimal[][] A, BigDecimal[] b, int i, int j) {
        BigDecimal[] t = A[i];
        A[i] = A[j];
        A[j] = t;
        BigDecimal tb = b[i];
        b[i] = b[j];
        b[j] = tb;
    }

    static boolean permCols(int c, BigDecimal[][] A) {
        int n = A.length;
        if (c == n) return checkDominance(A);
        for (int i = c; i < n; i++) {
            swapCols(A, c, i);
            int tmp = columnOrder[c];
            columnOrder[c] = columnOrder[i];
            columnOrder[i] = tmp;
            if (permCols(c + 1, A)) return true;
            swapCols(A, c, i);
            tmp = columnOrder[c];
            columnOrder[c] = columnOrder[i];
            columnOrder[i] = tmp;
        }
        return false;

    }

    public static void swapCols(BigDecimal[][] A, int i, int j) {
        for (int r = 0; r < A.length; r++) {
            BigDecimal t = A[r][i];
            A[r][i] = A[r][j];
            A[r][j] = t;
        }
    }

    public static boolean checkDominance(BigDecimal[][] A) {
        for (int i = 0; i < A.length; i++) {
            BigDecimal sum = BigDecimal.ZERO;
            for (int j = 0; j < A.length; j++){
                if (i != j) sum = sum.add(A[i][j].abs());
            }
            if (A[i][i].abs().compareTo(sum) < 0) return false;
        }
        return true;
    }
}
