package org.example;

import java.io.File;
import java.math.BigDecimal;
import java.util.Scanner;

public class Reader {
    static final int MAX_N = 20;

    public static int readInt(Scanner sc, String m, int a, int b) {
        while (true) {
            try {
                System.out.print(m);
                int value = sc.nextInt();
                if (value < a || value > b) throw new Exception();
                return value;
            } catch (Exception e) {
                System.out.println("Ошибка ввода");
                sc.nextLine();
            }
        }
    }


    public static BigDecimal readBD(Scanner sc, String m) {
        while (true) {
            try {
                System.out.print(m);
                return sc.nextBigDecimal();
            } catch (Exception e) {
                System.out.println("Ошибка числа");
                sc.nextLine();
            }
        }
    }

    public static InputData inputKeyboard(Scanner sc) {
        InputData d = new InputData();
        int n = Reader.readInt(sc, "n: ", 1, MAX_N);
        d.A = new BigDecimal[n][n];
        d.b = new BigDecimal[n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) d.A[i][j] = Reader.readBD(sc, "A[" + (i + 1) + "][" + (j + 1) + "]=");
        for (int i = 0; i < n; i++) d.b[i] = Reader.readBD(sc, "b[" + (i + 1) + "]=");
        d.eps = Reader.readBD(sc, "eps=");
        return d;
    }

    public static InputData inputFile(Scanner sc) {
        while (true) {
            try {
                System.out.print("Имя файла: ");
                Scanner f = new Scanner(new File(sc.next()));
                InputData d = new InputData();
                int n = f.nextInt();
                d.A = new BigDecimal[n][n];
                d.b = new BigDecimal[n];
                for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) d.A[i][j] = f.nextBigDecimal();
                for (int i = 0; i < n; i++) d.b[i] = f.nextBigDecimal();
                d.eps = f.nextBigDecimal();
                return d;
            } catch (Exception e) {
                System.out.println("Ошибка файла");
            }
        }
    }

    public static class InputData {
        BigDecimal[][] A;
        BigDecimal[] b;
        BigDecimal eps;
    }

    public static class Result {
        BigDecimal[] x, errors;
        int iterations;
    }
}
