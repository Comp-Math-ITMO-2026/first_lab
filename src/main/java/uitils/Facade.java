package uitils;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;

public class Facade {

    private int matrixSize;
    private Double[][] matrix;
    private double eps;
    private Double[] dVector;

    Scanner scanner = new Scanner(System.in);

    public Facade() {
        this.matrixSize = 0;
        this.eps = 0.0001;
    }

    private void matrixInput() {
        boolean isOk = false;
        do {
            System.out.println("Введите размер матрицы n (0 < n < 21)\n");
            System.out.print("-> ");
            String input;
            do {
                input = scanner.nextLine();
                try {
                    if (Integer.parseInt(input) > 0 && Integer.parseInt(input) < 21) {
                        this.matrixSize = Integer.parseInt(input);
                        break;
                    }
                } catch (NumberFormatException e) {
                    System.out.println("Некорректный формат ввода");
                    System.out.println("Введите размер матрицы n (0 < n < 21)\n");
                    System.out.print("-> ");
                }
            } while (true);

            this.matrix = new Double[this.matrixSize][this.matrixSize];
            int linesCounter = 0;

            System.out.println("Введите коэффициенты матрицы построчно через пробел");
            do {
                if(linesCounter == this.matrixSize) {
                    break;
                }
                System.out.format("Введите коэффициенты %d строки матрицы\n", linesCounter+1);
                System.out.print("-> ");
                input = scanner.nextLine();
                try {
                    String[] matrixValues = input.split(" ");
                    if(matrixValues.length != this.matrixSize) {
                        throw new ArrayIndexOutOfBoundsException();
                    }
                    for (int i = 0; i < this.matrixSize; ++i) {
                        this.matrix[linesCounter][i] = Double.parseDouble(matrixValues[i]);
                    }
                    linesCounter++;
                } catch (NumberFormatException e) {
                    System.out.println("Некорректный формат ввода");
                    System.out.println("Введите коэффициенты матрицы построчно через пробел");
                    System.out.format("Введите коэффициенты %d строки матрицы\n", linesCounter+1);
                    System.out.print("-> ");
                } catch(ArrayIndexOutOfBoundsException e) {
                    System.out.format("В строке должно быть ровно %d элементов\n", this.matrixSize);
                    System.out.println("Введите коэффициенты матрицы построчно через пробел");
                    System.out.format("Введите коэффициенты %d строки матрицы\n", linesCounter+1);
                    System.out.print("-> ");
                }
            } while (linesCounter < this.matrixSize);
            System.out.println("Ваша матрица: ");
            for (int i = 0; i < this.matrixSize; ++i) {
                for (int j = 0; j < this.matrixSize; ++j) {
                    System.out.printf("%10.3f", this.matrix[i][j]);
                }
                System.out.println();
            }
            System.out.println("Хотите заново ввести матрицу?:\n[1]: Ввести заново\n[2]: Оставить текущую");
            System.out.print("-> ");
            input = scanner.nextLine();
            if(input.equals("2")){
                break;
            }
        } while(!isOk);
    }

    private void dVectorInput() {
        boolean isOk = false;
        this.dVector = new Double[this.matrixSize];
        do {
            int linesCounter = 0;
            String input;
            System.out.println("Введите элементы вектора неизвестных построчно");
            do {
                if(linesCounter == this.matrixSize) {
                    break;
                }
                System.out.format("Введите коэффициенты %d строки матрицы\n", linesCounter+1);
                System.out.print("-> ");
                input = scanner.nextLine();
                try {
                    Double matrixValues = Double.parseDouble(input);
                    this.dVector[linesCounter] = matrixValues;
                    linesCounter++;
                } catch (NumberFormatException e) {
                    System.out.println("Некорректный формат ввода");
                    System.out.println("Введите коэффициенты матрицы построчно");
                    System.out.format("Введите коэффициенты %d строки матрицы\n", linesCounter+1);
                    System.out.print("-> ");
                }
            } while (linesCounter < this.matrixSize);
            System.out.println("Ваша вектор D: ");
            for (int i = 0; i < this.matrixSize; ++i) {
                System.out.format("%10.3f\n", this.dVector[i]);
            }
            System.out.println("Хотите заново ввести матрицу?:\n[1]: Ввести заново\n[2]: Оставить текущую");
            System.out.print("-> ");
            input = scanner.nextLine();
            if(input.equals("2")){
                break;
            }
        } while(!isOk);
    }

    private void epsilonInput() {
        System.out.println("Введите желаемую точность вычислений. (Например: 0.001): ");
        System.out.print("-> ");
        String input;
        do {
            input = scanner.nextLine();
            try {
                eps = Double.parseDouble(input);
                break;
            } catch (NumberFormatException e) {
                System.out.println("Введена неккоректная точность");
                System.out.println("Введите желаемую точность вычислений. (Например: 0.001): ");
                System.out.print("-> ");
            }
        } while(true);
    }

    private void readFromFile() {
        BufferedReader reader;
        try {
            reader = new BufferedReader(new FileReader("matrix.txt"));
            String line = reader.readLine();
            while(line != null) {
                this.matrixSize = Integer.parseInt(line);
                this.matrix = new Double[this.matrixSize][this.matrixSize];
                for(int k = 0; k < this.matrixSize; ++k) {
                    line = reader.readLine();
                    String[] matrixValues = line.split(" ");
                    if (matrixValues.length != this.matrixSize) {
                        throw new ArrayIndexOutOfBoundsException();
                    }
                    for (int i = 0; i < matrixValues.length; ++i) {
                        this.matrix[k][i] = Double.parseDouble(matrixValues[i]);
                    }
                }
                this.dVector = new Double[this.matrixSize];
                for(int k = 0; k < this.matrixSize; ++k) {
                    line = reader.readLine();
                    this.dVector[k] = Double.parseDouble(line);
                }
                line = reader.readLine();
                this.eps =  Double.parseDouble(line);
            }
            reader.close();
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (NumberFormatException e) {
            System.out.println("Некорректный формат ввода");
        } catch(ArrayIndexOutOfBoundsException e) {
            System.out.format("В строке должно быть ровно %d элементов\n", this.matrixSize);
        }
    }

    public void run() {
        System.out.println("Выберите режим ввода: \n[1]: Клавиатура\n[2]: с Файла\n");
        System.out.print("-> ");
        String input = scanner.nextLine();
        do {
            if(input.equals("1")) {
                this.matrixInput();
                this.dVectorInput();
                this.epsilonInput();
                SimpleIterationMethod simpleIterationMethod = new SimpleIterationMethod(this.matrixSize, this.matrix, this.eps, this.dVector);
                simpleIterationMethod.startMethod();
            } else if(input.equals("2")) {
                System.out.println("Система считывания файла активирована...");
                readFromFile();
                SimpleIterationMethod simpleIterationMethod = new SimpleIterationMethod(this.matrixSize, this.matrix, this.eps, this.dVector);
                simpleIterationMethod.startMethod();
            } else {
                System.out.println("Выберите корректный режим ввода");
                System.out.println("Выберите режим ввода: \n[1]: Клавиатура\n[2]: с Файла\n");
                System.out.print("-> ");
            }
            input = scanner.nextLine();
        } while(input != null && !input.isEmpty());
    }
}
