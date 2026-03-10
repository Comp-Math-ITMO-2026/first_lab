package uitils;

import java.util.Arrays;

public class SimpleIterationMethod {

    private int matrixSize = 0;
    private Double[][] matrix;
    private Double eps;
    private Double[] dVector;

    public SimpleIterationMethod(int matrixSize, Double[][] matrix, Double eps, Double[] dVector) {
        this.matrixSize = matrixSize;
        this.matrix = matrix;
        this.eps = eps;
        this.dVector = dVector;
    }

    private Double determinant(Double[][] matrix) {
        if(matrix.length == 1) return matrix[0][0];
        if(matrix.length == 2) return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        double det = 0;
        for(int i = 0; i < matrix.length; i++) {
            det += Math.pow(-1, i) * matrix[0][i] * determinant(getMinor(matrix,0, i));
        }
        return det;
    }
    private Double[][] getMinor(Double[][] matrix, int row, int col) {
        Double[][] minor = new Double[matrix.length - 1][matrix.length - 1];
        for(int i = 0; i < matrix.length; i++) {
            for(int j = 0; j < matrix.length; j++) {
                if(i != row && j != col) {
                    minor[i < row ? i: i -1][j < col ? j : j - 1] = matrix[i][j];
                }
            }
        }
        return minor;
    }

    private boolean diagonalDominant() {
        for(int i = 0; i < this.matrixSize; i++) {
            int maxRow = i;
            double maxElement = Math.abs(this.matrix[i][i]);
            for(int j = i + 1; j < this.matrixSize; j++) {
                if(Math.abs(this.matrix[j][i]) > maxElement) {
                    maxElement = Math.abs(this.matrix[j][i]);
                    maxRow = j;
                }
            }
            if(maxRow != i) {
                Double[] tmpMatrix = matrix[i];
                matrix[i] = matrix[maxRow];
                matrix[maxRow] = tmpMatrix;

                double tmpD = dVector[i];
                dVector[i] = dVector[maxRow];
                dVector[maxRow] = tmpD;
            }
        }
        boolean isOneStrict = false;
        for(int i = 0; i < this.matrixSize; i++) {
            double summ = 0;
            for(int j = 0; j < this.matrixSize; j++) {
                if(i != j)  {
                    summ += Math.abs(this.matrix[i][j]);
                }
            }
            if(Math.abs(this.matrix[i][i]) < summ) {
                return false;
            }
            if(Math.abs(this.matrix[i][i]) > summ) {
                isOneStrict = true;
            }
        }
        System.out.println("Матрица после перестановок: ");
        for(int i = 0; i < this.matrixSize; i++) {
            for(int j = 0; j < this.matrixSize; j++) {
                System.out.format("%10.3f", this.matrix[i][j]);
            }
            System.out.println();
        }
        return isOneStrict;
    }

    private double getMaxRowNorm() {
        double normal = 0.0;
        for(int i = 0; i < this.matrixSize; i++) {
            double summ = 0;
            for(int j = 0; j < this.matrixSize; j++) {
                summ += Math.abs(this.matrix[i][j]);
            }
            normal = Math.max(normal, summ);
        }
        return normal;
    }

    public Double[] multiplyMatrix(Double[][] matrixA,  Double[] matrixB) {
        Double[] matrixC = new Double[matrixA.length];
        for(int i = 0; i < matrixA.length; i++) {
            matrixC[i] = 0.0;
            for(int j = 0; j < matrixA[i].length; j++) {
                matrixC[i] += matrixA[i][j] * matrixB[j];
            }
        }
        return matrixC;
    }

    public Double[] sumMatrix(Double[] matrixA,  Double[] matrixB) {
        Double[] matrixC = new Double[matrixA.length];
        for(int i = 0; i < matrixA.length; i++) {
            matrixC[i] = matrixA[i] + matrixB[i];
        }
        return matrixC;
    }

    private void magicFunction() {
        double diag = 0;
        for(int i = 0; i < this.matrixSize; i++) {
            diag = matrix[i][i];
            for(int j = 0; j < this.matrixSize; j++) {
                if(i == j) {
                    this.matrix[i][j] = 0.0;
                } else {
                    this.matrix[i][j] = -(this.matrix[i][j] / diag);
                }
            }
        }
        for(int i = 0; i < this.matrixSize; i++) {
            this.dVector[i] /= diag;
        }
        System.out.println("Матрица после вывода коэффициентов");
        for(int i = 0; i < this.matrixSize; i++) {
            for(int j = 0; j < this.matrixSize; j++) {
                System.out.format("%10.3f", this.matrix[i][j]);
            }
            System.out.println();
        }
        System.out.println("Ваша вектор D: ");
        for (int i = 0; i < this.matrixSize; ++i) {
            System.out.format("%10.3f\n", this.dVector[i]);
        }
    }

    public void startMethod() {
        if(determinant(matrix) < 1e-9) {
            System.out.println("Матрица вырождена, потому решений нет!");
            return;
        }
        if(!diagonalDominant()) {
            System.out.println("Диагонального преобладания в матрице нет :( плаки-плак");
            return;
        }
        this.magicFunction();
        if(getMaxRowNorm() >= 1.0) {
            System.out.println("Норма некорректная");
            return;
        }
        System.out.println("Норма матрицы: ||C_1|| = " + getMaxRowNorm());
        Double[] xStart = Arrays.copyOf(this.dVector, this.dVector.length);
        Double[] xLast;
        int k = 0;
        double maxDisp;
        do {
            k++;
            maxDisp= 0;
            Double[] xNew = sumMatrix(multiplyMatrix(this.matrix, xStart),  this.dVector);
            for(int i = 0; i < xStart.length; i++) {
                maxDisp = Math.max(maxDisp, Math.abs(xStart[i] - xNew[i]));
            }
            xLast = Arrays.copyOf(xStart, xStart.length);
            xStart = Arrays.copyOf(xNew, xNew.length);
        } while(maxDisp > this.eps);
        System.out.format("Ответ был найден спустя %d итераций\n", k);
        System.out.println("Вектор неизвестных: ");
        for(int i = 0; i < xLast.length; i++) {
            System.out.format("X_%d =  %10.6f\n", i+1, xStart[i]);
        }
        System.out.println("Вектор погрешностей");
        for(int i = 0; i < xLast.length; i++) {
            System.out.format("|X_%d^(%d) - X_%d^(%d)| =  %10.6f\n", i+1, k, i + 1, k-1, Math.abs(xLast[i] -  xStart[i]));
        }
    }
}
