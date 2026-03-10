package uitils;

import java.util.List;

public class FileReader {

    public List<List<Double>> readFile(String filename) {
        try {
            try(java.io.FileReader fr = new java.io.FileReader(filename)) {
                char[] a = new char[200];
                fr.read(a);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    };
}
