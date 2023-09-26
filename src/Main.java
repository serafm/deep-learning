import java.nio.file.Path;
import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        MultilayerPerceptron mlp = new MultilayerPerceptron(1,3,1,1,1,"relu");
        mlp.LoadDataset(Path.of("data/train.csv"));
        ArrayList<ArrayList<String>> train = mlp.getData();

        mlp.LoadDataset(Path.of("data/test.csv"));
        ArrayList<ArrayList<String>> test = mlp.getData();

        System.out.println("Train dataset line 1: " + train.get(0));
        System.out.println("Test dataset line 1: " + test.get(0));
    }
}
