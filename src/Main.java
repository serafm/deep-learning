import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {
        MultilayerPerceptron mlp = new MultilayerPerceptron(2, 4, 38, 32, 32, "logistic", 0.1, 0.1, 40, 700);
        mlp.train();
        mlp.test();
        mlp.saveResultToCsv();
    }
}