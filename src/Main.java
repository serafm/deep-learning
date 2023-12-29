public class Main {
    public static void main(String[] args) {
        MultilayerPerceptron mlp = new MultilayerPerceptron(2, 4, 40, 40, 30, "logistic", 0.1, 0.5, 40, 700);
        mlp.train();
        //mlp.test();
    }
}