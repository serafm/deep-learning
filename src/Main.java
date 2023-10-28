public class Main {
    public static void main(String[] args) {
        MultilayerPerceptron mlp = new MultilayerPerceptron(2,3,4,8,4,"logistic",0.1F, 0.1F, 1, 700);
        mlp.train();
        mlp.test();
    }
}