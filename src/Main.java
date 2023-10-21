public class Main {
    public static void main(String[] args) {
        MultilayerPerceptron mlp = new MultilayerPerceptron(2,3,4,2,4,"logistic");
        mlp.MLPArchitecture(0.1F, 0.1F, 40, 700);
        mlp.train();
        mlp.test();
    }
}