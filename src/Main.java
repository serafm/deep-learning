import java.io.IOException;

public class Main {
    public static void experiments(int h1, int h2, int h3, String actv, int b) throws IOException {
        MultilayerPerceptron mlp = new MultilayerPerceptron(2,4,h1,h2,h3,actv,0.1F, 0.1F, b, 700);
        mlp.train();
        mlp.test();
        mlp.saveResultToCsv();
    }

    public static void main(String[] args) throws IOException {
        experiments(18,20,20,"logistic",400);
        /*int[] hiddenLayers = {24,24,24};
        String[] actvFunc = {"logistic"};
        int[] B = {40};

        for(int i=0; i<1; i++){
            for(int b: B){
                for(String actv: actvFunc){
                    for(int h1: hiddenLayers){
                        for(int h2: hiddenLayers){
                            for(int h3: hiddenLayers){
                                experiments(h1,h2,h3,actv,b);
                            }
                        }
                    }
                }
            }
        }*/
    }
}