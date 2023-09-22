import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class MLP {

    int d, K, H1, H2, H3;
    String actv_func;

    MLP(int d, int K, int H1, int H2, int H3, String actv_func){
        this.d = d;
        this.K = K;
        this.H1 = H1;
        this.H2 = H2;
        this.H3 = H3;
        this.actv_func = actv_func;
    }

    private void LoadDataset(File dataset_file){
        // Load datasets (train/test) from file
        // Encode categories(K)
        try {
            dataset_file = new File("train.txt");
            Scanner dataset = new Scanner(dataset_file);
            while (dataset.hasNextLine()) {
                String data_line = dataset.nextLine();
                System.out.println(data_line);
            }
            dataset.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

    }

    private void MLPArchitecture(float learningRate, float threshold){
        // Definition of required tables and others
        // structures as universal variables. Determining the rate of learning and the threshold of termination.
        // Random initialization of weights / polarizations in space (-1,1).
    }

    private void ForwardPass(float x, int d, float y, int K){
        // calculates the vector
        // output y (dimension K) of MLP given the input vector x (Dimension d)
    }

    private void Backpropagation(float x, int d, float t, int K){
        // takes vectors x
        // Dimension d (input) and T dimension K (desired output) and computes the derivatives of the error
        // as to any parameter (weight or polarization) of the network by updating the corresponding tables
    }

    private void gradientDescent(){

    }

}