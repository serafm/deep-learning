import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Scanner;

public class MLP {

    int d, K, H1, H2, H3;
    String actv_func;
    public ArrayList<ArrayList<String>> data = new ArrayList<>();

    MLP(int d, int K, int H1, int H2, int H3, String actv_func){
        this.d = d;
        this.K = K;
        this.H1 = H1;
        this.H2 = H2;
        this.H3 = H3;
        this.actv_func = actv_func;
    }


    public void LoadDataset(Path datapath){
        // Load datasets (train/test) from file
        // Encode categories(K)
        this.data = new ArrayList<>();

        try {
            File train_dataset = new File(String.valueOf(datapath));
            Scanner dataset = new Scanner(train_dataset);
            while (dataset.hasNextLine()) {
                String[] data_line = dataset.nextLine().split(",");
                ArrayList<String> line = new ArrayList<>();

                for(int i=0; i<data_line.length; i++) {
                    if (data_line[i].equals("C1")){
                        data_line[i] = "0";
                    } else if (data_line[i].equals("C2")){
                        data_line[i] = "1";
                    } else if (data_line[i].equals("C3")) {
                        data_line[i] = "2";
                    }
                    line.add(data_line[i]);
                }
                data.add(line); // Add the entire line to the data ArrayList
            }
            dataset.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    public ArrayList<ArrayList<String>> getData(){
        return data;
    }

    private void MLPArchitecture(float learningRate, float threshold){
        // Definition of required tables and others
        // structures as universal variables. Determining the rate of learning and the threshold of termination.
        // Random initialization of weights / polarizations in space (-1,1).

        Float[][] h1_weigths = null;
        Float[][] h2_weigths = null;
        Float[][] h3_weigths = null;
        Float[][] h1_inputs = null;
        Float[][] h2_inputs = null;
        Float[][] h3_inputs = null;
        Float[][] h1_outputs = null;
        Float[][] h2_outputs = null;
        Float[][] h3_outputs = null;

    }

    private void ForwardPass(float[] x, int d, float y, int K){
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