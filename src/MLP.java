import java.io.File;

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

    private void LoadDatasets(File filename){
        // Load datasets (train/test) from file
        // Encode categories(K)

    }

    private void MLPArchitecture(){
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

}