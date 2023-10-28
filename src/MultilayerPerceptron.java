import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.Path;
import java.lang.Math;
import java.util.Random;
import java.util.Scanner;

public class MultilayerPerceptron {

    private int numOfInputs;
    private int numOfCategories;
    private int numOfHidden1;
    private int numOfHidden2;
    private int numOfHidden3;
    private int epochs_size;
    private int batch_size;
    private String activationFunction;
    private Datapoint[] inputData;
    private final int trainSize = 4000;

    private float[][] hidden1Weights;
    private float[][] hidden2Weights;
    private float[][] hidden3Weights;
    private float[][] outputWeights;
    private float[] hidden1BiasWeights;
    private float[] hidden2BiasWeights;
    private float[] hidden3BiasWeights;
    private float[] outputBiasWeights;
    private float[] hidden1Outputs;
    private float[] hidden2Outputs;
    private float[] hidden3Outputs;
    private float[] outputs;
    private OutputCategory[] outputCategoryPerInput;
    private float learningRate;
    private float threshold;

    private float[] deltaOfHidden3;
    private float[] deltaOfHidden2;
    private float[] deltaOfHidden1;
    private float[] deltaOfOutput;

    private float[][] outputWeightsDerivatives;
    private float[][] hidden3WeightsDerivatives;
    private float[][] hidden2WeightsDerivatives;
    private float[][] hidden1WeightsDerivatives;
    private float[] outputBiasDerivatives;
    private float[] hidden3BiasDerivatives;
    private float[] hidden2BiasDerivatives;
    private float[] hidden1BiasDerivatives;
    private float[] totalInputOfOutputLayer;
    private float[] totalInputOfHidden3Layer;
    private float[] totalInputOfHidden2Layer;
    private float[] totalInputOfHidden1Layer;
    private float[][] outputWeightsPartialSum;
    private float[][] hidden3WeightsPartialSum;
    private float[][] hidden2WeightsPartialSum;
    private float[][] hidden1WeightsPartialSum;

    private ActivationFunctions actvFunc = new ActivationFunctions();

    /**
     * Multilayer perceptron constructor initialization.
     * @param d input data dimension
     * @param K number of output categories
     * @param H1 number of neurons in 1 hidden layer
     * @param H2 number of neurons in 2 hidden layer
     * @param H3 number of neurons in 3 hidden layer
     * @param actv activation function
     * @param learningRate learning rate
     * @param threshold threshold
     * @param batch_size size of batches
     * @param epochs_size size of epochs
     */
    MultilayerPerceptron(int d, int K, int H1, int H2, int H3, String actv, float learningRate, float threshold, int batch_size, int epochs_size) {
        this.numOfInputs = d;
        this.numOfCategories = K;
        this.numOfHidden1 = H1;
        this.numOfHidden2 = H2;
        this.numOfHidden3 = H3;
        this.activationFunction = actv;
        this.learningRate = learningRate;
        this.threshold = threshold;
        this.batch_size = batch_size;
        this.epochs_size = epochs_size;
    }

    /**
     * Load datasets (train/test) from file and add the data into structures.
     * Encode categories(K)
     * @param datapath path of csv file
     */
    public void LoadDataset(Path datapath){
        inputData = new Datapoint[this.trainSize];
        int row = 0;

        try {
            File train_dataset = new File(String.valueOf(datapath));
            Scanner dataset = new Scanner(train_dataset);
            while (dataset.hasNextLine()) {
                String[] data_line = dataset.nextLine().split(",");

                if (data_line[2].equals("C1")){
                    data_line[2] = "0";
                } else if (data_line[2].equals("C2")){
                    data_line[2] = "1";
                } else if (data_line[2].equals("C3")) {
                    data_line[2] = "2";
                }

                float x1 = Float.valueOf(data_line[0]);
                float x2 = Float.valueOf(data_line[1]);
                int category = Integer.valueOf(data_line[2]);

                inputData[row] = new Datapoint(x1, x2, category);

                row++;
            }
            dataset.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    /**
     * Random initialization of weights in (-1,1).
     */
    private void initializeWeights() {
        Random random = new Random();

        this.outputCategoryPerInput = new OutputCategory[inputData.length];
        this.hidden1Weights = new float[numOfInputs][numOfHidden1];
        this.hidden2Weights = new float[numOfHidden1][numOfHidden2];
        this.hidden3Weights = new float[numOfHidden2][numOfHidden3];
        this.outputWeights = new float[numOfHidden3][numOfCategories];
        this.hidden1BiasWeights = new float[numOfHidden1];
        this.hidden2BiasWeights = new float[numOfHidden2];
        this.hidden3BiasWeights = new float[numOfHidden3];
        this.outputBiasWeights = new float[numOfCategories];
        this.hidden1Outputs = new float[numOfHidden1];
        this.hidden2Outputs = new float[numOfHidden2];
        this.hidden3Outputs = new float[numOfHidden3];
        this.outputs = new float[numOfCategories];

        for (int i = 0; i < numOfInputs; i++) {
            for (int j = 0; j < numOfHidden1; j++) {
                this.hidden1BiasWeights[j] = random.nextFloat(-1, 1);
                this.hidden1Weights[i][j] = random.nextFloat(-1, 1); // Random weight between -1 and 1
            }
        }
        for (int i = 0; i < numOfHidden1; i++) {
            for (int j = 0; j < numOfHidden2; j++) {
                this.hidden2BiasWeights[j] = random.nextFloat(-1, 1);
                this.hidden2Weights[i][j] = random.nextFloat(-1, 1); // Random weight between -1 and 1
            }
        }
        for (int i = 0; i < numOfHidden2; i++) {
            for (int j = 0; j < numOfHidden3; j++) {
                this.hidden3BiasWeights[j] = random.nextFloat(-1, 1);
                this.hidden3Weights[i][j] = random.nextFloat(-1, 1); // Random weight between -1 and 1
            }
        }
        for (int i = 0; i < numOfHidden3; i++) {
            for (int j = 0; j < numOfCategories; j++) {
                this.outputBiasWeights[j] = random.nextFloat(-1, 1);
                this.outputWeights[i][j] = random.nextFloat(-1, 1); // Random weight between -1 and 1
            }
        }
    }

    /**
     * Train the network with gradient descent method. Update weights in batches or serial.
     */
    private void GradientDescent(){
        int t = 0;
        float previousErrorRate;
        float difference = Float.POSITIVE_INFINITY;
        float error;
        float mse = Float.POSITIVE_INFINITY;

        while(t<epochs_size || difference > threshold) {
            int n = 0;
            previousErrorRate = mse;
            error = 0;
            outputWeightsPartialSum = new float[numOfCategories][numOfHidden3];
            hidden3WeightsPartialSum = new float[numOfHidden3][numOfHidden2];
            hidden2WeightsPartialSum = new float[numOfHidden2][numOfHidden1];
            hidden1WeightsPartialSum = new float[numOfHidden1][numOfInputs];
            if(batch_size>1 && (inputData.length % batch_size == 0)) {
                for(int k=1; k<(inputData.length/batch_size)+1; k++) {
                    for (int b=0; b<batch_size; b++) {
                        ForwardPass(n);
                        error += LossFunction(n);
                        Backpropagation(n);

                        for (int i = 0; i < numOfCategories; i++) {
                            for (int j = 0; j < numOfHidden3; j++) {
                                outputWeightsPartialSum[i][j] += outputWeightsDerivatives[i][j];
                            }
                        }
                        for (int i = 0; i < numOfHidden3; i++) {
                            for (int j = 0; j < numOfHidden2; j++) {
                                hidden3WeightsPartialSum[i][j] += hidden3WeightsDerivatives[i][j];
                            }
                        }
                        for (int i = 0; i < numOfHidden2; i++) {
                            for (int j = 0; j < numOfHidden1; j++) {
                                hidden2WeightsPartialSum[i][j] += hidden2WeightsDerivatives[i][j];
                            }
                        }
                        for (int i = 0; i < numOfHidden1; i++) {
                            for (int j = 0; j < numOfInputs; j++) {
                                hidden1WeightsPartialSum[i][j] += hidden1WeightsDerivatives[i][j];
                            }
                        }
                        n += 1;
                    }
                }
                // Group Update of weights
                groupUpdateWeights();

            } else {
                // if Batch size=1. Serial update of weights
                for (int p = 0; p < inputData.length; p++) {
                    ForwardPass(p);
                    error += LossFunction(p);
                    Backpropagation(p);
                    serialUpdateWeights();
                }
            }

            System.out.println("Epoch " + t  + "\n" +  "Loss Function = " + mse);
            mse = error /inputData.length;
            difference = Math.abs(previousErrorRate - mse);
            t += 1;
        }
    }

    private void serialUpdateWeights(){
        for(int i = 0; i < numOfHidden3; i++){
            for (int j = 0; j < numOfCategories; j++) {
                outputWeights[i][j] = outputWeights[i][j] - learningRate * outputWeightsDerivatives[j][i];
            }
        }
        for(int i = 0; i < numOfHidden2; i++) {
            for (int j = 0; j < numOfHidden3; j++) {
                hidden3Weights[i][j] = hidden3Weights[i][j] - learningRate * hidden3WeightsDerivatives[j][i];
            }
        }
        for(int i = 0; i < numOfHidden1; i++) {
            for (int j = 0; j < numOfHidden2; j++) {
                hidden2Weights[i][j] = hidden2Weights[i][j] - learningRate * hidden2WeightsDerivatives[j][i];
            }
        }
        for(int i = 0; i < numOfInputs; i++) {
            for (int j = 0; j < numOfHidden1; j++) {
                hidden1Weights[i][j] = hidden1Weights[i][j] - learningRate * hidden1WeightsDerivatives[j][i];
            }
        }
    }

    /**
     * Group update of weights
     */
    private void groupUpdateWeights(){
        //System.out.println("\nNew output weights: ");
        for(int i = 0; i < numOfHidden3; i++) {
            for (int j = 0; j < numOfCategories; j++) {
                outputWeights[i][j] = outputWeights[i][j] - learningRate * outputWeightsPartialSum[j][i];
                //System.out.println(outputWeights[i][j]);
            }
        }
        //System.out.println("\nNew hidden 3 weights: ");
        for(int i = 0; i < numOfHidden2; i++) {
            for (int j = 0; j < numOfHidden3; j++) {
                hidden3Weights[i][j] = hidden3Weights[i][j] - learningRate * hidden3WeightsPartialSum[j][i];
                //System.out.println(hidden3Weights[i][j]);
            }
        }
        //System.out.println("\nNew hidden 2 weights: ");
        for(int i = 0; i < numOfHidden1; i++) {
            for (int j = 0; j < numOfHidden2; j++) {
                hidden2Weights[i][j] = hidden2Weights[i][j] - learningRate * hidden2WeightsPartialSum[j][i];
                //System.out.println(hidden2Weights[i][j]);
            }
        }
        //System.out.println("\nNew hidden 1 weights: ");
        for(int i = 0; i < numOfInputs; i++) {
            for (int j = 0; j < numOfHidden1; j++) {
                hidden1Weights[i][j] = hidden1Weights[i][j] - learningRate * hidden1WeightsPartialSum[j][i];
                //System.out.println(hidden1Weights[i][j]);
            }
        }
    }

    /**
     * Forward pass of each input data and classification.
     * @param data_point input data point
     */
    private void ForwardPass(int data_point){
        // calculates the vector
        // output y (dimension K) of MLP given the input vector x (Dimension d)
        totalInputOfOutputLayer = new float[numOfCategories];
        totalInputOfHidden3Layer = new float[numOfHidden3];
        totalInputOfHidden2Layer = new float[numOfHidden2];
        totalInputOfHidden1Layer = new float[numOfHidden1];

        for(int i=0; i<numOfHidden1; i++) {
            hidden1Outputs[i] = inputData[data_point].getX1() * hidden1Weights[0][i] + inputData[data_point].getX2() * hidden1Weights[1][i];
            totalInputOfHidden1Layer[i] = hidden1Outputs[i] + hidden1BiasWeights[i];
            hidden1Outputs[i] = actvFunc.selectActivationFunction(totalInputOfHidden1Layer[i], activationFunction);
        }

        for(int i=0; i<numOfHidden2; i++){
            for(int j=0; j<numOfHidden1; j++){
                hidden2Outputs[i] += hidden1Outputs[j]*hidden2Weights[j][i];
            }
            totalInputOfHidden2Layer[i] = hidden2Outputs[i] + hidden2BiasWeights[i];
            hidden2Outputs[i] = actvFunc.selectActivationFunction(totalInputOfHidden2Layer[i], activationFunction);
        }
        for(int i=0; i<numOfHidden3; i++){
            for(int j=0; j<numOfHidden2; j++){
                hidden3Outputs[i] += hidden2Outputs[j]*hidden3Weights[j][i];
            }
            totalInputOfHidden3Layer[i] = hidden3Outputs[i] + hidden3BiasWeights[i];
            hidden3Outputs[i] = actvFunc.selectActivationFunction(totalInputOfHidden3Layer[i], activationFunction);
        }
        for(int i=0; i<numOfCategories; i++){
            for(int j=0; j<this.numOfHidden3; j++){
                outputs[i] += hidden3Outputs[j]*outputWeights[j][i];
            }
            totalInputOfOutputLayer[i] = outputs[i] + outputBiasWeights[i];
            outputs[i] = actvFunc.tanh(totalInputOfOutputLayer[i]);
        }
        outputCategoryPerInput[data_point] = new OutputCategory(classification(outputs));
    }

    /**
     * Backpropagation method, delta calculation of each layer neurons.
     * @param i input data point
     */
    private void Backpropagation(int i){
        deltaOutput(i);
        deltaHiddenLayer3();
        deltaHiddenLayer2();
        deltaHiddenLayer1(i);
    }

    /**
     * Classify each input data point to a category.
     * @param outputs array of category outputs
     * @return category of input data
     */
    public int classification(float[] outputs) {
        // Initialize max to the first element of the array
        float max = 0F;
        int category = 0;
        // Iterate through the array to find the maximum value
        for (int i=0; i < numOfCategories; i++) {
            if (outputs[i] > max) {
                max = outputs[i];
                category = i;
            }
        }
        return category;
    }

    /**
     * Delta calculation of the output layer neurons.
     * Calculation of weights derivatives.
     * δ_i = g(u_i)*Σ(w_ji*δ_j)
     * @param inputDataId input data point
     */
    public void deltaOutput(int inputDataId){
        outputWeightsDerivatives = new float[numOfCategories][numOfHidden3];
        outputBiasDerivatives = new float[numOfCategories];
        deltaOfOutput = new float[numOfCategories];
        for(int i=0; i<numOfCategories; i++){
            float g = actvFunc.selectBackpropagationDerivative(totalInputOfOutputLayer[i], activationFunction);
            deltaOfOutput[i] = g*(outputCategoryPerInput[inputDataId].getOutputCategory() - inputData[inputDataId].getCategory());
            for(int j=0; j<numOfHidden3; j++){
                outputWeightsDerivatives[i][j] = deltaOfOutput[i]*hidden3Outputs[j];
            }
            outputBiasDerivatives[i] = deltaOfOutput[i];
        }
    }

    /**
     * Delta calculation of the 3 hidden layer neurons.
     * Calculation of weights derivatives.
     * δ_i = g(u_i)*Σ(w_ji*δ_j)
     */
    public void deltaHiddenLayer3(){
        this.hidden3WeightsDerivatives = new float[numOfHidden3][numOfHidden2];
        this.hidden3BiasDerivatives = new float[numOfHidden3];
        this.deltaOfHidden3 = new float[numOfHidden3];
        // list of output layer weights
        for(int i=0; i<numOfHidden3; i++){ // for each neuron
            float g = actvFunc.selectBackpropagationDerivative(totalInputOfHidden3Layer[i], activationFunction); // derivative of neuron output
            float sumOfOutputLayerNeuronsDelta = 0F; // initialize summation of weight*delta of each output neuron
            for(int j=0; j<numOfCategories; j++){
                sumOfOutputLayerNeuronsDelta += outputWeights[i][j]*deltaOfOutput[j];
            }
            deltaOfHidden3[i] = g*sumOfOutputLayerNeuronsDelta;; // add it in list of neuron deltas
            for(int j=0; j<this.numOfHidden2; j++) {
                hidden3WeightsDerivatives[i][j] = deltaOfHidden3[i] * hidden2Outputs[j];
                hidden3BiasDerivatives[i] = deltaOfHidden3[i];
            }
        }
    }

    /**
     * Delta calculation of the 2 hidden layer neurons.
     * Calculation of weights derivatives.
     * δ_i = g(u_i)*Σ(w_ji*δ_j)
     */
    public void deltaHiddenLayer2(){
        hidden2WeightsDerivatives = new float[numOfHidden2][numOfHidden1];
        hidden2BiasDerivatives = new float[numOfHidden2];
        deltaOfHidden2 = new float[this.numOfHidden2]; // list of delta values of neurons
        for(int i=0; i<numOfHidden2; i++){ // for each neuron
            float g = actvFunc.selectBackpropagationDerivative(totalInputOfHidden2Layer[i], activationFunction); // derivative of neuron output
            float sumOfOutputLayerNeuronsDelta = 0F; // initialize summation of weight*delta of each output neuron
            for(int j=0; j<numOfHidden3; j++){
                sumOfOutputLayerNeuronsDelta += hidden3Weights[i][j]*deltaOfHidden3[j]; // CHECK SWAP [i][j]
            }
            deltaOfHidden2[i] = g*sumOfOutputLayerNeuronsDelta;; // add it in list of neuron deltas
            for(int j=0; j<numOfHidden1; j++) {
                hidden2WeightsDerivatives[i][j] = deltaOfHidden2[i]*hidden1Outputs[j];
                hidden2BiasDerivatives[i] = deltaOfHidden2[i];
            }
        }
    }

    /**
     * Delta calculation of the 1 hidden layer neurons.
     * Calculation of weights derivatives.
     * δ_i = g(u_i)*Σ(w_ji*δ_j)
     * @param inputDataId input data point
     */
    public void deltaHiddenLayer1(int inputDataId){
        hidden1WeightsDerivatives = new float[numOfHidden1][numOfInputs];
        hidden1BiasDerivatives = new float[numOfHidden1];
        deltaOfHidden1 = new float[this.numOfHidden1]; // list of delta values of neurons
        for(int i=0; i<numOfHidden1; i++){ // for each neuron
            float g = actvFunc.selectBackpropagationDerivative(totalInputOfHidden1Layer[i], activationFunction); // derivative of neuron output
            float sumOfOutputLayerNeuronsDelta = 0F; // initialize summation of weight*delta of each output neuron
            for(int j=0; j<numOfHidden2; j++){
                sumOfOutputLayerNeuronsDelta += hidden2Weights[i][j]*deltaOfHidden2[j];
            }
            deltaOfHidden1[i] = g*sumOfOutputLayerNeuronsDelta;; // add it in list of neuron deltas
            hidden1WeightsDerivatives[i][0] = deltaOfHidden1[i] * inputData[inputDataId].getX1();
            hidden1WeightsDerivatives[i][1] = deltaOfHidden1[i] * inputData[inputDataId].getX2();
            hidden1BiasDerivatives[i] = deltaOfHidden1[i];
        }
    }

    /**
     * Loss function calculation of actual and predicted output
     * E = (target - output)^2
     * @param data_point input data point
     * @return classification error
     */
    public int LossFunction(int data_point) {
        return (int) Math.pow((inputData[data_point].getCategory() - outputCategoryPerInput[data_point].getOutputCategory()), 2);
    }

    /**
     * Calculation of the ability(%) of the MLP to generalize for unseen data.
     */
    public void generalizationAbility(){
        int counter = 0;
        for(int i=0; i<inputData.length; i++){
            if( inputData[i].getCategory() == outputCategoryPerInput[i].getOutputCategory()){
                counter += 1;
            }
        }
        Float result = (float) counter / inputData.length;
        System.out.println("Generalization Ability: " + result*100 + "%");
    }

    /**
     * Test dataset and generalization ability of MLP
     */
    public void test(){
        LoadDataset(Path.of("data/test.csv"));
        generalizationAbility();
    }

    /**
     * Training of MLP
     */
    public void train(){
        LoadDataset(Path.of("data/train.csv"));
        initializeWeights();
        GradientDescent();
    }
}