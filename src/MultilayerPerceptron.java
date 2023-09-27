import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.Path;
import java.lang.Math;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class MultilayerPerceptron {

    private int numOfInputs;
    private int numOfCategories;
    private int numOfHidden1;
    private int numOfHidden2;
    private int numOfHidden3;
    private String activationFunction;
    private float[][] inputData;
    private int trainSize = 4000;

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
    private MLPParameters[] parametersPerInput;
    private float learningRate;
    private float threshold;

    MultilayerPerceptron(int d, int K, int H1, int H2, int H3, String actv){
        this.numOfInputs = d;
        this.numOfCategories = K;
        this.numOfHidden1 = H1;
        this.numOfHidden2 = H2;
        this.numOfHidden3 = H3;
        this.activationFunction = actv;
    }

    public float logistic(float x){
        return (float) (1/(1+Math.exp(Double.valueOf(-x))));
    }

    public float tanh(float x){
        return (float) ((Math.exp(Double.valueOf(x) - Math.exp(Double.valueOf(-x))))
                        / (Math.exp(Double.valueOf(x) + Math.exp(Double.valueOf(-x)))));
    }

    public float relu(float x){
        return Math.max(0,x);
    }

    public void LoadDataset(Path datapath){
        // Load datasets (train/test) from file
        // Encode categories(K)
        this.inputData = new float[this.trainSize][3];
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

                for(int col=0; col<data_line.length; col++) {
                    inputData[row][col] = Float.valueOf(data_line[col]);
                }
                row++;
            }
            dataset.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    // Initialize weights randomly (-1,1)
    private void initializeWeights() {
        Random random = new Random();

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

    private void MLPArchitecture(float learningRate, float threshold){
        // Definition of required tables and others
        // structures as universal variables. Determining the rate of learning and the threshold of termination.
        // Random initialization of weights / polarizations in space (-1,1).
        initializeWeights();
        this.learningRate = learningRate;
        this.threshold = threshold;

    }

    private void ForwardPass(float[][] x, int d, float[] y, int K){
        // calculates the vector
        // output y (dimension K) of MLP given the input vector x (Dimension d)
        this.parametersPerInput = new MLPParameters[this.inputData.length];

        for(int i=0; i<inputData.length; i++){
            for(int h=0; h<hidden1Outputs.length; h++){
                for(int j=0; j<d; j++){
                    hidden1Outputs[h] += inputData[i][j]*hidden1Weights[i][j];
                }
                hidden1Outputs[h] = selectActivationFunction(hidden1Outputs[h] + hidden1BiasWeights[h]);
            }
            for(int h=0; h<hidden2Outputs.length; h++){
                for(int j=0; j<this.numOfHidden1; j++){
                    hidden2Outputs[h] += hidden1Outputs[j]*hidden2Weights[i][j];
                }
                hidden2Outputs[h] = selectActivationFunction(hidden1Outputs[h] + hidden2BiasWeights[h]);
            }
            for(int h=0; h<hidden3Outputs.length; h++){
                for(int j=0; j<this.numOfHidden2; j++){
                    hidden3Outputs[h] += hidden2Outputs[j]*hidden3Weights[i][j];
                }
                hidden3Outputs[h] = selectActivationFunction(hidden1Outputs[h] + hidden3BiasWeights[h]);
            }
            for(int h=0; h<outputs.length; h++){
                for(int j=0; j<this.numOfHidden3; j++){
                    outputs[h] += hidden3Outputs[j]*outputWeights[i][j];
                }
                outputs[h] = logistic(outputs[h] + outputBiasWeights[h]);
            }

            MLPParameters mlpParameters = new MLPParameters(hidden1Weights, hidden2Weights, hidden3Weights,
                    outputWeights, hidden1BiasWeights, hidden2BiasWeights, hidden3BiasWeights, outputBiasWeights,
                    hidden1Outputs, hidden2Outputs, hidden3Outputs, outputs);

            parametersPerInput[i] = mlpParameters;
        }
    }

    private void Backpropagation(float x, int d, float t, int K){
        // takes vectors x
        // Dimension d (input) and T dimension K (desired output) and computes the derivatives of the error
        // as to any parameter (weight or polarization) of the network by updating the corresponding tables


    }

    private void train(int batch_size){

    }

    public float selectActivationFunction(float x){
        switch(activationFunction) {
            case "tanh":
                return tanh(x);
            case "relu":
                return relu(x);
            case "logistic":
                return logistic(x);
            default:
                return logistic(x);
        }
    }

    public float selectBackpropagationDerivative(float x){
        switch(activationFunction) {
            case "tanh":
                return derivativeTanh(x);
            case "relu":
                return derivativeRelu(x);
            case "logistic":
                return derivativeLogistic(x);
            default:
                return derivativeLogistic(x);
        }
    }

    public float derivativeTanh(float x){
        return (float) (1 - Math.pow(tanh(x), 2));
    }

    public float derivativeRelu(float x){
        if (x < 0){
            return 0F;
        }
        return 1F;
    }

    public float derivativeLogistic(float x){
        return (float) (Math.exp(-x)/Math.pow((1 + Math.exp(-x)), 2));
    }

    public int classification(float[] outputs) {
        // Initialize max to the first element of the array
        float max = outputs[0];
        int category = 0;
        // Iterate through the array to find the maximum value
        for (int i=0; i < outputs.length; i++) {
            if (outputs[i] > max) {
                max = outputs[i];
                category = i;
            }
        }
        return category;
    }

    public float deltaOutput(int outputNeuron, int inputDataId){
        float outputLayerOutput = this.parametersPerInput[inputDataId].getOutputs()[outputNeuron];
        return selectBackpropagationDerivative(outputLayerOutput)*(outputLayerOutput - this.inputData[inputDataId][2]);
    }

    public float deltaHiddenLayer3(int hiddenNeuron, int inputDataId){
        int nextLayerSize = this.parametersPerInput[inputDataId].getOutputs().length;
        float[][] w = parametersPerInput[inputDataId].getOutputWeights();
        float[] hiddenNeuronOutputs = this.parametersPerInput[inputDataId].getHidden3Outputs();
        float g = selectBackpropagationDerivative(hiddenNeuronOutputs[hiddenNeuron]);
        float delta = 0;
        float sumOfOutputLayerNeuronsDelta = 0F;
        for(int j=0; j<nextLayerSize; j++){
            sumOfOutputLayerNeuronsDelta += w[inputDataId][j]*deltaOutput(hiddenNeuron, inputDataId);
        }
        delta = g*sumOfOutputLayerNeuronsDelta;
        return delta;
    }

    public float deltaHiddenLayer2(int hiddenNeuron, int inputDataId){
        int nextLayerSize = this.parametersPerInput[inputDataId].getHidden3Outputs().length;
        float[][] w = parametersPerInput[inputDataId].getHidden3Weights();
        float[] hiddenNeuronOutputs = this.parametersPerInput[inputDataId].getHidden2Outputs();
        float g = selectBackpropagationDerivative(hiddenNeuronOutputs[hiddenNeuron]);
        float delta = 0;
        float sumOfOutputLayerNeuronsDelta = 0F;
        for(int j=0; j<nextLayerSize; j++){
            sumOfOutputLayerNeuronsDelta += w[inputDataId][j]*deltaOutput(hiddenNeuron, inputDataId);
        }
        delta = g*sumOfOutputLayerNeuronsDelta;
        return delta;
    }

    public float deltaHiddenLayer1(int hiddenNeuron, int inputDataId){
        int nextLayerSize = this.parametersPerInput[inputDataId].getHidden2Outputs().length;
        float[][] w = parametersPerInput[inputDataId].getHidden2Weights();
        float[] hiddenNeuronOutputs = this.parametersPerInput[inputDataId].getHidden1Outputs();
        float g = selectBackpropagationDerivative(hiddenNeuronOutputs[hiddenNeuron]);
        float delta = 0;
        float sumOfOutputLayerNeuronsDelta = 0F;
        for(int j=0; j<nextLayerSize; j++){
            sumOfOutputLayerNeuronsDelta += w[inputDataId][j]*deltaOutput(hiddenNeuron, inputDataId);
        }
        delta = g*sumOfOutputLayerNeuronsDelta;
        return delta;
    }

    public float[] selectHiddenLayerOutputs(int hiddenLayer, int inputDataId){
        switch (hiddenLayer) {
            case 1:
                return this.parametersPerInput[inputDataId].getHidden1Outputs();
            case 2:
                return this.parametersPerInput[inputDataId].getHidden2Outputs();
            case 3:
                return this.parametersPerInput[inputDataId].getHidden3Outputs();
            default:
                throw new RuntimeException("Wrong hidden layer. The available are 1, 2, 3.");
        }
    }

    public float[] selectNextLayerOutputs(int layer, int inputDataId){
        switch (layer) {
            case 0:
                return this.parametersPerInput[inputDataId].getHidden1Outputs();
            case 1:
                return this.parametersPerInput[inputDataId].getHidden2Outputs();
            case 2:
                return this.parametersPerInput[inputDataId].getHidden3Outputs();
            case 3:
                return this.parametersPerInput[inputDataId].getOutputs();
            default:
                throw new RuntimeException("Wrong layer. The available layers are 0, 1, 2, 3.");
        }
    }

    public float[][] selectLayerWeights(int layer, int inputDataId){
        switch (layer) {
            case 0:
                return this.parametersPerInput[inputDataId].getHidden1Weights();
            case 1:
                return this.parametersPerInput[inputDataId].getHidden2Weights();
            case 2:
                return this.parametersPerInput[inputDataId].getHidden3Weights();
            case 3:
                return this.parametersPerInput[inputDataId].getOutputWeights();
            default:
                throw new RuntimeException("Wrong layer. The available layers are 0, 1, 2, 3.");
        }
    }

    public int getNumOfInputs() {
        return numOfInputs;
    }

    public void setNumOfInputs(int numOfInputs) {
        this.numOfInputs = numOfInputs;
    }

    public int getNumOfCategories() {
        return numOfCategories;
    }

    public void setNumOfCategories(int numOfCategories) {
        this.numOfCategories = numOfCategories;
    }

    public int getNumOfHidden1() {
        return numOfHidden1;
    }

    public void setNumOfHidden1(int numOfHidden1) {
        this.numOfHidden1 = numOfHidden1;
    }

    public int getNumOfHidden2() {
        return numOfHidden2;
    }

    public void setNumOfHidden2(int numOfHidden2) {
        this.numOfHidden2 = numOfHidden2;
    }

    public int getNumOfHidden3() {
        return numOfHidden3;
    }

    public void setNumOfHidden3(int numOfHidden3) {
        this.numOfHidden3 = numOfHidden3;
    }

    public String getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(String activationFunction) {
        this.activationFunction = activationFunction;
    }

    public float[][] getHidden1Weights() {
        return hidden1Weights;
    }

    public void setHidden1Weights(float[][] hidden1Weights) {
        this.hidden1Weights = hidden1Weights;
    }

    public float[][] getHidden2Weights() {
        return hidden2Weights;
    }

    public void setHidden2Weights(float[][] hidden2Weights) {
        this.hidden2Weights = hidden2Weights;
    }

    public float[][] getHidden3Weights() {
        return hidden3Weights;
    }

    public void setHidden3Weights(float[][] hidden3Weights) {
        this.hidden3Weights = hidden3Weights;
    }

    public float[] getOutputWeights() {
        return outputWeights;
    }

    public void setOutputWeights(float[] outputWeights) {
        this.outputWeights = outputWeights;
    }

    public float[] getHidden1Outputs() {
        return hidden1Outputs;
    }

    public void setHidden1Outputs(float[] hidden1Outputs) {
        this.hidden1Outputs = hidden1Outputs;
    }

    public float[] getHidden2Outputs() {
        return hidden2Outputs;
    }

    public void setHidden2Outputs(float[] hidden2Outputs) {
        this.hidden2Outputs = hidden2Outputs;
    }

    public float[] getHidden3Outputs() {
        return hidden3Outputs;
    }

    public void setHidden3Outputs(float[] hidden3Outputs) {
        this.hidden3Outputs = hidden3Outputs;
    }

    public float[] getOutputs() {
        return outputs;
    }

    public void setOutputs(float[] outputs) {
        this.outputs = outputs;
    }

    public float getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public float getThreshold() {
        return threshold;
    }

    public void setThreshold(float threshold) {
        this.threshold = threshold;
    }

    public void setInputData(float[][] inputData) {
        this.inputData = inputData;
    }

    public float[][] getInputData(){
        return inputData;
    }


}