import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

public class MultilayerPerceptron {

    private int numOfInputs;
    private int numOfCategories;
    private int numOfHidden1;
    private int numOfHidden2;
    private int numOfHidden3;
    private final int epochs_size;
    private final int batch_size;
    private final String activationFunction;
    private Datapoint[] inputData;

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
    private OutputCategory[] outputCategoryPerInput = new OutputCategory[4000];
    private final float learningRate;
    private final float threshold;
    private ArrayList<Float> msePerEpoch;
    private ArrayList<OutputCategory[]> outputsPerEpoch;

    private float[] deltaOfHidden3;
    private float[] deltaOfHidden2;
    private float[] deltaOfHidden1;
    private float[] deltaOfOutput;

    private float[][] outputWeightsDerivatives;
    private float[] outputBiasDerivatives;
    private float[] hidden3BiasDerivatives;
    private float[] hidden2BiasDerivatives;
    private float[] hidden1BiasDerivatives;
    private float[][] hidden3WeightsDerivatives;
    private float[][] hidden2WeightsDerivatives;
    private float[][] hidden1WeightsDerivatives;
    private float[] totalInputOfOutputLayer;
    private float[] totalInputOfHidden3Layer;
    private float[] totalInputOfHidden2Layer;
    private float[] totalInputOfHidden1Layer;
    private float[][] outputWeightsPartialSum;
    private float[][] hidden3WeightsPartialSum;
    private float[][] hidden2WeightsPartialSum;
    private float[][] hidden1WeightsPartialSum;

    private final ActivationFunctions actvFunc = new ActivationFunctions();

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
     * @param dataPath path of csv file
     */
    public void LoadDataset(Path dataPath){
        int trainSize = 4000;
        inputData = new Datapoint[trainSize];
        int row = 0;

        try {
            File train_dataset = new File(String.valueOf(dataPath));
            Scanner dataset = new Scanner(train_dataset);
            while (dataset.hasNextLine()) {
                String[] data_line = dataset.nextLine().split(",");

                // Encode the category of input data
                encodeCategory(data_line, row);

                row++;
            }
            dataset.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    /**
     * Encode the category of the input data point
     * @param data_line Input data (x1,x2,category)
     * @param row Index of the input data
     */
    private void encodeCategory(String[] data_line, int row){
        int[] category;

        switch (data_line[2]) {
            case "C1" -> category = new int[]{1,0,0,0};
            case "C2" -> category = new int[]{0,1,0,0};
            case "C3" -> category = new int[]{0,0,1,0};
            default -> category = new int[]{0,0,0,1};
        }

        float x1 = Float.parseFloat(data_line[0]);
        float x2 = Float.parseFloat(data_line[1]);

        inputData[row] = new Datapoint(x1, x2, category);
    }

    /**
     * Random initialization of weights in (-1,1).
     */
    private void initializeWeights() {
        hidden1Weights = new float[numOfHidden1][numOfInputs];
        hidden2Weights = new float[numOfHidden2][numOfHidden1];
        hidden3Weights = new float[numOfHidden3][numOfHidden2];
        outputWeights = new float[numOfCategories][numOfHidden3];
        hidden1BiasWeights = new float[numOfHidden1];
        hidden2BiasWeights = new float[numOfHidden2];
        hidden3BiasWeights = new float[numOfHidden3];
        outputBiasWeights = new float[numOfCategories];
        hidden1Outputs = new float[numOfHidden1];
        hidden2Outputs = new float[numOfHidden2];
        hidden3Outputs = new float[numOfHidden3];
        outputs = new float[numOfCategories];

        // Initialize hidden layer 1 weights and bias
        init(numOfHidden1, numOfInputs, hidden1Weights, hidden1BiasWeights);

        // Initialize hidden layer 2 weights and bias
        init(numOfHidden2, numOfHidden1, hidden2Weights, hidden2BiasWeights);

        // Initialize hidden layer 3 weights and bias
        init(numOfHidden3, numOfHidden2, hidden3Weights, hidden3BiasWeights);

        // Initialize output layer weights and bias
        init(numOfCategories, numOfHidden3, outputWeights, outputBiasWeights);
    }

    /**
     * Initialize weights and bias of a layer
     * @param currentLayerSize Number of neurons in the current layer
     * @param previousLayerSize Number of neurons in the previous layer
     * @param weights Array of current layer weights
     * @param bias Array of current layer bias weights
     */
    private void init(int currentLayerSize, int previousLayerSize, float[][] weights, float[] bias){
        Random random = new Random();
        for (int i = 0; i < currentLayerSize; i++) {
            for (int j = 0; j < previousLayerSize; j++) {
                weights[i][j] = random.nextFloat() * 2 - 1; // Random weight between -1 and 1
            }
            bias[i] = random.nextFloat() * 2 - 1;
        }
    }

    /**
     * Train the network with gradient descent method. Update weights in batches or serial.
     */
    private void GradientDescent(){
        int t = 0;
        float previousErrorRate;
        float difference = Float.POSITIVE_INFINITY;
        float error = 0;
        float mse = Float.POSITIVE_INFINITY;
        msePerEpoch = new ArrayList<>();
        outputsPerEpoch = new ArrayList<>();

        outputWeightsPartialSum = new float[numOfCategories][numOfHidden3];
        hidden3WeightsPartialSum = new float[numOfHidden3][numOfHidden2];
        hidden2WeightsPartialSum = new float[numOfHidden2][numOfHidden1];
        hidden1WeightsPartialSum = new float[numOfHidden1][numOfInputs];

        while(t < epochs_size || difference > threshold) {
            int n = 0;
            previousErrorRate = error;
            error = 0;
            if(batch_size>1 && (inputData.length % batch_size == 0)) {
                for(int k=1; k<(inputData.length/batch_size)+1; k++) {
                    for (int b=0; b<batch_size; b++) {
                        ForwardPass(n);
                        error += LossFunction(n);
                        Backpropagation(n);

                        // Calculate partial summation of output weights
                        partialSum(numOfCategories, numOfHidden3, outputWeightsPartialSum, outputWeightsDerivatives);

                        // Calculate partial summation of hidden layer 3 weights
                        partialSum(numOfHidden3, numOfHidden2, hidden3WeightsPartialSum, hidden3WeightsDerivatives);

                        // Calculate partial summation of hidden layer 2 weights
                        partialSum(numOfHidden2, numOfHidden1, hidden2WeightsPartialSum, hidden2WeightsDerivatives);

                        // Calculate partial summation of hidden layer 1 weights
                        partialSum(numOfHidden1, numOfInputs, hidden1WeightsPartialSum, hidden1WeightsDerivatives);

                        n += 1;
                    }
                    // Group Update of weights
                    groupUpdateWeights();
                }
            } else {
                // if Batch size=1. Serial update of weights
                for (int p = 0; p < inputData.length; p++) {
                    ForwardPass(p);
                    error += LossFunction(p);
                    Backpropagation(p);
                    serialUpdateWeights();
                }
            }

            mse = (error /inputData.length);
            System.out.println("Epoch " + t  + "\n" +  "Loss Function = " + mse);
            msePerEpoch.add(mse);
            outputsPerEpoch.add(outputCategoryPerInput);
            difference = Math.abs(previousErrorRate - error);
            System.out.println("Previous: " + previousErrorRate + "  Current: " + error);
            System.out.println("Difference: " + difference);
            t += 1;
        }
    }

    /**
     * Calculate the partial sum of layer weights
     * @param currentLayerSize Size of current layer
     * @param previousLayerSize Size of previous layer
     * @param weightsPartialSum Partial sum weights array of current layer
     * @param weightsDerivatives Weights derivatives array of current layer
     */
    private void partialSum(int currentLayerSize, int previousLayerSize, float[][] weightsPartialSum, float[][] weightsDerivatives){
        for (int i = 0; i < currentLayerSize; i++) {
            for (int j = 0; j < previousLayerSize; j++) {
                weightsPartialSum[i][j] += weightsDerivatives[i][j];
            }
        }
    }

    private OutputCategory[] bestEpoch(){
        float minMSE = Float.POSITIVE_INFINITY;
        int minIndex = 0;

        for(int i=0; i<msePerEpoch.size(); i++){
            if(msePerEpoch.get(i) < minMSE){
                minMSE = msePerEpoch.get(i);
                minIndex = i;
            }
        }

        return outputsPerEpoch.get(minIndex);
    }

    /**
     * Serial update of weights
     */
    private void serialUpdateWeights(){
        for(int i = 0; i < numOfCategories; i++){
            for (int j = 0; j < numOfHidden3; j++) {
                outputWeights[i][j] = outputWeights[i][j] - learningRate * outputWeightsDerivatives[i][j];
            }
        }
        for(int i = 0; i < numOfHidden3; i++) {
            for (int j = 0; j < numOfHidden2; j++) {
                hidden3Weights[i][j] = hidden3Weights[i][j] - learningRate * hidden3WeightsDerivatives[i][j];
            }
        }
        for(int i = 0; i < numOfHidden2; i++) {
            for (int j = 0; j < numOfHidden1; j++) {
                hidden2Weights[i][j] = hidden2Weights[i][j] - learningRate * hidden2WeightsDerivatives[i][j];
            }
        }
        for(int i = 0; i < numOfHidden1; i++) {
            for (int j = 0; j < numOfInputs; j++) {
                hidden1Weights[i][j] = hidden1Weights[i][j] - learningRate * hidden1WeightsDerivatives[i][j];
            }
        }
    }

    /**
     * Group update of weights
     */
    private void groupUpdateWeights(){
        //System.out.println("\nNew output weights: ");
        for(int i = 0; i < numOfCategories; i++) {
            for (int j = 0; j < numOfHidden3; j++) {
                outputWeights[i][j] = outputWeights[i][j] - learningRate * outputWeightsPartialSum[i][j];
                //System.out.println(outputWeights[i][j]);
            }
            outputBiasWeights[i] = outputBiasWeights[i] - learningRate * outputBiasDerivatives[i];
        }
        //System.out.println("\nNew hidden 3 weights: ");
        for(int i = 0; i < numOfHidden3; i++) {
            for (int j = 0; j < numOfHidden2; j++) {
                hidden3Weights[i][j] = hidden3Weights[i][j] - learningRate * hidden3WeightsPartialSum[i][j];
                //System.out.println(hidden3Weights[i][j]);
            }
            hidden3BiasWeights[i] = hidden3BiasWeights[i] - learningRate * hidden3BiasDerivatives[i];
        }
        //System.out.println("\nNew hidden 2 weights: ");
        for(int i = 0; i < numOfHidden2; i++) {
            for (int j = 0; j < numOfHidden1; j++) {
                hidden2Weights[i][j] = hidden2Weights[i][j] - learningRate * hidden2WeightsPartialSum[i][j];
                //System.out.println(hidden2Weights[i][j]);
            }
            hidden2BiasWeights[i] = hidden2BiasWeights[i] - learningRate * hidden2BiasDerivatives[i];
        }
        //System.out.println("\nNew hidden 1 weights: ");
        for(int i = 0; i < numOfHidden1; i++) {
            for (int j = 0; j < numOfInputs; j++) {
                hidden1Weights[i][j] = hidden1Weights[i][j] - learningRate * hidden1WeightsPartialSum[i][j];
                //System.out.println(hidden1Weights[i][j]);
            }
            hidden1BiasWeights[i] = hidden1BiasWeights[i] - learningRate * hidden1BiasDerivatives[i];
        }
    }

    /**
     * Forward pass of each input data and classification.
     * @param data_point Input data point
     */
    private void ForwardPass(int data_point){
        totalInputOfOutputLayer = new float[numOfCategories];
        totalInputOfHidden3Layer = new float[numOfHidden3];
        totalInputOfHidden2Layer = new float[numOfHidden2];
        totalInputOfHidden1Layer = new float[numOfHidden1];

        // Input layer -> hidden layer 1 forward pass
        for(int i=0; i<numOfHidden1; i++) {
            hidden1Outputs[i] = inputData[data_point].getX1() * hidden1Weights[i][0] + inputData[data_point].getX2() * hidden1Weights[i][1];
            totalInputOfHidden1Layer[i] = hidden1Outputs[i] + hidden1BiasWeights[i];
            hidden1Outputs[i] = actvFunc.selectActivationFunction(totalInputOfHidden1Layer[i], activationFunction);
        }

        // Hidden Layer 1 -> hidden layer 2 forward pass
        layerForwardPass(hidden1Outputs, hidden2Weights, hidden2Outputs, hidden2BiasWeights, totalInputOfHidden2Layer, false);

        // Hidden Layer 2 -> hidden layer 3 forward pass
        layerForwardPass(hidden2Outputs, hidden3Weights, hidden3Outputs, hidden3BiasWeights, totalInputOfHidden3Layer, false);

        // Hidden Layer 3 -> output layer forward pass
        layerForwardPass(hidden3Outputs, outputWeights, outputs, outputBiasWeights, totalInputOfOutputLayer, true);

        // Classification of output
        outputCategoryPerInput[data_point] = new OutputCategory(classification(outputs));

    }

    /**
     * Forward Pass
     * @param previousLayerOutputs The outputs of the previous layer (H-1)
     * @param layerWeights Current layer (H) weights
     * @param finalLayerOutputs Layer final outputs y(u)
     * @param bias Current layer bias of each neuron
     * @param layerOutputs Current layer outputs (u) without the activation function
     */
    private void layerForwardPass(float[] previousLayerOutputs, float[][] layerWeights, float[] finalLayerOutputs, float[] bias, float[] layerOutputs, boolean outputLayer){
        for(int i=0; i<layerOutputs.length ; i++){
            for(int j=0; j<previousLayerOutputs.length ; j++){
                finalLayerOutputs[i] += previousLayerOutputs[j]*layerWeights[i][j];
            }
            layerOutputs[i] = finalLayerOutputs[i] + bias[i];
            if(outputLayer){
                finalLayerOutputs[i] = actvFunc.logistic(layerOutputs[i]);
            }else{
                finalLayerOutputs[i] = actvFunc.selectActivationFunction(layerOutputs[i], activationFunction);
            }
        }
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
     * Classify each input data point to a category. Select the output with the maximum value as the category.
     * @param outputs array of category outputs
     * @return category of input data
     */
    public int[] classification(float[] outputs) {
        // Initialize max to the first element of the array
        float max = 0F;
        int[] category = new int[4];
        // Iterate through the array to find the maximum value
        for (int i=0; i < numOfCategories; i++) {
            if (outputs[i] > max) {
                max = outputs[i];
                if(i == 0){
                    category = new int[]{1,0,0,0};
                }else if(i == 1){
                    category = new int[]{0, 1, 0, 0};
                }else if(i == 2){
                    category = new int[]{0, 0, 1, 0};
                }else if(i == 3){
                    category = new int[]{0, 0, 0, 1};
                }
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
            float g = actvFunc.selectBackpropagationDerivative(totalInputOfOutputLayer[i], "logistic");
            int[] o = inputData[inputDataId].getCategory();
            int[] t = outputCategoryPerInput[inputDataId].getOutputCategory();
            deltaOfOutput[i] = g*(equalCategories(o, t));
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
        deltaOfHidden3 = new float[numOfHidden3];
        hidden3BiasDerivatives = new float[numOfHidden3];
        hidden3WeightsDerivatives = new float[numOfHidden3][numOfHidden2];
        // list of output layer weights
        for(int i=0; i<numOfHidden3; i++){ // for each neuron
            float g = actvFunc.selectBackpropagationDerivative(totalInputOfHidden3Layer[i], activationFunction); // derivative of neuron output
            float sumOfOutputLayerNeuronsDelta = 0F; // initialize summation of weight*delta of each output neuron
            for(int j=0; j<numOfCategories; j++){
                sumOfOutputLayerNeuronsDelta += outputWeights[j][i]*deltaOfOutput[j];
            }
            deltaOfHidden3[i] = g*sumOfOutputLayerNeuronsDelta; // add it in list of neuron deltas
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
        deltaOfHidden2 = new float[numOfHidden2];
        hidden2BiasDerivatives = new float[numOfHidden2];
        hidden2WeightsDerivatives = new float[numOfHidden2][numOfHidden1];
        for(int i=0; i<numOfHidden2; i++){ // for each neuron
            float g = actvFunc.selectBackpropagationDerivative(totalInputOfHidden2Layer[i], activationFunction); // derivative of neuron output
            float sumOfOutputLayerNeuronsDelta = 0F; // initialize summation of weight*delta of each output neuron
            for(int j=0; j<numOfHidden3; j++){
                sumOfOutputLayerNeuronsDelta += hidden3Weights[j][i]*deltaOfHidden3[j]; // CHECK SWAP [i][j]
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
        deltaOfHidden1 = new float[numOfHidden1];
        hidden1BiasDerivatives = new float[numOfHidden1];
        hidden1WeightsDerivatives = new float[numOfHidden1][numOfInputs];
        for(int i=0; i<numOfHidden1; i++){ // for each neuron
            float g = actvFunc.selectBackpropagationDerivative(totalInputOfHidden1Layer[i], activationFunction); // derivative of neuron output
            float sumOfOutputLayerNeuronsDelta = 0F; // initialize summation of weight*delta of each output neuron
            for(int j=0; j<numOfHidden2; j++){
                sumOfOutputLayerNeuronsDelta += hidden2Weights[j][i]*deltaOfHidden2[j];
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
        int[] o = inputData[data_point].getCategory();
        int[] t = outputCategoryPerInput[data_point].getOutputCategory();
        return equalCategories(o, t);
    }

    public int equalCategories(int[] o, int[] t){
        int e;
        if (Arrays.equals(o, t)){
            e = 0;
        }else {
            e = 1;
        }
        return e;
    }

    /**
     * Training of MLP
     */
    public void train(){
        LoadDataset(Path.of("data/train.csv"));
        initializeWeights();
        GradientDescent();
    }

    /**
     * Test dataset and generalization ability of MLP
     */
    public void test() throws IOException {
        LoadDataset(Path.of("data/test.csv"));
        generalizationAbility();
    }

    /**
     * Calculation of the ability(%) of the MLP to generalize for unseen data.
     */
    public float generalizationAbility() throws IOException {
        int counter = 0;

        FileWriter test = new FileWriter("data/test_plot.csv"); ;

        for(int i=0; i<inputData.length; i++){
            ForwardPass(i);
            if( Arrays.equals(inputData[i].getCategory(), outputCategoryPerInput[i].getOutputCategory())){
                counter += 1;
                test.write(inputData[i].getX1() + "," + inputData[i].getX2() +  ",Correct" + "\n");
            } else {
                test.write(inputData[i].getX1() + "," + inputData[i].getX2() +  ",Wrong" + "\n");
            }
        }
        return ((float) counter / inputData.length)*100;
    }

    public void saveResultToCsv(){
        try {
            FileWriter csv = new FileWriter("data/results.csv", true);
            csv.write("\n" + numOfHidden1 + ", " + numOfHidden2 + ", " + numOfHidden3 + ", " + activationFunction + ", " + batch_size + ", " + generalizationAbility());
            csv.close();
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }
}