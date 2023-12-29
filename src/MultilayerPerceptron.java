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

    private double[][] hidden1Weights;
    private double[][] hidden2Weights;
    private double[][] hidden3Weights;
    private double[][] outputWeights;
    private double[] hidden1BiasWeights;
    private double[] hidden2BiasWeights;
    private double[] hidden3BiasWeights;
    private double[] outputBiasWeights;
    private double[] hidden1Outputs;
    private double[] hidden2Outputs;
    private double[] hidden3Outputs;
    private double[] outputs;
    private final double learningRate;
    private final double threshold;
    private ArrayList<Double> msePerEpoch;

    private double[] deltaOfHidden3;
    private double[] deltaOfHidden2;
    private double[] deltaOfHidden1;
    private double[] deltaOfOutput;

    private double[][] outputWeightsDerivatives;
    private double[] outputBiasDerivatives;
    private double[] hidden3BiasDerivatives;
    private double[] hidden2BiasDerivatives;
    private double[] hidden1BiasDerivatives;
    private double[][] hidden3WeightsDerivatives;
    private double[][] hidden2WeightsDerivatives;
    private double[][] hidden1WeightsDerivatives;
    private double[] totalInputOfOutputLayer;
    private double[] totalInputOfHidden3Layer;
    private double[] totalInputOfHidden2Layer;
    private double[] totalInputOfHidden1Layer;
    private double[][] outputWeightsPartialSum;
    private double[][] hidden3WeightsPartialSum;
    private double[][] hidden2WeightsPartialSum;
    private double[][] hidden1WeightsPartialSum;
    private double[] outputBiasPartialSum;
    private double[] hidden3BiasPartialSum;
    private double[] hidden2BiasPartialSum;
    private double[] hidden1BiasPartialSum;

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
    MultilayerPerceptron(int d, int K, int H1, int H2, int H3, String actv, double learningRate, double threshold, int batch_size, int epochs_size) {
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

        /**for (int i = 0; i < inputData.length; i++){
            System.out.println(inputData[i].getX1() + " " + inputData[i].getX2() + " " + Arrays.toString(inputData[i].getTarget()) + " " + Arrays.toString(inputData[i].getOutput()));
        }**/
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

        double x1 = Double.parseDouble(data_line[0]);
        double x2 = Double.parseDouble(data_line[1]);

        inputData[row] = new Datapoint(x1, x2, category, null);
    }

    /**
     * Random initialization of weights in (-1,1).
     */
    private void initializeWeights() {
        hidden1Weights = new double[numOfHidden1][numOfInputs];
        hidden2Weights = new double[numOfHidden2][numOfHidden1];
        hidden3Weights = new double[numOfHidden3][numOfHidden2];
        outputWeights = new double[numOfCategories][numOfHidden3];
        hidden1BiasWeights = new double[numOfHidden1];
        hidden2BiasWeights = new double[numOfHidden2];
        hidden3BiasWeights = new double[numOfHidden3];
        outputBiasWeights = new double[numOfCategories];
        hidden1Outputs = new double[numOfHidden1];
        hidden2Outputs = new double[numOfHidden2];
        hidden3Outputs = new double[numOfHidden3];
        outputs = new double[numOfCategories];

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
    private void init(int currentLayerSize, int previousLayerSize, double[][] weights, double[] bias){
        Random random = new Random();
        for (int i = 0; i < currentLayerSize; i++) {
            for (int j = 0; j < previousLayerSize; j++) {
                weights[i][j] = random.nextDouble() * 2 - 1; // Random weight between -1 and 1
            }
            bias[i] = random.nextDouble() * 2 - 1;
        }
    }

    /**
     * Train the network with gradient descent method. Update weights in batches or serial.
     */
    private void GradientDescent(){
        int t = 0;
        double difference = Double.POSITIVE_INFINITY;
        double error = 0;
        double mse = Double.POSITIVE_INFINITY;
        msePerEpoch = new ArrayList<>();

        outputWeightsPartialSum = new double[numOfCategories][numOfHidden3];
        hidden3WeightsPartialSum = new double[numOfHidden3][numOfHidden2];
        hidden2WeightsPartialSum = new double[numOfHidden2][numOfHidden1];
        hidden1WeightsPartialSum = new double[numOfHidden1][numOfInputs];
        outputBiasPartialSum = new double[numOfCategories];
        hidden3BiasPartialSum = new double[numOfHidden3];
        hidden2BiasPartialSum = new double[numOfHidden2];
        hidden1BiasPartialSum = new double[numOfHidden1];

        while(t < epochs_size || difference > threshold) {
            int n = 0;
            double previousErrorRate = error;
            error = 0;
            if(batch_size>1 && (inputData.length % batch_size == 0)) {
                for(int k=1; k<(inputData.length/batch_size)+1; k++) {
                    for (int b=0; b<batch_size; b++) {
                        ForwardPass(n);
                        error += LossFunction(n);
                        Backpropagation(n);

                        // Calculate partial summation of output weights
                        partialSum(numOfCategories, numOfHidden3, outputWeightsPartialSum, outputWeightsDerivatives);
                        partialSumBias(numOfCategories, outputBiasPartialSum, outputBiasDerivatives);

                        // Calculate partial summation of hidden layer 3 weights
                        partialSum(numOfHidden3, numOfHidden2, hidden3WeightsPartialSum, hidden3WeightsDerivatives);
                        partialSumBias(numOfHidden3, hidden3BiasPartialSum, hidden3BiasDerivatives);

                        // Calculate partial summation of hidden layer 2 weights
                        partialSum(numOfHidden2, numOfHidden1, hidden2WeightsPartialSum, hidden2WeightsDerivatives);
                        partialSumBias(numOfHidden2, hidden2BiasPartialSum, hidden2BiasDerivatives);

                        // Calculate partial summation of hidden layer 1 weights
                        partialSum(numOfHidden1, numOfInputs, hidden1WeightsPartialSum, hidden1WeightsDerivatives);
                        partialSumBias(numOfHidden1, hidden1BiasPartialSum, hidden1BiasDerivatives);

                        n += 1;
                    }
                    // Group Update of weights
                    groupUpdateWeights();
                    outputWeightsPartialSum = new double[numOfCategories][numOfHidden3];
                    hidden3WeightsPartialSum = new double[numOfHidden3][numOfHidden2];
                    hidden2WeightsPartialSum = new double[numOfHidden2][numOfHidden1];
                    hidden1WeightsPartialSum = new double[numOfHidden1][numOfInputs];
                    outputBiasPartialSum = new double[numOfCategories];
                    hidden3BiasPartialSum = new double[numOfHidden3];
                    hidden2BiasPartialSum = new double[numOfHidden2];
                    hidden1BiasPartialSum = new double[numOfHidden1];
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
            difference = Math.abs(previousErrorRate - error);
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
    private void partialSum(int currentLayerSize, int previousLayerSize, double[][] weightsPartialSum, double[][] weightsDerivatives){
        for (int i = 0; i < currentLayerSize; i++) {
            for (int j = 0; j < previousLayerSize; j++) {
                weightsPartialSum[i][j] += weightsDerivatives[i][j];
            }
        }
    }

    /**
     * Calculate the partial sum of layer biases
     * @param currentLayerSize Size of current layer
     * @param biasPartialSum Partial sum biases array of current layer
     * @param biasDerivatives Biases derivatives array of current layer
     */
    private void partialSumBias(int currentLayerSize, double[] biasPartialSum, double[] biasDerivatives){
        for (int i = 0; i < currentLayerSize; i++) {
            biasPartialSum[i] += biasDerivatives[i];
        }
    }

    /**
     * Serial update of weights
     */
    private void serialUpdateWeights(){
        for(int i = 0; i < numOfCategories; i++){
            for (int j = 0; j < numOfHidden3; j++) {
                outputWeights[i][j] = outputWeights[i][j] - learningRate * outputWeightsDerivatives[i][j];
            }
            outputBiasWeights[i] = outputBiasWeights[i] - learningRate * outputBiasDerivatives[i];
        }
        for(int i = 0; i < numOfHidden3; i++) {
            for (int j = 0; j < numOfHidden2; j++) {
                hidden3Weights[i][j] = hidden3Weights[i][j] - learningRate * hidden3WeightsDerivatives[i][j];
            }
            hidden3BiasWeights[i] = hidden3BiasWeights[i] - learningRate * hidden3BiasDerivatives[i];
        }
        for(int i = 0; i < numOfHidden2; i++) {
            for (int j = 0; j < numOfHidden1; j++) {
                hidden2Weights[i][j] = hidden2Weights[i][j] - learningRate * hidden2WeightsDerivatives[i][j];
            }
            hidden2BiasWeights[i] = hidden2BiasWeights[i] - learningRate * hidden2BiasDerivatives[i];
        }
        for(int i = 0; i < numOfHidden1; i++) {
            for (int j = 0; j < numOfInputs; j++) {
                hidden1Weights[i][j] = hidden1Weights[i][j] - learningRate * hidden1WeightsDerivatives[i][j];
            }
            hidden1BiasWeights[i] = hidden1BiasWeights[i] - learningRate * hidden1BiasDerivatives[i];
        }
    }

    /**
     * Group update of weights
     */
    private void groupUpdateWeights(){
        for(int i = 0; i < numOfCategories; i++) {
            for (int j = 0; j < numOfHidden3; j++) {
                outputWeights[i][j] = outputWeights[i][j] - learningRate * outputWeightsPartialSum[i][j];
            }
            outputBiasWeights[i] = outputBiasWeights[i] - learningRate * outputBiasPartialSum[i];
        }
        for(int i = 0; i < numOfHidden3; i++) {
            for (int j = 0; j < numOfHidden2; j++) {
                hidden3Weights[i][j] = hidden3Weights[i][j] - learningRate * hidden3WeightsPartialSum[i][j];
            }
            hidden3BiasWeights[i] = hidden3BiasWeights[i] - learningRate * hidden3BiasPartialSum[i];
        }
        for(int i = 0; i < numOfHidden2; i++) {
            for (int j = 0; j < numOfHidden1; j++) {
                hidden2Weights[i][j] = hidden2Weights[i][j] - learningRate * hidden2WeightsPartialSum[i][j];
            }
            hidden2BiasWeights[i] = hidden2BiasWeights[i] - learningRate * hidden2BiasPartialSum[i];
        }
        for(int i = 0; i < numOfHidden1; i++) {
            for (int j = 0; j < numOfInputs; j++) {
                hidden1Weights[i][j] = hidden1Weights[i][j] - learningRate * hidden1WeightsPartialSum[i][j];
            }
            hidden1BiasWeights[i] = hidden1BiasWeights[i] - learningRate * hidden1BiasPartialSum[i];
        }
    }

    /**
     * Forward pass of each input data and classification.
     * @param data_point Input data point
     */
    private void ForwardPass(int data_point){
        totalInputOfOutputLayer = new double[numOfCategories];
        totalInputOfHidden3Layer = new double[numOfHidden3];
        totalInputOfHidden2Layer = new double[numOfHidden2];
        totalInputOfHidden1Layer = new double[numOfHidden1];

        // Input layer -> hidden layer 1 forward pass
        for(int i=0; i<numOfHidden1; i++) {
            totalInputOfHidden1Layer[i] = inputData[data_point].getX1() * hidden1Weights[i][0] + inputData[data_point].getX2() * hidden1Weights[i][1] + hidden1BiasWeights[i];
            hidden1Outputs[i] = actvFunc.selectActivationFunction(totalInputOfHidden1Layer[i], activationFunction);
        }

        // Hidden Layer 1 -> hidden layer 2 forward pass
        layerForwardPass(hidden1Outputs, hidden2Weights, hidden2Outputs, hidden2BiasWeights, totalInputOfHidden2Layer, false);

        // Hidden Layer 2 -> hidden layer 3 forward pass
        layerForwardPass(hidden2Outputs, hidden3Weights, hidden3Outputs, hidden3BiasWeights, totalInputOfHidden3Layer, false);

        // Hidden Layer 3 -> output layer forward pass
        layerForwardPass(hidden3Outputs, outputWeights, outputs, outputBiasWeights, totalInputOfOutputLayer, true);

        // Classification of output
        inputData[data_point].setOutput(classification(outputs));

    }

    /**
     * Forward Pass
     * @param previousLayerOutputs The outputs of the previous layer (H-1)
     * @param layerWeights Current layer (H) weights
     * @param finalLayerOutputs Layer final outputs y(u)
     * @param bias Current layer bias of each neuron
     * @param currentLayerOutputs Current layer outputs (u) without the activation function
     */
    private void layerForwardPass(double[] previousLayerOutputs, double[][] layerWeights, double[] finalLayerOutputs, double[] bias, double[] currentLayerOutputs, boolean outputLayer){
        for(int i=0; i<currentLayerOutputs.length ; i++){
            for(int j=0; j<previousLayerOutputs.length ; j++){
                currentLayerOutputs[i] += previousLayerOutputs[j]*layerWeights[i][j];
            }
            currentLayerOutputs[i] += bias[i];
            if(outputLayer){
                finalLayerOutputs[i] = actvFunc.logistic(currentLayerOutputs[i]);
            }else{
                finalLayerOutputs[i] = actvFunc.selectActivationFunction(currentLayerOutputs[i], activationFunction);
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
    public int[] classification(double[] outputs) {
        // Initialize max to the first element of the array
        double max = 0F;
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
        outputWeightsDerivatives = new double[numOfCategories][numOfHidden3];
        outputBiasDerivatives = new double[numOfCategories];
        deltaOfOutput = new double[numOfCategories];
        for(int i=0; i<numOfCategories; i++){
            double g = outputs[i] * (1 - outputs[i]);
            int[] target = inputData[inputDataId].getTarget();
            deltaOfOutput[i] = g * (outputs[i] - target[i]);
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
        deltaOfHidden3 = new double[numOfHidden3];
        hidden3BiasDerivatives = new double[numOfHidden3];
        hidden3WeightsDerivatives = new double[numOfHidden3][numOfHidden2];
        // list of output layer weights
        for(int i=0; i<numOfHidden3; i++){ // for each neuron
            double g = actvFunc.selectBackpropagationDerivative(totalInputOfHidden3Layer[i], activationFunction); // derivative of neuron output
            double sumOfOutputLayerNeuronsDelta = 0F; // initialize summation of weight*delta of each output neuron
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
        deltaOfHidden2 = new double[numOfHidden2];
        hidden2BiasDerivatives = new double[numOfHidden2];
        hidden2WeightsDerivatives = new double[numOfHidden2][numOfHidden1];
        for(int i=0; i<numOfHidden2; i++){ // for each neuron
            double g = actvFunc.selectBackpropagationDerivative(totalInputOfHidden2Layer[i], activationFunction); // derivative of neuron output
            double sumOfOutputLayerNeuronsDelta = 0F; // initialize summation of weight*delta of each output neuron
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
        deltaOfHidden1 = new double[numOfHidden1];
        hidden1BiasDerivatives = new double[numOfHidden1];
        hidden1WeightsDerivatives = new double[numOfHidden1][numOfInputs];
        for(int i=0; i<numOfHidden1; i++){ // for each neuron
            double g = actvFunc.selectBackpropagationDerivative(totalInputOfHidden1Layer[i], activationFunction); // derivative of neuron output
            double sumOfOutputLayerNeuronsDelta = 0F; // initialize summation of weight*delta of each output neuron
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
        int[] o = inputData[data_point].getOutput();
        int[] t = inputData[data_point].getTarget();
        return equalCategories(o, t);
    }

    public int equalCategories(int[] output, int[] target){
        if (Arrays.equals(output, target)) {
            return 0;
        }else{
            return 1;
        }
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
    public void test() throws IOException {
        LoadDataset(Path.of("data/test.csv"));
        generalizationAbility();
    }**/

    /**
     * Calculation of the ability(%) of the MLP to generalize for unseen data.

    public double generalizationAbility() throws IOException {
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
        return ((double) counter / inputData.length)*100;
    }**/

    /**
    public void saveResultToCsv(){
        try {
            FileWriter csv = new FileWriter("data/results.csv", true);
            csv.write("\n" + numOfHidden1 + ", " + numOfHidden2 + ", " + numOfHidden3 + ", " + activationFunction + ", " + batch_size + ", " + generalizationAbility());
            csv.close();
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }**/
}