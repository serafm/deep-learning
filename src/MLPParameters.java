public class MLPParameters {

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

    public float[][] getOutputWeightsDerivatives() {
        return outputWeightsDerivatives;
    }

    public void setOutputWeightsDerivatives(float[][] outputWeightsDerivatives) {
        this.outputWeightsDerivatives = outputWeightsDerivatives;
    }

    public float[][] getHidden3WeightsDerivatives() {
        return hidden3WeightsDerivatives;
    }

    public void setHidden3WeightsDerivatives(float[][] hidden3WeightsDerivatives) {
        this.hidden3WeightsDerivatives = hidden3WeightsDerivatives;
    }

    public float[][] getHidden2WeightsDerivatives() {
        return hidden2WeightsDerivatives;
    }

    public void setHidden2WeightsDerivatives(float[][] hidden2WeightsDerivatives) {
        this.hidden2WeightsDerivatives = hidden2WeightsDerivatives;
    }

    public float[][] getHidden1WeightsDerivatives() {
        return hidden1WeightsDerivatives;
    }

    public void setHidden1WeightsDerivatives(float[][] hidden1WeightsDerivatives) {
        this.hidden1WeightsDerivatives = hidden1WeightsDerivatives;
    }

    public float[] getOutputBiasDerivatives() {
        return outputBiasDerivatives;
    }

    public void setOutputBiasDerivatives(float[] outputBiasDerivatives) {
        this.outputBiasDerivatives = outputBiasDerivatives;
    }

    public float[] getHidden3BiasDerivatives() {
        return hidden3BiasDerivatives;
    }

    public void setHidden3BiasDerivatives(float[] hidden3BiasDerivatives) {
        this.hidden3BiasDerivatives = hidden3BiasDerivatives;
    }

    public float[] getHidden2BiasDerivatives() {
        return hidden2BiasDerivatives;
    }

    public void setHidden2BiasDerivatives(float[] hidden2BiasDerivatives) {
        this.hidden2BiasDerivatives = hidden2BiasDerivatives;
    }

    public float[] getHidden1BiasDerivatives() {
        return hidden1BiasDerivatives;
    }

    public void setHidden1BiasDerivatives(float[] hidden1BiasDerivatives) {
        this.hidden1BiasDerivatives = hidden1BiasDerivatives;
    }

    private float[][] outputWeightsDerivatives;
    private float[][] hidden3WeightsDerivatives;
    private float[][] hidden2WeightsDerivatives;
    private float[][] hidden1WeightsDerivatives;
    private float[] outputBiasDerivatives;
    private float[] hidden3BiasDerivatives;
    private float[] hidden2BiasDerivatives;

    public MLPParameters(float[][] hidden1Weights, float[][] hidden2Weights, float[][] hidden3Weights, float[][] outputWeights, float[] hidden1BiasWeights, float[] hidden2BiasWeights, float[] hidden3BiasWeights, float[] outputBiasWeights, float[] hidden1Outputs, float[] hidden2Outputs, float[] hidden3Outputs, float[] outputs, float[][] outputWeightsDerivatives, float[][] hidden3WeightsDerivatives, float[][] hidden2WeightsDerivatives, float[][] hidden1WeightsDerivatives, float[] outputBiasDerivatives, float[] hidden3BiasDerivatives, float[] hidden2BiasDerivatives, float[] hidden1BiasDerivatives) {
        this.hidden1Weights = hidden1Weights;
        this.hidden2Weights = hidden2Weights;
        this.hidden3Weights = hidden3Weights;
        this.outputWeights = outputWeights;
        this.hidden1BiasWeights = hidden1BiasWeights;
        this.hidden2BiasWeights = hidden2BiasWeights;
        this.hidden3BiasWeights = hidden3BiasWeights;
        this.outputBiasWeights = outputBiasWeights;
        this.hidden1Outputs = hidden1Outputs;
        this.hidden2Outputs = hidden2Outputs;
        this.hidden3Outputs = hidden3Outputs;
        this.outputs = outputs;
        this.outputWeightsDerivatives = outputWeightsDerivatives;
        this.hidden3WeightsDerivatives = hidden3WeightsDerivatives;
        this.hidden2WeightsDerivatives = hidden2WeightsDerivatives;
        this.hidden1WeightsDerivatives = hidden1WeightsDerivatives;
        this.outputBiasDerivatives = outputBiasDerivatives;
        this.hidden3BiasDerivatives = hidden3BiasDerivatives;
        this.hidden2BiasDerivatives = hidden2BiasDerivatives;
        this.hidden1BiasDerivatives = hidden1BiasDerivatives;
    }

    private float[] hidden1BiasDerivatives;


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

    public float[][] getOutputWeights() {
        return outputWeights;
    }

    public void setOutputWeights(float[][] outputWeights) {
        this.outputWeights = outputWeights;
    }

    public float[] getHidden1BiasWeights() {
        return hidden1BiasWeights;
    }

    public void setHidden1BiasWeights(float[] hidden1BiasWeights) {
        this.hidden1BiasWeights = hidden1BiasWeights;
    }

    public float[] getHidden2BiasWeights() {
        return hidden2BiasWeights;
    }

    public void setHidden2BiasWeights(float[] hidden2BiasWeights) {
        this.hidden2BiasWeights = hidden2BiasWeights;
    }

    public float[] getHidden3BiasWeights() {
        return hidden3BiasWeights;
    }

    public void setHidden3BiasWeights(float[] hidden3BiasWeights) {
        this.hidden3BiasWeights = hidden3BiasWeights;
    }

    public float[] getOutputBiasWeights() {
        return outputBiasWeights;
    }

    public void setOutputBiasWeights(float[] outputBiasWeights) {
        this.outputBiasWeights = outputBiasWeights;
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
}
