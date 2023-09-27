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

    public MLPParameters(float[][] hidden1Weights, float[][] hidden2Weights, float[][] hidden3Weights, float[][] outputWeights, float[] hidden1BiasWeights, float[] hidden2BiasWeights, float[] hidden3BiasWeights, float[] outputBiasWeights, float[] hidden1Outputs, float[] hidden2Outputs, float[] hidden3Outputs, float[] outputs) {
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