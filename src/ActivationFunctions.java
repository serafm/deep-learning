public class ActivationFunctions {

    public ActivationFunctions(){}

    protected float logistic(float x){
        return (float) (1/(1+Math.exp(Double.valueOf(-x))));
    }

    protected float tanh(float x){
        return (float) ((Math.exp(Double.valueOf(x) - Math.exp(Double.valueOf(-x))))
                / (Math.exp(Double.valueOf(x) + Math.exp(Double.valueOf(-x)))));
    }

    protected float relu(float x){
        return Math.max(0,x);
    }

    protected float derivativeTanh(float x){
        return (float) (1 - Math.pow(tanh(x), 2));
    }

    protected float derivativeRelu(float x){
        if (x < 0){
            return 0F;
        }
        return 1F;
    }

    protected float derivativeLogistic(float x){
        return (float) (Math.exp(-x)/Math.pow((1 + Math.exp(-x)), 2));
    }

    public float selectActivationFunction(float x, String activationFunction){
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

    public float selectBackpropagationDerivative(float x, String activationFunction){
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

}
