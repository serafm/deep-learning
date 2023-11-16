public class ActivationFunctions {

    public ActivationFunctions(){}

    protected float logistic(float x){
        return (float) (1/(1+Math.exp(-x)));
    }

    protected float tanh(float x){
        return (float) ((Math.exp( x - Math.exp(-x)))
                / ( Math.exp( x + Math.exp(-x) ) ) );
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
        return switch (activationFunction) {
            case "tanh" -> tanh(x);
            case "relu" -> relu(x);
            default -> logistic(x);
        };
    }

    public float selectBackpropagationDerivative(float x, String activationFunction){
        return switch (activationFunction) {
            case "tanh" -> derivativeTanh(x);
            case "relu" -> derivativeRelu(x);
            default -> derivativeLogistic(x);
        };
    }

}
