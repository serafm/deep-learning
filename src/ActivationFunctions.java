public class ActivationFunctions {

    public ActivationFunctions(){}

    protected double logistic(double x){
        return 1 / (1+Math.exp(-x));
    }

    protected double tanh(double x){
        return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
    }

    protected double relu(double x){
        return Math.max(0,x);
    }

    protected double derivativeTanh(double x){
        return 1 - Math.pow(tanh(x), 2);
    }

    protected double derivativeRelu(double x){
        if (x < 0){
            return 0F;
        }
        return 1F;
    }

    protected double derivativeLogistic(double x){
        return Math.exp(-x) / Math.pow((1 + Math.exp(-x)), 2);
    }

    public double selectActivationFunction(double x, String activationFunction){
        return switch (activationFunction) {
            case "tanh" -> tanh(x);
            case "relu" -> relu(x);
            default -> logistic(x);
        };
    }

    public double selectBackpropagationDerivative(double x, String activationFunction){
        return switch (activationFunction) {
            case "tanh" -> derivativeTanh(x);
            case "relu" -> derivativeRelu(x);
            default -> derivativeLogistic(x);
        };
    }

}
