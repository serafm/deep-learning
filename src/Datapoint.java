public class Datapoint {

    private double x1;
    private double x2;
    private int[] target;
    private int[] output;

    public Datapoint(double x1, double x2, int[] target, int[] output){
        this.x1 = x1;
        this.x2 = x2;
        this.target = target;
        this.output = output;
    }

    public double getX1() {
        return this.x1;
    }

    public double getX2() {
        return this.x2;
    }

    public int[] getTarget() {
        return this.target;
    }

    public int[] getOutput() { return this.output; }

    public void setOutput(int[] output) {
        this.output = output;
    }
}
