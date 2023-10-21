public class Datapoint {

    private float x1;
    private float x2;
    private int category;

    public Datapoint(float x1, float x2, int category){
        this.x1 = x1;
        this.x2 = x2;
        this.category = category;
    }

    public float getX1() {
        return x1;
    }

    public float getX2() {
        return x2;
    }

    public int getCategory() {
        return category;
    }
}
