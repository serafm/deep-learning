import java.util.*;

public class RandomClusteringDataset {
    public static void main(String[] args) {
        // create instance of Random class
        Random rand = new Random();
        ArrayList<ArrayList<Float>> dataset = new ArrayList<>();

        // Generate random points (x1,x2) in [-1,1]
        for(int i=0; i<1200; i++) {

            if(i<150){
                float x1 = rand.nextFloat(0.8F, 1.2F);
                float x2 = rand.nextFloat(0.8F, 1.2F);
                ArrayList<Float> points = new ArrayList<Float>();
                points.add(x1);
                points.add(x2);
                dataset.add(points);
            }
            if(i>=150 && i<300){
                float x1 = rand.nextFloat(0F, 0.5F);
                float x2 = rand.nextFloat(0F, 0.5F);
                ArrayList<Float> points = new ArrayList<Float>();
                points.add(x1);
                points.add(x2);
                dataset.add(points);
            }
            if(i>=300 && i<450){
                float x1 = rand.nextFloat(0F,0.5F);
                float x2 = rand.nextFloat(1.5F,2F);
                ArrayList<Float> points = new ArrayList<Float>();
                points.add(x1);
                points.add(x2);
                dataset.add(points);
            }
            if(i>=450 && i<600){
                float x1 = rand.nextFloat(1.5F, 2F);
                float x2 = rand.nextFloat(0F, 0.5F);
                ArrayList<Float> points = new ArrayList<Float>();
                points.add(x1);
                points.add(x2);
                dataset.add(points);
            }
            if(i>=600 && i<750){
                float x1 = rand.nextFloat(1.5F, 2F);
                float x2 = rand.nextFloat(1.5F, 2F);
                ArrayList<Float> points = new ArrayList<Float>();
                points.add(x1);
                points.add(x2);
                dataset.add(points);
            }
            if(i>=750 && i<825){
                float x1 = rand.nextFloat(0.8F, 1.2F);
                float x2 = rand.nextFloat(0F, 0.4F);
                ArrayList<Float> points = new ArrayList<Float>();
                points.add(x1);
                points.add(x2);
                dataset.add(points);
            }
            if(i>=825 && i<900){
                float x1 = rand.nextFloat(0.8F, 1.2F);
                float x2 = rand.nextFloat(1.6F, 2F);
                ArrayList<Float> points = new ArrayList<Float>();
                points.add(x1);
                points.add(x2);
                dataset.add(points);
            }
            if(i>=900 && i<975){
                float x1 = rand.nextFloat(0.3F, 0.7F);
                float x2 = rand.nextFloat(0.8F, 1.2F);
                ArrayList<Float> points = new ArrayList<Float>();
                points.add(x1);
                points.add(x2);
                dataset.add(points);
            }
            if(i>=975 && i<1050){
                float x1 = rand.nextFloat(1.3F, 1.7F);
                float x2 = rand.nextFloat(0.8F, 1.2F);
                ArrayList<Float> points = new ArrayList<Float>();
                points.add(x1);
                points.add(x2);
                dataset.add(points);
            }
            if(i>=1050 && i<1200){
                float x1 = rand.nextFloat(0F, 2F);
                float x2 = rand.nextFloat(0F, 2F);
                ArrayList<Float> points = new ArrayList<Float>();
                points.add(x1);
                points.add(x2);
                dataset.add(points);
            }
        }
        System.out.println(dataset);
    }

}
