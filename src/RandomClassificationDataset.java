import java.io.IOException;
import java.util.*;
import static java.lang.Math.pow;

public class RandomClassificationDataset {
    public static void main(String[] args) throws IOException {

        // create instance of Random class
        Random rand = new Random();
        Map<List<Float>, String> train_data = new HashMap<>();
        Map<List<Float>, String> test_data = new HashMap<>();

        for(int i=0; i<8000; i++) {
            // Generate random points (x1,x2) in [-1,1]
            float x1 = rand.nextFloat(-1, 1);
            float x2 = rand.nextFloat(-1, 1);
            String category = getCategory(x1, x2);
            List<Float> points = new ArrayList<>();

            // split the 8000 random points into train and test datasets
            if (i < 4000) {
                points.add(x1);
                points.add(x2);
                train_data.put(points, category);
            }else{
                points.add(x1);
                points.add(x2);
                test_data.put(points, category);
            }
        }
        System.out.println(train_data);
    }

    // Classification of (x1,x2) in C1,C2,C3 categories
    private static String getCategory(float x1, float x2) {
        String category = "";

        if(((x1 - pow(0.5, 2)) + (x2 - pow(0.5, 2))) < 0.2 && (x2 >0.5)){
            category = "C1";
        }else if (((x1 - pow(0.5, 2)) + (x2 - pow(0.5, 2))) < 0.2 && (x2 <0.5)){
            category = "C2";
        }else if(((x1 + pow(0.5, 2)) + (x2 + pow(0.5, 2))) < 0.2 && (x2 >-0.5)){
            category = "C1";
        }else if(((x1 + pow(0.5, 2)) + (x2 + pow(0.5, 2))) < 0.2 && (x2 <-0.5)){
            category = "C2";
        }else if(((x1 - pow(0.5, 2)) + (x2 + pow(0.5, 2))) < 0.2 && (x2 >-0.5)){
            category = "C1";
        } else if(((x1 - pow(0.5, 2)) + (x2 + pow(0.5, 2))) < 0.2 && (x2 <-0.5)) {
            category = "C2";
        } else if(((x1 + pow(0.5, 2)) + (x2 - pow(0.5, 2))) < 0.2 && (x2 >0.5)) {
            category = "C1";
        } else if(((x1 + pow(0.5, 2)) + (x2 - pow(0.5, 2))) < 0.2 && (x2 >0.5)) {
            category = "C1";
        }else{
            category = "C3";
        }
        return category;
    }
}
