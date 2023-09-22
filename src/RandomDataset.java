import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import static java.lang.Math.pow;
import org.javatuples.Triplet;

public class RandomDataset {
    public static void main(String[] args) throws IOException {

        // create instance of Random class
        Random rand = new Random();
        ArrayList<Triplet<Float, Float, String>> train_data = new ArrayList<>();
        ArrayList<Triplet<Float, Float, String>> test_data = new ArrayList<>();

        //FileWriter train_dataset = new FileWriter("train.csv");
        //FileWriter test_dataset = new FileWriter("test.csv");


        for(int i=0; i<8000; i++) {
            // Generate random points (x1,x2) in [-1,1]
            float x1 = rand.nextFloat(-1, 1);
            float x2 = rand.nextFloat(-1, 1);
            String category = setCategory(x1, x2);
            Triplet<Float, Float, String> pairs = new Triplet<>(x1, x2, category);

            // split the 8000 random points into train and test datasets
            if (i < 4000) {
                train_data.add(pairs);
            }else{
                test_data.add(pairs);
            }
        }
    }

    // Classification of (x1,x2) in C1,C2,C3 categories
    private static String setCategory(float x1, float x2) {
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
