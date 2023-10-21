import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import static java.lang.Math.pow;

public class RandomClassificationDataset {
    public static void main(String[] args) throws IOException {

        // create instance of Random class
        Random rand = new Random();
        FileWriter train = new FileWriter("data/train.csv"); ;
        FileWriter test = new FileWriter("data/test.csv"); ;

        for(int i=0; i<8000; i++) {
            // Generate random points (x1,x2) in [-1,1]
            float x1 = rand.nextFloat(-1, 1);
            x1 = Math.round(x1 * 100) / 100.0f;
            float x2 = rand.nextFloat(-1, 1);
            x2 = Math.round(x2 * 100) / 100.0f;
            String category = getCategory(x1, x2);

            // split the 8000 random points into train and test datasets
            if (i < 4000) {
                train.write(x1 + "," + x2 + "," + category + "\n");
            }else{
                test.write(x1 + "," + x2 + "," + category + "\n");
            }
        }
        train.close();
        test.close();
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
