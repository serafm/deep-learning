import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import static java.lang.Math.pow;

public class GenerateRandomDataset {
    public static void main(String[] args) throws IOException {

        // create instance of Random class
        Random rand = new Random();
        FileWriter train = new FileWriter("data/train.csv");
        FileWriter test = new FileWriter("data/test.csv");

        for(int i=0; i<8000; i++) {
            // Generate random points (x1,x2) in [-1,1]
            float x1 = rand.nextFloat(-1, 1);
            //x1 = Math.round(x1 * 100) / 100.0f;
            float x2 = rand.nextFloat(-1, 1);
            //x2 = Math.round(x2 * 100) / 100.0f;
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

    // Classification of (x1,x2) in C1,C2,C3,C4 categories
    private static String getCategory(float x1, float x2) {

        boolean condition_1 = pow((x1 - 0.5), 2) + pow((x2 - 0.5), 2) < 0.2;
        boolean condition_2 = pow((x1 + 0.5), 2) + pow((x2 + 0.5), 2) < 0.2;
        boolean condition_3 = pow((x1 - 0.5), 2) + pow((x2 + 0.5), 2) < 0.2;
        boolean condition_4 = pow((x1 + 0.5), 2) + pow((x2 - 0.5), 2) < 0.2;
        boolean conditionCategory_1 = (condition_1 && x1 > 0.5) || (condition_2 && x1 > -0.5)|| (condition_3 && x1 > 0.5) || (condition_4 && x1 > -0.5);
        boolean conditionCategory_2 = (condition_1 && x1 < 0.5) || (condition_2 && x1 < -0.5) || (condition_3 && x1 < 0.5) || (condition_4 && x1 < -0.5);

        if (conditionCategory_1) {
            return "C1";
        } else if (conditionCategory_2) {
            return "C2";
        } else if (x1 > 0) {
            return "C3";
        } else {
            return "C4";
        }
    }
}
