package main.java;

import java.util.HashMap;
import java.util.Map;

import org.apache.commons.lang.StringUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;

import scala.Tuple2;

import com.google.common.collect.Table;
import com.google.common.collect.TreeBasedTable;

/**
 * 
 * @author 
 * 
 * This classifier aims to solve the machine learning problem explained here: http://groupware.les.inf.puc-rio.br/har#literature
 * Along with this publication:
 * 
 * Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. 
 * Proceedings of 21st Brazilian Symposium on Artificial Intelligence. 
 * Advances in Artificial Intelligence - SBIA 2012. 
 * In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. 
 * ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6. 
 * 
 * 
 * The input to main is the data file that has headers as following:
 * user;gender;age;how_tall_in_meters;weight;body_mass_index;x1;y1;z1;x2;y2;z2;x3;y3;z3;x4;y4;z4;class
 * 
 * The x_i, y_i, z_i values are values collected from sensors on the belt, left thigh and right arm accelerometers
 * Similar to the original paper, we choose not to use the user to body_mass_index data fields as features based on Mark Hall's selection algorithm based on correlation
 * the class, or activity is what we're trying to train the model to predict
 * 
 * The original paper used AdaBoost and decision trees. We are using a Random Forest as the multi-label classifier due to its inherent powers to generalize and not overfit
 *
 * A grid search was ran for best number of tree and tree depth, 40 trees and a max depth of 10 are sufficient and yield good results for the testing error to be around 1%
 *   
 *
 */
public class HARLearning {

	// create JavaSparkContext
	private final SparkConf sparkConf = new SparkConf().setAppName("HAR").setMaster("local[4]").set("spark.executor.memory", "1g");
	private final JavaSparkContext sc = new JavaSparkContext(sparkConf);
	
	public static void main(String[] args) {
		if (args.length < 1) 
			throw new RuntimeException("Please enter the source file to train on");
		
		HARLearning learn = new HARLearning();
		learn.train(args[0]);
	}

	/**
	 * 
	 * Train method to take the file, parse and convert to MlLib internal format and train the data
	 * 
	 * @param file: input file to train
	 */
	private void train(String file) {
		
		//read, parse and convert the data into JavaRDD for processing in MlLib
		InputProcessor ip = new InputProcessor(sc);
		JavaRDD<LabeledPoint> data = ip.readData(file);
		
		// Split the data into training and test sets (10% held out for out of sample validation)
		JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.9, 0.1});
		
		JavaRDD<LabeledPoint> trainingData = splits[0];
		trainingData.cache();
		
		JavaRDD<LabeledPoint> testData = splits[1];
		testData.cache();
		
		// Model Training

		// We know before hand there are five classes, these classes were also mapped in InputProcessor.java
		Integer numClasses = 5; 
		
		// Features are collected accelerator measurements, we assume that they're continuous 
		Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>(); 

		// More tress has diminishing returns and increases computation time
		Integer numTrees = 40; 

		// Let the algorithm choose the feature subset strategy
		String featureSubsetStrategy = "auto";

		// use Gini Impurity
		String impurity = "gini";

		// Maximum depth of tree is 10
		Integer maxDepth = 10;

		// Max number of bins is 100
		Integer maxBins = 100;

		// Any seed
		Integer seed = 9999;


		// train model with above parameters
	
		RandomForestModel model = RandomForest.trainClassifier(
				trainingData, numClasses, 
				categoricalFeaturesInfo, numTrees, 
				featureSubsetStrategy, impurity, 
				maxDepth, maxBins, 
				seed);

		// Evaluate model on test instances and compute test error
		JavaPairRDD<Double, Double> predictionAndLabel = testData.mapToPair(p -> new Tuple2<Double, Double>(model.predict(p.features()), p.label()));

		
		// outputs the confusion matrix
		printConfusionMatrix(predictionAndLabel);

		// calculate the out of sample error with the test data
		Double testErr = 1.0 * predictionAndLabel.filter(pl -> !pl._1().equals(pl._2())).count() / testData.count();
		System.out.println(" Test Error: " + testErr);

	}

	/**
	 * prints the confusion matrix in csv format
	 * @param predictionAndLabel, the created predictions and labels, in a collection of pairs
	 * 
	 */
	
	private static void printConfusionMatrix(JavaPairRDD<Double, Double> predictionAndLabel) {
		Table<Integer, Integer, Long> table = TreeBasedTable.create();
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				final int x = i, y = j;
				long count = predictionAndLabel.filter(pl -> (pl._1() == x && pl._2() == y)).count();
				table.put(x, y, count);
			}
		}
		System.out.println("," + StringUtils.join(table.columnKeySet(), ","));
		for (Integer x : table.rowKeySet()) {
			System.out.println(x+","+StringUtils.join(table.row(x).values(), ","));
		}
	}
	
	
	/*
	 * 
Sample grid search for optimal tree depth (horizontal) and number of trees (vertical)
	4				7				10				13				16
20	0.1832351497	0.0681762396	0.0290255277	0.0161389298	0.0115979381
40	0.1737849779	0.0612420226	0.0266323024	0.015586647		0.0104933726
60	0.1813328424	0.0599533628	0.0286573392	0.0154639175	0.0101865488
80	0.1795532646	0.0624079529	0.0273073147	0.013868434		0.0106774669
100	0.1834806087	0.0632670594	0.0271845852	0.0145434462	0.0106774669
					
					
Sample confusion matrix					
	0		1		2		3		4
0	4640	0		52		112		147
1	0		5032	7		9		0
2	0		3		1072	53		19
3	4		4		19		1045	5
4	18		0		23		21		4191

	 */
}
