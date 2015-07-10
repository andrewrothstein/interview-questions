package main.java;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import com.google.common.collect.ImmutableMap;
import com.google.common.primitives.Doubles;

public class InputProcessor {

	private final JavaSparkContext sc;
	
	private static final Map<String, Double> labelsMap = 
			ImmutableMap.<String , Double>builder()
			.put("standing", 0d)
			.put("sitting", 1d)
			.put("sittingdown", 2d)
			.put("standingup", 3d)
			.put("walking", 4d).build();
	
	public InputProcessor(JavaSparkContext sc) {
		this.sc = sc;
	}
	
	public JavaRDD<LabeledPoint> readData(String file) {
		JavaRDD<String> textFile = sc.textFile(file);
		return textFile.map(InputProcessor::line2Point);
	}
	
	private static LabeledPoint line2Point(String line) {
		String[] fields = line.replace(",", ".").split(";");
		double[] vector = Doubles.toArray(Arrays.stream(Arrays.copyOfRange(fields, 6, 17)).map(Double::parseDouble).collect(Collectors.toList()));
		return new LabeledPoint(labelsMap.get(fields[18]), Vectors.dense(vector));
	}
}
