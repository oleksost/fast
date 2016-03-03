import java.io.File;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.meta.Stacking;
import weka.classifiers.meta.ThresholdSelector;
import weka.classifiers.trees.ADTree;
import weka.classifiers.trees.BFTree;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Crossvalidator {

	public static Classifier validate(int folds, Instances dataFiltered) throws Exception {
		StringToWordVector filter = new StringToWordVector();
		filter.setLowerCaseTokens(true);
		filter.setTFTransform(true);
		filter.setIDFTransform(true);
		filter.setOutputWordCounts(true);
		File stopList=new File("/Users/Alexey/Documents/workspace/javaws/Assignment3/stop_words_german+english");
		filter.setStopwords(stopList);
		filter.setUseStoplist(true);
		filter.setInputFormat(dataFiltered);

		FilteredClassifier filtClassifier = new FilteredClassifier();
		filtClassifier.setFilter(filter);
		Bagging bagging =new Bagging();
		bagging.setClassifier(new NaiveBayes());
		bagging.setNumIterations(10);
		//Classifier model = new Bagging().setClassifier(new NaiveBayes());
		
		//ThresholdSelector ts = new ThresholdSelector(); 
		//ts.setManualThresholdValue(0.7);
		//ts.setClassifier(model);
		filtClassifier.setClassifier(bagging);
		
		int seed = 10;
		double previousfalsPos = 0;

		// randomize data
		Random rand = new Random(seed);
		Instances randData = new Instances(dataFiltered);

		randData.randomize(rand);
/*
		if (randData.classAttribute().isNominal()) {
			randData.stratify(folds);

		}*/
		// perform cross-validation
		Evaluation eval = new Evaluation(randData);
		for (int n = 0; n < folds; n++) {
			Instances train = randData.trainCV(folds, n);
			// classifier.writeArf(train, "train"+n);
			Instances test = randData.testCV(folds, n);

			// check stratification
			/*int numspam = 0;
			int numham = 0;

			for (int a = 0; a < test.numInstances(); a++) {

				// System.out.println(test.instance(a).stringValue(0));
				if (test.instance(a).stringValue(0) == "neg") {
					numspam++;
				} else
					numham++;

			}
			double relation = numspam / numham;
			System.out.println("Spam / Ham = " + relation);
*/
			// build and evaluate classifier
			filtClassifier.buildClassifier(train);

			// cModel = (Classifier) new RandomTree();

			eval.evaluateModel(filtClassifier, test);

			System.out.println("Fold Number " + n);
			/*
			System.out.println("P Fold:" + eval.precision(0));
			System.out.println("E Fod:" + eval.errorRate());
			System.out.println("R Fod:" + eval.recall(0));
			System.out.println("fMeasure Fold:" + eval.fMeasure(0));

			System.out.println("TRUE POSITIVES " + eval.numTruePositives(0));
			System.out.println("TRUE Negatives " + eval.numTrueNegatives(0));
			System.out.println("Correct Total " + eval.correct());
			*/

			double falsePos = eval.numFalsePositives(0) - previousfalsPos;
			previousfalsPos += falsePos;

			//System.out.println("Num False POSITIVES " + falsePos);
			// System.out.println("Num False POSITIVES
			// "+eval.numFalsePositives(0));

			System.out.println("Train length " + train.numInstances());
			System.out.println("Test length " + test.numInstances());
			System.out.println(" ");

		}

		// output evaluation
		System.out.println();
		System.out.println("=== Result ===");
		System.out.println("Folds: " + folds);
		System.out.println("Seed: " + seed);
		System.out.println();
		System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation ===", false));

		return filtClassifier;

	}

}
