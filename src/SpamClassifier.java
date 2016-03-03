
import java.io.BufferedInputStream;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.jsoup.Jsoup;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;

public class SpamClassifier {
	static FastVector records;
	static FastVector unlabeledRecords;
	static ArrayList<String> fileNamesEmailSet;
	static Attribute emailMessage;
	public static final int LEARNING = 1;
	public static final int CLASSIFICATION = 2;
	public int mode = SpamClassifier.LEARNING;
	private static Attribute eClass;
	public String negDir;
	public String posDir;
	public String modelName;
	public String resultFile;
	public String dirName;
    

	// create Arff file from path
	public Instances loadTextstoArff(String path) throws IOException {
		TextDirectoryLoader loader = new TextDirectoryLoader();
		loader.setDirectory(new File(path));
		Instances dataRaw = loader.getDataSet();
		return dataRaw;
	}

	public void writeArf(Instances arff, String directory) throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(directory));
		writer.write(arff.toString());
		writer.flush();
		writer.close();
	}

	private Instances readTrainingDataset() throws IOException {
		Instances trainingSet = new Instances("SpamClsfyTraining", records, 8000);
		trainingSet.setClass(eClass);
		
		ArrayList<String> dataFilesSpam = this.listFiles(negDir);
		ArrayList<String> dataFilesHam = this.listFiles(posDir);
		//int size = dataFilesSpam.size()+dataFilesHam.size();
		//System.out.println("TrainingSet length " + size);
		//System.out.println("TrainingSet length spam" + trainingSet.numInstances());
		//System.out.println("TrainingSet length Ham" + trainingSet.numInstances());
		
		
		String content = "";
		
		for (int looper = 0; looper < dataFilesSpam.size(); looper++) {
			//data = this.readFileAsString(dataFilesSpam.get(looper));
			// System.out.println(data);
			
			content = this.readFileAsString(dataFilesSpam.get(looper));
			Instance rec = new Instance(2);
			
			
			rec.setValue((Attribute) records.elementAt(0), content);
            rec.setValue((Attribute) records.elementAt(1), "neg");
			
			trainingSet.add(rec);
		}
		for (int looper = 0; looper < dataFilesHam.size(); looper++) {
		//data = this.readFileAsString(dataFilesHam.get(looper));
		// System.out.println(data);
		
		content = this.readFileAsString(dataFilesHam.get(looper));
		Instance rec = new Instance(2);
		
		
		
		rec.setValue((Attribute) records.elementAt(0), content);
		rec.setValue((Attribute) records.elementAt(1), "pos");
		//Oversampling Hams
		/*
		 * 	for (int i = 0; i < 5; i++) {
		 
				trainingSet.add(rec);
			}
			*/
		trainingSet.add(rec);
		
	}
		System.out.println("FINISHED READING DATA");
		return trainingSet;
	}

	


	private String readFileAsString(String filePath) throws IOException {
		File file = new File(filePath);
		byte[] buffer = new byte[(int) file.length()];
		BufferedInputStream f = null;
		try {
			f = new BufferedInputStream(new FileInputStream(filePath));
			f.read(buffer);
		} finally {
			if (f != null)
				try {
					f.close();
				} catch (IOException ignored) {
				}
		}
		return new String(buffer);
	}

	private ArrayList<String> listFiles(String path) {
		ArrayList<String> fileNames = new ArrayList<String>();
		File dir = new File(path);
		File[] files = dir.listFiles();
		//System.out.println(files.length);

		for (int loop = 0; loop < files.length; loop++) {
			/*if (files[loop].isDirectory()) {
				listFiles(files[loop].getAbsolutePath(), fileNames);
			} else*/
				fileNames.add(files[loop].getAbsolutePath());
		}
		// Collections.shuffle(fileNames);
		return fileNames;
	}

	public Classifier loadClassifier(String rPath) {
		Classifier cl = null;
		try{
			//String dir = System.getProperty("user.dir");
			cl = (Classifier) weka.core.SerializationHelper.read(rPath);
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("not found the model");
		}
		return cl;
	}

	public void saveClassifier(Classifier model, String rPath) {
		//String path = System.getProperty("user.dir");
		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(rPath));

			oos.writeObject(model);
			oos.flush();
			oos.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void filterEmails(Classifier model, String pfadzuEmails) throws Exception {

		Instances unlabeled = readUnlabeled(pfadzuEmails);
		unlabeled.setClassIndex(1);

		Instances labeled = new Instances(unlabeled);
		writeArf(unlabeled, "unlabaled_mails_filtered.arff");
		File file = new File(resultFile);
		FileWriter f = new FileWriter(file);
		BufferedWriter s = new BufferedWriter(f);

		for (int i = 0; i < unlabeled.numInstances(); i++) {
			String fileName = fileNamesEmailSet.get(i);
			String [] split = fileName.split("\\/|\\.");
			fileName = split[split.length-2];
			// System.out.println(fileName);
			double clsLabel = model.classifyInstance(unlabeled.instance(i));
			labeled.instance(i).setClassValue(clsLabel);
			String label;
			if (clsLabel == 0){
				label = "neg";
			}else{
				label = "pos";
			}
			s.write(fileName + "\t" + label + "\n");

			labeled.instance(i).setClassValue(clsLabel);
			// System.out.println("Labeled "+labeled.instance(i));
		}
		writeArf(labeled, "classified_mails.arff");
		s.close();
	}

	public Instances readUnlabeled(String pfad) throws Exception {

		//Attribute a_emailSubject = new Attribute("emailSubject", (FastVector) null);
		Attribute Content = new Attribute("content", (FastVector) null);
		FastVector emailClass = new FastVector(2);
		emailClass.addElement("neg");
		emailClass.addElement("pos");
		Attribute eClass = new Attribute("class", emailClass);
		unlabeledRecords = new FastVector(2);
		//unlabeledRecords.addElement(a_emailSubject);
		unlabeledRecords.addElement(Content);
		unlabeledRecords.addElement(eClass);
		// records.addElement(emailMessage);
		Instances ESet = new Instances("SpamClsfy", unlabeledRecords, 8000);
		ESet.setClass(eClass);
		fileNamesEmailSet = new ArrayList<String>();
		fileNamesEmailSet = this.listFiles(pfad);
		
		String content = "";
		for (int looper = 0; looper < fileNamesEmailSet.size(); looper++) {
			
			// System.out.println(data);
			//emailSubject = extractEmailSubject(data);
			//content = extractEmailFature(data);
			content=this.readFileAsString(fileNamesEmailSet.get(looper));
			Instance rec = new Instance(2);
			
			
			//rec.setValue(a_emailSubject, emailSubject);
			rec.setValue(Content, content);

			ESet.add(rec);
		}
		writeArf(ESet, "unlabaled_posts.arff");
		return ESet;

	}

	public static void main(String[] args) throws Exception {
		SpamClassifier classifier = new SpamClassifier();
		
			String m = 	args[0];
			if (m.equals("learn")){
				classifier.mode = SpamClassifier.LEARNING;
				classifier.negDir = args[1];
				classifier.posDir = args[2];
				classifier.modelName = args[3];
				//System.out.println(classifier.spamDir);
				
				
			}else if (m.equals("classify")){
				classifier.mode = SpamClassifier.CLASSIFICATION;
				//classifier.modelName = args[1];
				classifier.dirName = args[1];
				classifier.resultFile = args[2];
			}else{
				System.out.println("Mode not supported");
				return;
			
		}

		
		//classifier.mode = SpamClassifier.CLASSIFICATION;
		if (classifier.mode == SpamClassifier.LEARNING) {

			//Attribute emailSubject = new Attribute("emailSubject", (FastVector) null);
			Attribute emailContent = new Attribute("content", (FastVector) null);
			FastVector emailClass = new FastVector(2);
			emailClass.addElement("neg");
			emailClass.addElement("pos");
			eClass = new Attribute("class", emailClass);
			records = new FastVector(2);
			//records.addElement(emailSubject);
			records.addElement(emailContent);
			records.addElement(eClass);
			
			Instances trainingset = classifier.readTrainingDataset();
			//classifier.writeArf(trainingset, "test.arff");
			classifier.writeArf(trainingset, "labaled.arff");

			Classifier bestModel = Crossvalidator.validate(10, trainingset);
			// write model to file
			classifier.saveClassifier(bestModel, classifier.modelName);
			System.out.println("MODEL SAVED");
			
		} else {
			//Classifier model = classifier.loadClassifier(classifier.modelName);
			Instances unlabeled = classifier.readUnlabeled(classifier.dirName);
			//classifier.filterEmails(model, classifier.dirName);
			System.out.println("Finished Classification");
		}
	}
}