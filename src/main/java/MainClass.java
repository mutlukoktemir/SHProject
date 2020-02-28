
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

import weka.classifiers.bayes.NaiveBayesMultinomial;

import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.classifiers.Classifier;
import zemberek.core.logging.Log;
import zemberek.morphology.TurkishMorphology;
import zemberek.ner.*;

public class MainClass {

    private static MyFilteredLearner learner;
    private static MyFilteredClassifier classifier;

    /**
     * Main method. With an example usage of this class.
     */
//    public static void main(String[] args) throws IOException {
//
//
//        Classifier m_Classifier = null;
//
//        Path lookupRoot = Paths.get("/Users/test/IdeaProjects/message analysis/data/normalization");
//        Path lmFile = Paths.get("/Users/test/IdeaProjects/message analysis/data/lm/lm.2gram.slm");
//        TurkishMorphology morphology = TurkishMorphology.createWithDefaults();
//        TurkishSentenceNormalizer normalizer = new
//                TurkishSentenceNormalizer(morphology, lookupRoot, lmFile);
//
//        //TurkishMorphology morphology = TurkishMorphology.createWithDefaults();
//
//        String sentence = "Siktir gir göt ederi 210.000₺ normalde, ben yine sana kıyak olsun diye 275.000₺ dedim";
//        String normalizedSentence = normalizer.normalize(sentence);
//        System.out.println(normalizedSentence);
//
//        System.out.println("--");
//
//        String[] arr = normalizedSentence.split(" ");
//
//        for ( String ss : arr)
//            System.out.println(ss);
//
//
////        WordAnalysis results = morphology.analyze(sentence);
////        for (SingleAnalysis result : results)
////            System.out.println(result);
//
//        Log.info("Word = " + arr[0]);
//        WordAnalysis results = morphology.analyze(arr[0]);
//
//        for (SingleAnalysis result : results) {
//            Log.info(result);
//            Log.info("Lexical and Surface : " + result.formatLong());
//            Log.info("Only Lexical        : " + result.formatLexical());
//            Log.info("Oflazer style       : " +
//                    AnalysisFormatters.OFLAZER_STYLE.format(result));
//            Log.info();
//        }
//
//        final String MODEL = "models/sms.dat";
//
//        WekaClassifier wt = new WekaClassifier();
//
//        if (new File(MODEL).exists()) {
//            wt.loadModel(MODEL);
//        } else {
//            wt.transform();
//            wt.fit();
//            wt.saveModel(MODEL);
//        }
//
//        Logger logger
//                = Logger.getLogger(
//                MainClass.class.getName());
//
//        //run few predictions
//        logger.info("text 'how are you' is " + wt.predict("how are you ?"));
//        logger.info("text 'u have won the 1 lakh prize' is " + wt.predict("u have won the 1 lakh prize"));
//
//        //run evaluation
//        logger.info("Evaluation Result: \n"+wt.evaluate());
//
//
////        // load CSV
////        CSVLoader loader = new CSVLoader();
////        loader.setSource(new File("/Users/test/IdeaProjects/message analysis/data/real_train_set.csv"));
////        Instances data = loader.getDataSet();
////
////        // save ARFF
////        ArffSaver saver = new ArffSaver();
////        saver.setInstances(data);
////        saver.setFile(new File("/Users/test/IdeaProjects/message analysis/data/real_train_set.arff"));
////        saver.setDestination(new File("/Users/test/IdeaProjects/message analysis/data/real_train_set.arff"));
////        saver.writeBatch();
//
//        System.out.println("done.");
//
//    }
    public static void main(String[] args) throws Exception {

        String trainingFileName = new String("/Users/test/IdeaProjects/message analysis/data/real_train_set.arff");
        String testFileName = new String("/Users/test/IdeaProjects/message analysis/data/real_test_set.arff");
        String txtTestFileName = new String("/Users/test/IdeaProjects/message analysis/data/real_test_set.txt");
        String txtTestFileName2 = new String("/Users/test/IdeaProjects/message analysis/data/real_test_set_2.txt");
        String modelFile = new String("/Users/test/IdeaProjects/message analysis/data/real_model.dat");

//        withNaive(trainingFileName,testFileName);

//        readFileResults();

        learnByFiltered(trainingFileName,modelFile);
//
        clasifyByFiltered(txtTestFileName2,modelFile);

//        trial();



////        BufferedReader datafile = readDataFile("/Users/test/IdeaProjects/message analysis/data/real_train_set.arff");
//        BufferedReader inputReader = null;
//
//        try {
//            inputReader = new BufferedReader(new FileReader("/Users/test/IdeaProjects/message analysis/data/real_train_set.arff"));
//        } catch (FileNotFoundException ex) {
//            System.err.println("File not found: " + "/Users/test/IdeaProjects/message analysis/data/real_train_set.arff");
//        }
//        Instances data = new Instances(inputReader);
//        data.setClassIndex(0);
//
////        StringToWordVector filter = null;
////        weka.filters.unsupervised.attribute.RemoveType filter = new weka.filters.unsupervised.attribute.RemoveType();
//        weka.filters.unsupervised.attribute.StringToNominal filter = new weka.filters.unsupervised.attribute.StringToNominal();
//
////        filter.setAttributeIndices("last");
//
//
//        FilteredClassifier classifier = new FilteredClassifier();
//        classifier.setFilter(filter);
//        classifier.setClassifier(new NaiveBayesMultinomial());
//
//        Evaluation eval = new Evaluation(data);
//        eval.crossValidateModel(classifier, data, 4, new Random(1));
//        System.out.println(eval.toSummaryString());





    }

    public static void generateNerModel() throws IOException {
        // you will need ner-train and ner-test files to run this example.

        Path trainPath = Paths.get("/Users/test/IdeaProjects/message analysis/data/ner-train");
        Path testPath = Paths.get("/Users/test/IdeaProjects/message analysis/data/ner-test");
        Path modelRoot = Paths.get("/Users/test/IdeaProjects/message analysis/data/my-model");

        NerDataSet trainingSet = NerDataSet.load(trainPath, NerDataSet.AnnotationStyle.BRACKET);
        Log.info(trainingSet.info()); // prints information

        NerDataSet testSet = NerDataSet.load(testPath, NerDataSet.AnnotationStyle.BRACKET);
        Log.info(testSet.info());

        TurkishMorphology morphology = TurkishMorphology.createWithDefaults();

        // Training occurs here. Result is a PerceptronNer instance.
        // There will be 7 iterations with 0.1 learning rate.
        PerceptronNer ner = new PerceptronNerTrainer(morphology)
                .train(trainingSet, testSet, 13, 0.1f);

        Files.createDirectories(modelRoot);
        ner.saveModelAsText(modelRoot);
    }

    public static void useNer() throws IOException {
        // assumes you generated a model in my-model directory.
        Path modelRoot = Paths.get("/Users/test/IdeaProjects/message analysis/data/my-model");

        TurkishMorphology morphology = TurkishMorphology.createWithDefaults();

        PerceptronNer ner = PerceptronNer.loadModel(modelRoot, morphology);

        String sentence = "Ali Kaan yarın İstanbul'a gidecek.";

        NerSentence result = ner.findNamedEntities(sentence);

        List<NamedEntity> namedEntities = result.getNamedEntities();

        for (NamedEntity namedEntity : namedEntities) {
            System.out.println(namedEntity);
        }
    }

    public static void withNaive(String trainingFileName, String testFileName) throws Exception {

        StringToWordVector filter = new StringToWordVector();

//        Classifier naive = new NaiveBayesMultinomial();
//        LibSVM naive = new LibSVM();
        Classifier naive = new SMO();

        //training data
        Instances train = new Instances(new BufferedReader(new FileReader(trainingFileName)));
        int lastIndex = train.numAttributes() - 1;
        train.setClassIndex(lastIndex);
        filter.setInputFormat(train);
        train = Filter.useFilter(train, filter);

        //testing data
        Instances test = new Instances(new BufferedReader(new FileReader(testFileName)));
        test.setClassIndex(lastIndex);
//        filter.setInputFormat(test);
        Instances test2 = Filter.useFilter(test, filter);

        System.out.println("-----1");
        naive.buildClassifier(train);
        System.out.println("-----2");


            //      -------
            File file = new File("/Users/test/IdeaProjects/message analysis/data/results.txt");
            FileWriter fr = null;
            try {
                fr = new FileWriter(file);
                for(int i=0; i<test2.numInstances(); i++) {

//                    System.out.println(test.instance(i));
                    double index = naive.classifyInstance(test2.instance(i));
//            String className = train.attribute(0).value((int)index);
                    String className = train.classAttribute().value((int)index);

//                    writeUsingFileWriter(className,fileName);
                    //      -------

                    fr.write(className);
                    fr.write("\n");
                }

            } catch (IOException e) {
                e.printStackTrace();
            }finally{
                //close resources
                try {
                    fr.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }



//            System.out.println(className);


    }

    public static void readFileResults(){
        try {
            File myObj = new File("/Users/test/IdeaProjects/message analysis/data/results.txt");
            Scanner myReader = new Scanner(myObj);
            int i = 1, truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0;
            double precision = 0.0;
            double recall = 0.0;
            double accuracy = 0.0;


            while (myReader.hasNextLine()) {
                String data = myReader.nextLine();
                if( i % 18 == 0 || i % 19 == 0 || i % 20 == 0)
                {
                    if( data.equals("bad") )
                        truePositive++;
                    else
                        falsePositive++;

                }
                else
                {
                    if( data.equals("noBad") )
                        trueNegative++;
                    else
                        falseNegative++;
                }
                ++i;
            }

            System.out.println("TruePositive: " + truePositive + "\nTrueNegative: " + trueNegative + "\nFalsePositive: " + falsePositive + "\nFalseNegative: " + falseNegative);

            accuracy = (double)(truePositive + trueNegative) / (double)(truePositive + trueNegative + falseNegative + falsePositive);
            precision = (double)(truePositive) / (double)(truePositive+falsePositive);
            recall = (double)(truePositive) / (double)(truePositive+falseNegative);

            System.out.println("Accuracy: " + accuracy + "\nPrecision: " + precision + "\nRecall: " + recall);

            myReader.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    /**
     * Use FileWriter when number of write operations are less
     * @param data
     */
    private static void writeUsingFileWriter(String data, String fileName) {
        File file = new File(fileName);
        FileWriter fr = null;
        try {
            fr = new FileWriter(file);
            fr.write(data);
        } catch (IOException e) {
            e.printStackTrace();
        }finally{
            //close resources
            try {
                fr.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }


    public static void learnByFiltered(String fileData, String fileModel){

        learner = new MyFilteredLearner();

        learner.loadDataset(fileData);
        // Evaluation mus be done before training
        // More info in: http://weka.wikispaces.com/Use+WEKA+in+your+Java+code

        // optional
//        learner.evaluate();
        //

        learner.learn();
        learner.saveModel(fileModel);


    }


    public static void clasifyByFiltered(String fileTest, String modelFile){

        classifier = new MyFilteredClassifier();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileTest));
            String line;
            while ((line = reader.readLine()) != null) {
                String text = "";
                text = text + " " + line;
//                System.out.println(text);
                classifier.load(text);
                classifier.loadModel(modelFile);
                classifier.makeInstance();
                classifier.classify();
            }
        }
        catch (IOException e) {
            System.out.println("Problem found when reading: " + fileTest);
        }


    }

    public static void trial() throws Exception {

        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader("/Users/test/IdeaProjects/message analysis/data/real_train_set.arff"));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + "/Users/test/IdeaProjects/message analysis/data/real_train_set.arff");
        }
        Instances dataRaw = new Instances(inputReader);
        dataRaw.setClassIndex(dataRaw.numAttributes() - 1);

        // apply the StringToWordVector
        // (see the source code of setOptions(String[]) method of the filter
        // if you want to know which command-line option corresponds to which
        // bean property)
        StringToWordVector filter = new StringToWordVector();
        filter.setInputFormat(dataRaw);
        Instances dataFiltered = Filter.useFilter(dataRaw, filter);

        //System.out.println("\n\nFiltered data:\n\n" + dataFiltered);

        // train J48 and output model
        J48 classifier = new J48();
        classifier.buildClassifier(dataFiltered);
        System.out.println("\n\nClassifier model:\n\n" + classifier);


    }


}