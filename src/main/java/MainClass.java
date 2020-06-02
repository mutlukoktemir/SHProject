
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.spec.RSAOtherPrimeInfo;
import java.util.*;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
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
import zemberek.morphology.analysis.SingleAnalysis;
import zemberek.morphology.analysis.WordAnalysis;
import zemberek.morphology.lexicon.RootLexicon;
import zemberek.ner.*;
import zemberek.normalization.TurkishSentenceNormalizer;
import zemberek.tokenization.Token;
import zemberek.tokenization.TurkishTokenizer;

import javax.activation.FileDataSource;

public class MainClass {

    private static MyFilteredLearner learner;
    private static MyFilteredClassifier classifier;
    
    private static Locale trLocale = new Locale("tr","TR");

    /**
     * Main method. With an example usage of this class.
     */
    public static void main(String[] args) throws Exception {

        String trainingFileName = new String("data/real_train_set.arff");
        String testFileName = new String("data/real_test_set.arff");
        String txtTestFileName = new String("data/real_test_set.txt");
        String txtTestFileName2 = new String("data/real_test_set_2.txt");
        String modelFile = new String("data/real_model.dat");

//        withNaive(trainingFileName,testFileName);

//        readFileResults();

//        learnByFiltered(trainingFileName,modelFile);
//
//        clasifyByFiltered(txtTestFileName2,modelFile);

//        trial();


        Path lookupRoot = Paths.get("data/normalization");
        Path lmFile = Paths.get("data/lm/lm.2gram.slm");
        
//        TurkishMorphology morphology = TurkishMorphology.createWithDefaults();
//        TurkishMorphology morphology = TurkishMorphology.builder().setLexicon(RootLexicon.getDefault()).useInformalAnalysis().build();
//        TurkishSentenceNormalizer normalizer = new TurkishSentenceNormalizer(morphology, lookupRoot, lmFile);

        String sentence = "Siktir gir a.q. göt ederi 210.000₺ normalde, ben yine sana kıyak olsun diye 275.000₺ dedim";
        
        String sentence2 = "Çok Şazla yoruma gerek yÖk . Çünkü hepsiburada sitesi on line alış veriş konusunda Ilk 5te .";
        
//        String sentence3 = "Ürünü yorumlara bakarak aldım iyi https://w ki de almışım :D. Uygun'da fiyat ve yüksek kalitede bir ürün pişman olmazsınız . Tavsiye ederim";
//
//        List<String> tokensOfSentenceOrigin = tokenizeSentence(sentence3);
//
//        StringBuilder strBuild = new StringBuilder();
//        for(int i = 0; i < tokensOfSentenceOrigin.size(); ++i){
//            strBuild.append(tokensOfSentenceOrigin.get(i)+" ");
//        }
//
//        System.out.println("origin:");
//        String strOrg = strBuild.toString();
//        System.out.println(strOrg);
//
//        String lowerSentence3 = strOrg.toLowerCase(trLocale);
//
//        String[] lowerStrArray = lowerSentence3.split(" ");
//        StringBuilder strBuild2 = new StringBuilder();
//
//        for(int i = 0; i < lowerStrArray.length; ++i){
//            strBuild2.append(lowerStrArray[i]+" ");
//        }
//
//        String strLower = strBuild2.toString().trim();
//
//        System.out.println("lower:");
//        System.out.println(strLower);





//        orderStringsInFileByNumOfWords("data/blackListShort.txt","data/blackListShortOrderedByNumOfWords.txt");
//        orderStringsInFileByNumOfWords("data/bads.txt","data/badsOrderedByNumOfWords.txt");
//
//        tagSentencesFromFile("data/mk_hb_train_set1.txt");
//
//        generateNerModel("data/tagged_hb_training.txt","data/mk_hb_test_set1.txt","data/my-hb-ner-model");
    
//        try {
//            File myObj = new File("data/mk_hb_test_set2.txt");
//            Scanner myReader = new Scanner(myObj);
//            int i = 0;
//            while (myReader.hasNextLine()) {
//                String data = myReader.nextLine();
//                System.out.println("test sentence:\n"+data+"\nResults from ner model:\n");
//                useNer("data/my-hb-ner-model",data);
//                ++i;
//                if( i == 63 )
//                    break;
//            }
//            myReader.close();
//        } catch (FileNotFoundException e) {
//            System.out.println("An error occurred.");
//            e.printStackTrace();
//        }
    
        String sentence3 = "siktiriniz hüremlak ta gördüğüm ilan için YouTubeda aramıştım.";
        String sentence4 = "orosbu yapma ederi 210.000₺ normalde, ben yine sana kıyak olsun diye 275.000₺ dedim";
        
//        System.out.println("Sentence:" + sentence3);
        
        PerceptronNer myNer = generatePerceptronNer("data/my-hb-ner-model");
        
        
        String testFile1 = "data/mk_hb_test_set1.txt";
        
//        testNerModelZ(myNer,testFile1);
        
//        useNer(myNer,sentence4);
        
        
        /*
        List<SingleAnalysis> singleAnalysisList = morphology.analyzeAndDisambiguate(sentence).bestAnalysis();
        for(int i = 0; i < singleAnalysisList.size(); ++i){
            System.out.println(singleAnalysisList.get(i));
        }
        */
        
        /*
        for (int i = 0; i < tokens.size(); ++i) {
            WordAnalysis result = morphology.analyze(tokens.get(i));
            Iterator<SingleAnalysis> iter = result.iterator();
            while( iter.hasNext() ){
                SingleAnalysis nextAnalysis = iter.next();
                System.out.println(nextAnalysis.formatLong());
//                System.out.println("\tStems = " + analysis.getStems());
                System.out.println("\tLemmas = " + nextAnalysis.getLemmas());
            }
            
        }
        */
        
        
//        generateTrainingFileFromCsv("data/hb.csv");

    }
    
    public static void testNerModelZ(PerceptronNer myNer, String testFileName) throws IOException {
    
    
        List<String> listOfString = readSentencesFromFile(testFileName);
    
        ListIterator<String> listIteratorOfStrings = listOfString.listIterator();
    
        int truePositive = 0, falsePositive = 0, trueNegative = 0, falseNegative = 0;
        double accuracy = 0.0, precision = 0.0, recall = 0.0;
    
        while( listIteratorOfStrings.hasNext() ){
            int index = listIteratorOfStrings.nextIndex() + 1;
        
            boolean checkFound = useNer(myNer,listIteratorOfStrings.next());
        
            if( index % 20 == 0 || index % 20 == 18 || index % 20 == 19)
            {
                if( checkFound )
                    truePositive++;
                else
                    falsePositive++;
            
            }
            else
            {
                if( !checkFound )
                    trueNegative++;
                else
                    falseNegative++;
            }
        
        }
    
        System.out.println("TruePositive: " + truePositive + "\nTrueNegative: " + trueNegative + "\nFalsePositive: " + falsePositive + "\nFalseNegative: " + falseNegative);
    
        accuracy = (double)(truePositive + trueNegative) / (double)(truePositive + trueNegative + falseNegative + falsePositive);
        precision = (double)(truePositive) / (double)(truePositive+falsePositive);
        recall = (double)(truePositive) / (double)(truePositive+falseNegative);
    
        System.out.println("Accuracy: " + accuracy + "\nPrecision: " + precision + "\nRecall: " + recall);
    
    
    }
    
    
    public static void generateTrainingFileFromCsv(String fileName) throws IOException {
    
        String[] HEADERS = { "Rating (Star)", "Review", "URL"};
    
        Reader in = new FileReader(fileName);
        Iterable<CSVRecord> records = CSVFormat.DEFAULT.withHeader(HEADERS).withFirstRecordAsHeader().parse(in);
    
        
        for (CSVRecord record : records) {
            String reviewStr = record.get("Review");
            
            String[] strArray = reviewStr.split(" ");
            
            List<String> strList = new ArrayList<>();
            
            int randomIndex = (int)Math.random() % strArray.length;
            
            int whichBadWordFile = (int)Math.random() % 2;
            
            
            
            
        }
        
        
    
    
    }
    
    
    public static void orderStringsInFileByNumOfWords(String fileName, String orderedFileName){
        try
        {
            File file = new File(fileName);    //creates a new file instance
            FileReader fr = new FileReader(file);   //reads the file
            BufferedReader br = new BufferedReader(fr);  //creates a buffering character input stream
    
    
    
            Set<String> hash_Set = new HashSet<String>();
            String line;
            int maxNumOfWords = 0;
            while((line = br.readLine()) != null)
            {
                if( hash_Set.contains(line) ){
                    System.out.println(line);
                }
                else{
                    
                    String[] strArray = line.split(" ");
                    int numOfWords = strArray.length;
    
                    if( maxNumOfWords < numOfWords ){
                        maxNumOfWords = numOfWords;
                    }
    
                    hash_Set.add(line);
                }
                
                
            }
            fr.close();
            
            
            List<String> listBlackList = new ArrayList<>();
            
            Iterator<String> iterHashSet = hash_Set.iterator();
            
            while( iterHashSet.hasNext() ){
                listBlackList.add(iterHashSet.next());
            }
    
    
            FileWriter fw = null;
            fw = new FileWriter(orderedFileName);
            
            for(int i = maxNumOfWords; i > 0; --i){
            
                List<String> listStr = new ArrayList<>();
                int sizeOfBlackList = listBlackList.size();
                
                for(int j = 0; j < sizeOfBlackList; ++j){
    
                    String strTemp = listBlackList.get(j);
                    String[] strArray = strTemp.split(" ");
                    
                    if( strArray.length == i ){
                        listStr.add(strTemp);
                    }
                    
                }
                
                java.util.Collections.sort(listStr);
                int sizeOfListStr = listStr.size();
                
                for(int k = 0; k < sizeOfListStr; ++k){
                    fw.write(listStr.get(k));
                    fw.write("\n");
                }
                
            
            }
            
            
            fw.close();
            
        }
        catch(IOException e)
        {
            e.printStackTrace();
        }
        
    }
    
    public static void tagSentencesFromFile(String fileName){
    
        try
        {
            File file = new File(fileName);    //creates a new file instance
            FileReader fr = new FileReader(file);   //reads the file
            BufferedReader br = new BufferedReader(fr);  //creates a buffering character input stream
    
    
            FileWriter fwTagged = null;
            String taggedFileName = "data/tagged_hb_training.txt";
            fwTagged = new FileWriter(taggedFileName);
            
            
            FileWriter fw = null;
            String notTaggedFileName = "data/notTagged_hb_training.txt";
            fw = new FileWriter(notTaggedFileName);
            
        
            String line;
            while((line = br.readLine()) != null)
            {
                String taggedSent = tagSentence(line);
                if( taggedSent != null ){
                    if( line.compareTo(taggedSent) == 0 ){
        
                        fw.write(line);
                        fw.write("\n");
        
                    }
                    else{
                        fwTagged.write(taggedSent);
                        fwTagged.write("\n");
                    }
                }
                
                
            }
            
            fr.close();
            fwTagged.close();
            fw.close();
            
        }
        catch(IOException e)
        {
            e.printStackTrace();
        }
        
        
    }
    
    public static String tagSentence(String sentence){
    
        List<String> tokensOfSentenceOrigin = tokenizeSentence(sentence);
        
        StringBuilder strBld = new StringBuilder();
        
        for(int i = 0; i < tokensOfSentenceOrigin.size(); ++i){
            strBld.append(tokensOfSentenceOrigin.get(i)+" ");
        }
        String strOrg = strBld.toString().trim();
        
        String lowerSentence = strOrg.toLowerCase(trLocale);
        
//        List<String> tokensOfSentence = tokenizeSentence(lowerSentence);
        List<String> tokensOfSentence = new ArrayList<>();
        String[] strArrayLower = lowerSentence.split(" ");
        for(int i = 0; i < strArrayLower.length; ++i){
            tokensOfSentence.add(strArrayLower[i]);
        }
        
        
        if( tokensOfSentenceOrigin.size() != tokensOfSentence.size() ){
            System.out.println("something wrong");
            System.out.println("origin:"+tokensOfSentenceOrigin.toString());
            System.out.println("lowerr:"+tokensOfSentence.toString());
            return null;
        }
        else{
    
            StringBuilder sentenceBuilder = new StringBuilder();
            
            String fileBads = "data/badsOrderedByNumOfWords.txt";    //creates a new file instance
            List<String> badList = readSentencesFromFile(fileBads);
    
    
            String fileBWords = "data/blackListShortOrderedByNumOfWords.txt";
            List<String> badWordList = readSentencesFromFile(fileBWords);
    
            int i = 0;
            int sizeTokensOfSentence = tokensOfSentence.size();
            while( i < sizeTokensOfSentence ){
        
                boolean badCheck = false;
                boolean badWordCheck = false;
        
                // bad list tagging
                int badListSize = badList.size();
                for(int badIndex = 0; badIndex < badListSize; ++badIndex ){
            
                    String[] strArray = badList.get(badIndex).split(" ");
                    int lenString = strArray.length;
            
                    if( lenString == 1 ){
                        if( tokensOfSentence.get(i).indexOf(strArray[0]) >= 0 ){
                            badCheck = true;
                            sentenceBuilder.append("[BAD "+tokensOfSentenceOrigin.get(i)+"] ");
                            ++i;
                            break;
                        }
                    }else if( lenString == 2 ){
                
                        if( i+1 < sizeTokensOfSentence ){
                    
                            StringBuilder strBuild = new StringBuilder();
                            strBuild.append(tokensOfSentence.get(i)+" "+tokensOfSentence.get(i+1));
                            String strToken = strBuild.toString();
                    
                            StringBuilder strBuild2 = new StringBuilder();
                            strBuild2.append(strArray[0]+" "+strArray[1]);
                            String strBad = strBuild2.toString();
                    
                            if( strToken.indexOf(strBad) >= 0 ){
                                badCheck = true;
                                StringBuilder strBuild3 = new StringBuilder();
                                strBuild3.append(tokensOfSentenceOrigin.get(i)+" "+tokensOfSentenceOrigin.get(i+1));
                                String strToken2 = strBuild3.toString();
                                sentenceBuilder.append("[BAD "+strToken2+"] ");
                                i += 2;
                                break;
                            }
                    
                        }
                
                
                    }else if( lenString == 3 ){
                
                        if( i+2 < sizeTokensOfSentence ){
                    
                            StringBuilder strBuild = new StringBuilder();
                            strBuild.append(tokensOfSentence.get(i)+" "+tokensOfSentence.get(i+1)+" "+tokensOfSentence.get(i+2));
                            String strToken = strBuild.toString();
                    
                            StringBuilder strBuild2 = new StringBuilder();
                            strBuild2.append(strArray[0]+" "+strArray[1]+" "+strArray[2]);
                            String strBad = strBuild2.toString();
                    
                            if( strToken.indexOf(strBad) >= 0 ){
                                badCheck = true;
                                StringBuilder strBuild3 = new StringBuilder();
                                strBuild3.append(tokensOfSentenceOrigin.get(i)+" "+tokensOfSentenceOrigin.get(i+1)+" "+tokensOfSentenceOrigin.get(i+2));
                                String strToken2 = strBuild3.toString();
                                sentenceBuilder.append("[BAD "+strToken2+"] ");
                                i += 3;
                                break;
                            }
                    
                        }
                
                
                    }else if( lenString == 4 ){
                
                        if( i+3 < sizeTokensOfSentence ){
                    
                            StringBuilder strBuild = new StringBuilder();
                            strBuild.append(tokensOfSentence.get(i)+" "+tokensOfSentence.get(i+1)+" "+tokensOfSentence.get(i+2)+" "+tokensOfSentence.get(i+3));
                            String strToken = strBuild.toString();
                    
                            StringBuilder strBuild2 = new StringBuilder();
                            strBuild2.append(strArray[0]+" "+strArray[1]+" "+strArray[2]+" "+strArray[3]);
                            String strBad = strBuild2.toString();
                    
                            if( strToken.indexOf(strBad) >= 0 ){
                                badCheck = true;
                                StringBuilder strBuild3 = new StringBuilder();
                                strBuild3.append(tokensOfSentenceOrigin.get(i)+" "+tokensOfSentenceOrigin.get(i+1)+" "+tokensOfSentenceOrigin.get(i+2)+" "+tokensOfSentenceOrigin.get(i+3));
                                String strToken2 = strBuild3.toString();
                                sentenceBuilder.append("[BAD "+strToken2+"] ");
                                i += 4;
                                break;
                            }
                    
                        }
                
                    }
            
            
                }
                // end of bad list tagging
        
                // bad word list tagging
                int badWordListSize = badWordList.size();
                if( !badCheck ){
            
                    for( int badWordIndex = 0; badWordIndex < badWordListSize; ++badWordIndex ){
                
                        String[] strArray = badWordList.get(badWordIndex).split(" ");
                        int lenString = strArray.length;
                
                        if( lenString == 1 ){
                            if( tokensOfSentence.get(i).compareTo(strArray[0]) == 0 ){
                                badWordCheck = true;
                                sentenceBuilder.append("[BWORD "+tokensOfSentenceOrigin.get(i)+"] ");
                                ++i;
                                break;
                            }
                        }else if( lenString == 2 ){
                    
                            if( i+1 < sizeTokensOfSentence ){
                        
                                StringBuilder strBuild = new StringBuilder();
                                strBuild.append(tokensOfSentence.get(i)+" "+tokensOfSentence.get(i+1));
                                String strToken = strBuild.toString();
                        
                                StringBuilder strBuild2 = new StringBuilder();
                                strBuild2.append(strArray[0]+" "+strArray[1]);
                                String strBad = strBuild2.toString();
                        
                                if( strToken.compareTo(strBad) == 0 ){
                                    badWordCheck = true;
                                    StringBuilder strBuild3 = new StringBuilder();
                                    strBuild3.append(tokensOfSentenceOrigin.get(i)+" "+tokensOfSentenceOrigin.get(i+1));
                                    String strToken2 = strBuild3.toString();
                                    sentenceBuilder.append("[BWORD "+strToken2+"] ");
                                    i += 2;
                                    break;
                                }
                        
                            }
                    
                    
                        }else if( lenString == 3 ){
                    
                            if( i+2 < sizeTokensOfSentence ){
                        
                                StringBuilder strBuild = new StringBuilder();
                                strBuild.append(tokensOfSentence.get(i)+" "+tokensOfSentence.get(i+1)+" "+tokensOfSentence.get(i+2));
                                String strToken = strBuild.toString();
                        
                                StringBuilder strBuild2 = new StringBuilder();
                                strBuild2.append(strArray[0]+" "+strArray[1]+" "+strArray[2]);
                                String strBad = strBuild2.toString();
                        
                                if( strToken.compareTo(strBad) == 0 ){
                                    badWordCheck = true;
                                    StringBuilder strBuild3 = new StringBuilder();
                                    strBuild3.append(tokensOfSentenceOrigin.get(i)+" "+tokensOfSentenceOrigin.get(i+1)+" "+tokensOfSentenceOrigin.get(i+2));
                                    String strToken2 = strBuild3.toString();
                                    sentenceBuilder.append("[BWORD "+strToken2+"] ");
                                    i += 3;
                                    break;
                                }
                        
                            }
                    
                    
                    
                        }else if( lenString == 4 ){
                    
                            if( i+3 < sizeTokensOfSentence ){
                        
                                StringBuilder strBuild = new StringBuilder();
                                strBuild.append(tokensOfSentence.get(i)+" "+tokensOfSentence.get(i+1)+" "+tokensOfSentence.get(i+2)+" "+tokensOfSentence.get(i+3));
                                String strToken = strBuild.toString();
                        
                                StringBuilder strBuild2 = new StringBuilder();
                                strBuild2.append(strArray[0]+" "+strArray[1]+" "+strArray[2]+" "+strArray[3]);
                                String strBad = strBuild2.toString();
                        
                                if( strToken.compareTo(strBad) == 0 ){
                                    badWordCheck = true;
                                    StringBuilder strBuild3 = new StringBuilder();
                                    strBuild3.append(tokensOfSentenceOrigin.get(i)+" "+tokensOfSentenceOrigin.get(i+1)+" "+tokensOfSentenceOrigin.get(i+2)+" "+tokensOfSentenceOrigin.get(i+3));
                                    String strToken2 = strBuild3.toString();
                                    sentenceBuilder.append("[BWORD "+strToken2+"] ");
                                    i += 4;
                                    break;
                                }
                        
                            }
                    
                        }
                
                    }
            
                }
                // end of bad word list tagging
        
                if( !badCheck && !badWordCheck ){
                    sentenceBuilder.append(tokensOfSentenceOrigin.get(i)+" ");
                    ++i;
                }
        
        
            }
    
    
            return sentenceBuilder.toString().trim();
        }
        
    }
    
    
    public static List<String> tokenizeSentence(String sentence){
        
        //        TurkishTokenizer tokenizer = TurkishTokenizer.DEFAULT;
        TurkishTokenizer tokenizer = TurkishTokenizer.builder().ignoreTypes(Token.Type.NewLine, Token.Type.SpaceTab).build();
        List<String> tokens = tokenizer.tokenizeToStrings(sentence);
    
//
//        for(int i = 0; i < tokens.size(); ++i){
//            System.out.println("token :"+tokens.get(i));
//        }
//
//        System.out.println("size :"+tokens.size());
        
        return tokens;
    }
    
    
    public static void generateNerModel(String trainingFile, String testFile, String modelFile) throws IOException {
        // you will need ner-train and ner-test files to run this example.
        
        Path trainPath = Paths.get(trainingFile);
        Path testPath = Paths.get(testFile);
        Path modelRoot = Paths.get(modelFile);
        
        NerDataSet trainingSet = NerDataSet.load(trainPath, NerDataSet.AnnotationStyle.BRACKET);
        Log.info(trainingSet.info()); // prints information
        
        NerDataSet testSet = NerDataSet.load(testPath, NerDataSet.AnnotationStyle.BRACKET);
        Log.info(testSet.info());
        
        TurkishMorphology morphology = TurkishMorphology.createWithDefaults();
        
        // Training occurs here. Result is a PerceptronNer instance.
        // There will be 7 iterations with 0.1 learning rate.
        PerceptronNer ner = new PerceptronNerTrainer(morphology).train(trainingSet, testSet, 7, 0.1f);
        
        Files.createDirectories(modelRoot);
        ner.saveModelAsText(modelRoot);
    }
    
    public static PerceptronNer generatePerceptronNer(String modelFile) throws IOException {
        // assumes you generated a model in my-model directory.
        Path modelRoot = Paths.get(modelFile);
        TurkishMorphology morphology = TurkishMorphology.createWithDefaults();
    
        PerceptronNer ner = PerceptronNer.loadModel(modelRoot, morphology);
        
        return ner;
    }
    
    public static boolean useNer(PerceptronNer ner, String sentence) throws IOException {
    
//        String sentence = "Ali Kaan yarın İstanbul'a gidecek.";
    
        System.out.println("Sentence:" + sentence);
        
        NerSentence result = ner.findNamedEntities(sentence);
        
        List<NamedEntity> namedEntities = result.getNamedEntities();
        
        for (NamedEntity namedEntity : namedEntities) {
            System.out.println(namedEntity);
        }
        
        return !namedEntities.isEmpty();
    }
    
    
    public static String splitByPunc(String sentence){
        
        List<String> tokensOfSentence = tokenizeSentence(sentence);
        
        StringBuilder stringBuilder = new StringBuilder();
        for(String token:tokensOfSentence) {
//            System.out.println("token: " + token);
            stringBuilder.append(token);
            stringBuilder.append(" ");
        }
        String sent = stringBuilder.toString().trim();
//        System.out.println("punc: "+sent);
        
        return sent;
    }
    
    
    public static void fileWrite(String fileName, List<String> sentences){
    
        // attach a file to FileWriter
        FileWriter fw = null;
        try {
            fw = new FileWriter(fileName);
            // read character wise from string and write
            // into FileWriter
            for (int i = 0; i < sentences.size(); i++){
                fw.write(sentences.get(i));
                fw.write("\n");
            }
            
            
            fw.close();
            //close the file
            
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        System.out.println("Writing successful");
        
    }
    
    public static List<String> readSentencesFromFile(String fileName){
        
        List<String> sentences = new ArrayList<>();
    
        try
        {
            File file = new File(fileName);    //creates a new file instance
            FileReader fr = new FileReader(file);   //reads the file
            BufferedReader br = new BufferedReader(fr);  //creates a buffering character input stream
            
            String line;
            while((line = br.readLine()) != null)
            {
                sentences.add(line.toLowerCase());
            }
            fr.close();    //closes the stream and release the resources
        }
        catch(IOException e)
        {
            e.printStackTrace();
        }
    
//        System.out.println("reading is done.");
    
        return sentences;
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
            File file = new File("data/results.txt");
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
            File myObj = new File("data/results.txt");
            Scanner myReader = new Scanner(myObj);
            int i = 1, truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0;
            double precision = 0.0;
            double recall = 0.0;
            double accuracy = 0.0;


            while (myReader.hasNextLine()) {
                String data = myReader.nextLine();
                if( i % 20 == 0 || i % 20 == 18 || i % 20 == 19)
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
            inputReader = new BufferedReader(new FileReader("data/real_train_set.arff"));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + "data/real_train_set.arff");
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


}