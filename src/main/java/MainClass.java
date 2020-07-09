
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.spec.RSAOtherPrimeInfo;
import java.util.*;
import java.util.stream.Collectors;

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
    
    private static List<Long> badSentenceIndexes = new ArrayList<>();
    private static List<Long> bWordSentenceIndexes = new ArrayList<>();

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


//        Path lookupRoot = Paths.get("data/normalization");
//        Path lmFile = Paths.get("data/lm/lm.2gram.slm");
        
//        TurkishMorphology morphology = TurkishMorphology.createWithDefaults();
//        TurkishMorphology morphology = TurkishMorphology.builder().setLexicon(RootLexicon.getDefault()).useInformalAnalysis().build();
//        TurkishSentenceNormalizer normalizer = new TurkishSentenceNormalizer(morphology, lookupRoot, lmFile);



//        orderStringsInFileByNumOfWords("data/blackListShort.txt","data/blackListShortOrderedByNumOfWords.txt");
//        orderStringsInFileByNumOfWords("data/bads.txt","data/badsOrderedByNumOfWords.txt");
//
//        tagSentencesFromFile("data/mk_hb_train_set2.txt");
//
//        generateNerModel("data/tagged_hb_training_suffix_2.txt","data/mk_hb_test_set_filtered_2.txt","data/my-hb-ner-model-with-suffix-2");
    
        PerceptronNer myNer = generatePerceptronNer("data/my-hb-ner-model-with-suffix-2");
        
        
        String sentence3 = "hepsiburadadanlar oruspu da gördüğüm ilan için aramıştım. a.q.";
        String sentence4 = "orospumusun a.q ederi 210.000₺ normalde, ben yine sana kıyak olsun diye 275.000₺ dedim";
        String sentence5 = "piç motor mu araç";
        String sentence6 = "https://urun.gittigidiyor.com/cep-telefonu";
        String sentence7 = "Hocam Merhabalar, ben sattım ama buradan ürünün sıfırını alabilirsiniz https://www.hepsisurda.com/case-4u-tanix-tx6-4k-hdr-tv-box-android-9-4-gb-ram-32-gb-hafiza-p-HBV00000JGIDZ";
        String sentence8 = "https://www.facebook.com/marketplace/item/496397154264376/ takas düşünür müsün";
        
        List<NamedEntity> listOfEntities = findNamedEntities(myNer,sentence3);

        System.out.println("Sentence:" + sentence3);
        for(NamedEntity entity : listOfEntities)
            System.out.println("entity:" + entity);
    
    
    
        String testFile1 = "data/mk_hb_test_set_filtered_2.txt";
//        cleanTestFile(testFile1);
//
//        testNerModelZ(myNer,testFile1);



        
        
//        System.out.println("before split test");
//        splitTestFileIntoTwoParts(testFile1);
//        System.out.println("after split test");
//
//        for(long lineForBad : badSentenceIndexes){
//            System.out.println(lineForBad);
//        }
//        for(long lineForBad : bWordSentenceIndexes){
//            System.out.println(lineForBad);
//        }
//
//        testBadFile(myNer,"data/mk_hb_test_set_BAD.txt");
//        testBWordFile(myNer,"data/mk_hb_test_set_BWORD.txt");
        


        
        
//        generateTrainingFileFromCsv("data/hb.csv");

    }
    
    
    public static void testBadFile(PerceptronNer perceptronNer, String fileName){
        List<String> listOfString = readSentencesFromFile(fileName);
    
        ListIterator<String> listIteratorOfStrings = listOfString.listIterator();
    
        long truePositive = 0, falsePositive = 0, trueNegative = 0, falseNegative = 0;
        double accuracy = 0.0, precision = 0.0, recall = 0.0;
    
        long lineNumber = 1;
        while( listIteratorOfStrings.hasNext() ){
            int index = listIteratorOfStrings.nextIndex() + 1;
            
            String sentence = listIteratorOfStrings.next();
            boolean checkFound = false;
            try {
                List<NamedEntity> entities = findNamedEntities(perceptronNer,sentence);
    
                // for debugging
//                if( lineNumber == 72 ) {
//                    System.out.println("sentence:" + sentence);
//                    for(NamedEntity entity : entities)
//                        System.out.println("entities:" + entity);
//                }
    
                // The values below must be changed according to the file
                if( badSentenceIndexes.contains(lineNumber) )
                {
                    if( !entities.isEmpty() )
                        truePositive++;
                    else
                        falsePositive++;
        
                }
                else
                {
                    if( entities.isEmpty() )
                        trueNegative++;
                    else
                        falseNegative++;
                }
    
                ++lineNumber;
    
            } catch (IOException e) {
                e.printStackTrace();
                break;
            }
            
            // for debugging
//            if( index == 1 )
//                System.out.println("sentence:" + sentence);
    
        }
    
        System.out.println("TruePositive: " + truePositive + "\nTrueNegative: " + trueNegative + "\nFalsePositive: " + falsePositive + "\nFalseNegative: " + falseNegative);
    
        accuracy = (double)(truePositive + trueNegative) / (double)(truePositive + trueNegative + falseNegative + falsePositive);
        precision = (double)(truePositive) / (double)(truePositive+falsePositive);
        recall = (double)(truePositive) / (double)(truePositive+falseNegative);
    
        System.out.println("Accuracy: " + accuracy + "\nPrecision: " + precision + "\nRecall: " + recall);
        
    }
    
    
    public static void testBWordFile(PerceptronNer perceptronNer, String fileName){
        List<String> listOfString = readSentencesFromFile(fileName);
    
        ListIterator<String> listIteratorOfStrings = listOfString.listIterator();
    
        long truePositive = 0, falsePositive = 0, trueNegative = 0, falseNegative = 0;
        double accuracy = 0.0, precision = 0.0, recall = 0.0;
    
        long lineNumber = 1;
        while( listIteratorOfStrings.hasNext() ){
            int index = listIteratorOfStrings.nextIndex() + 1;
        
            String sentence = listIteratorOfStrings.next();
            boolean checkFound = false;
            try {
                List<NamedEntity> entities = findNamedEntities(perceptronNer,sentence);
                
                // The values below must be changed according to the file
                if( bWordSentenceIndexes.contains(lineNumber) )
                {
                    if( !entities.isEmpty() )
                        truePositive++;
                    else
                        falsePositive++;
        
                }
                else
                {
                    if( entities.isEmpty() )
                        trueNegative++;
                    else
                        falseNegative++;
                }
    
                ++lineNumber;
                
            } catch (IOException e) {
                e.printStackTrace();
                break;
            }
            
        }
    
        System.out.println("TruePositive: " + truePositive + "\nTrueNegative: " + trueNegative + "\nFalsePositive: " + falsePositive + "\nFalseNegative: " + falseNegative);
    
        accuracy = (double)(truePositive + trueNegative) / (double)(truePositive + trueNegative + falseNegative + falsePositive);
        precision = (double)(truePositive) / (double)(truePositive+falsePositive);
        recall = (double)(truePositive) / (double)(truePositive+falseNegative);
    
        System.out.println("Accuracy: " + accuracy + "\nPrecision: " + precision + "\nRecall: " + recall);
        
    }
    
    
    public static List<Integer> hasBadOrBWord(String sentence){
        
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
//            System.out.println("something wrong");
//            System.out.println("origin:"+tokensOfSentenceOrigin.toString());
//            System.out.println("lowerr:"+tokensOfSentence.toString());
            return null;
        }
        else{
            
    
            ArrayList<Integer> listBadOrBWords = new ArrayList<>();
            
            
            String fileBads = "data/badsOrderedByNumOfWords.txt";    //creates a new file instance
            List<String> badList = readSentencesFromFile(fileBads);
            
            
            String fileBWords = "data/blackListShortOrderedByNumOfWords.txt";
            List<String> badWordList = readSentencesFromFile(fileBWords);
            
            boolean badExistInSentence = false;
            
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
                            badExistInSentence = true;
    
                            listBadOrBWords.add(1);
                            
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
                                badExistInSentence = true;
    
                                listBadOrBWords.add(1);
                                
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
                                badExistInSentence = true;
    
                                listBadOrBWords.add(1);
                                
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
                                badExistInSentence = true;
    
                                listBadOrBWords.add(1);
                                
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
                                badExistInSentence = true;
    
                                listBadOrBWords.add(2);
                                
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
                                    badExistInSentence = true;
    
                                    listBadOrBWords.add(2);
                                    
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
                                    badExistInSentence = true;
    
                                    listBadOrBWords.add(2);
                                    
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
                                    badExistInSentence = true;
    
                                    listBadOrBWords.add(2);
                                    
                                    i += 4;
                                    break;
                                }
                                
                            }
                            
                        }
                        
                    }
                    
                }
                // end of bad word list tagging
                
                if( badCheck && badWordCheck )
                    break;
    
                if( !badCheck && !badWordCheck ){
                    ++i;
                }
                
            }
            
            
            return listBadOrBWords;
        }
        
    }
    
    
    public static void cleanTestFile(String fileName){
    
        File file = new File(fileName);
        FileReader fr = null;   //reads the file
        try {
            fr = new FileReader(file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        BufferedReader br;
        if( fr != null){
            br = new BufferedReader(fr);  //creates a buffering character input stream
        }
        else{
            System.err.println("Couldn't open the file:" + fileName);
            return;
        }
    
        FileWriter fwFiltered = null;
        String testFileFiltered = "data/mk_hb_test_set_2_filtered_2.txt";
        try {
            fwFiltered = new FileWriter(testFileFiltered);
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        String line;
        long lineNumber = 1;
        while(true){
            try {
                line = br.readLine();
                String lineLower;
                if ( line == null )
                    break;
                else{
                
                    boolean isFound = false;
                    lineLower = line.toLowerCase(trLocale);
                    
                    String str1 = "hepsiburada .com";
                    String str2 = "hepsiburada.com";
                    String str3 = "hepsiburada";
                    
                    for(int i = 0; i < 3; ++i){
                        String strTemp;
                        if( i == 0 ){
                            strTemp = str1;
                        }
                        else if( i == 1 ){
                            strTemp = str2;
                        }
                        else{
                            strTemp = str3;
                        }
    
                        int indexOfString = lineLower.indexOf(strTemp);
                        if( indexOfString > -1 ){
                            // The values below must be changed according to the file
                            if( lineNumber % 20 == 1 || lineNumber % 20 == 2 || lineNumber % 20 == 3 ){
            
                                fwFiltered.write(line);
                                fwFiltered.write("\n");
            
                                ++lineNumber;
                                isFound = true;
                                break;
                            }
                            else{
            
                                StringBuilder stringBuilder = new StringBuilder();
                                stringBuilder.append(line.substring(0,indexOfString));
                                stringBuilder.append(line.substring(indexOfString + strTemp.length()));
            
                                fwFiltered.write(stringBuilder.toString());
                                fwFiltered.write("\n");
            
                                ++lineNumber;
                                isFound = true;
                                break;
                            }
                        }
                        
                    }
                    
                    if( !isFound ){
                        
                        fwFiltered.write(line);
                        fwFiltered.write("\n");
    
                        ++lineNumber;
                    }
                    
                
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
    
        }
    
        try {
            fwFiltered.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    
    
    }
    
    
    public static void splitTestFileIntoTwoParts(String fileName){
        
        
    
        File file = new File(fileName);    //creates a new file instance
        FileReader fr = null;   //reads the file
        try {
            fr = new FileReader(file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        BufferedReader br = new BufferedReader(fr);  //creates a buffering character input stream
    
    
        FileWriter fwBAD = null;
        String testFileBAD = "data/mk_hb_test_set_BAD.txt";
        try {
            fwBAD = new FileWriter(testFileBAD);
        } catch (IOException e) {
            e.printStackTrace();
        }
    
    
        FileWriter fwBWORD = null;
        String testFileBWORD = "data/mk_hb_test_set_BWORD.txt";
        try {
            fwBWORD = new FileWriter(testFileBWORD);
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        long lineNumberBad = 1;
        long lineNumberBWord = 1;
        String line;
        while(true){
            try {
                if ( (line = br.readLine()) == null )
                    break;
                else{
                    
                    List<Integer> listOfHasBadOrBWords = hasBadOrBWord(line);
                    
                    if( listOfHasBadOrBWords != null ){
                        if( listOfHasBadOrBWords.isEmpty() ){
    
                            fwBAD.write(line);
                            fwBAD.write("\n");
                            
                            fwBWORD.write(line);
                            fwBWORD.write("\n");
                            
                            ++lineNumberBad;
                            ++lineNumberBWord;
                        }
                        else{
                            for(int badOrBwordNum : listOfHasBadOrBWords ){
                                if( badOrBwordNum == 1 ){ // bad
                                    badSentenceIndexes.add(lineNumberBad);
                                    ++lineNumberBad;
                                    fwBAD.write(line);
                                    fwBAD.write("\n");
                                }
                                if( badOrBwordNum == 2 ){ // bword
                                    bWordSentenceIndexes.add(lineNumberBWord);
                                    ++lineNumberBWord;
                                    fwBWORD.write(line);
                                    fwBWORD.write("\n");
                                }
                            }
                        }
                    }
                    
                }
                
            } catch (IOException e) {
                e.printStackTrace();
            }
    
        }
        
        
        
    }
    
    
    public static void testNerModelZ(PerceptronNer myNer, String testFileName) throws IOException {
    
    
        List<String> listOfString = readSentencesFromFile(testFileName);
    
        ListIterator<String> listIteratorOfStrings = listOfString.listIterator();
    
        long truePositive = 0, falsePositive = 0, trueNegative = 0, falseNegative = 0;
        double accuracy = 0.0, precision = 0.0, recall = 0.0;
    
        while( listIteratorOfStrings.hasNext() ){
            int index = listIteratorOfStrings.nextIndex() + 1;
        
            List<NamedEntity> entities = findNamedEntities(myNer,listIteratorOfStrings.next());
        
            // The values below must be changed according to the file
            if( index % 20 == 1 || index % 20 == 2 || index % 20 == 3 )
            {
                if( !entities.isEmpty() )
                    truePositive++;
                else
                    falsePositive++;
            
            }
            else
            {
                if( entities.isEmpty() )
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
            String taggedFileName = "data/tagged_hb_training_suffix_2.txt";
            fwTagged = new FileWriter(taggedFileName);
            
            
            FileWriter fw = null;
            String notTaggedFileName = "data/notTagged_hb_training_suffix_2.txt";
            fw = new FileWriter(notTaggedFileName);
            
        
            long lineNumber = 1;
            String line;
            while((line = br.readLine()) != null)
            {
                List<String> listTaggedSent = tagSentence(line);
                if( listTaggedSent != null ){
                    
                    for(String taggedSent: listTaggedSent){
        
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
                    
                }
    
                System.out.println("lineNumber:" + lineNumber++);
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
    
    
    public static boolean isVowel(char ch){
        boolean isVow = false;
        switch (ch) {
            case 'a':
            case 'e':
            case 'ı':
            case 'i':
            case 'o':
            case 'ö':
            case 'u':
            case 'ü':
                isVow = true;
                break;
            default:
                isVow = false;
        }
        
        return isVow;
    }
    
    public static List<String> wordEdittedShorter5(String word){
        
        List<String> listOfStrings = new ArrayList<>();
        List<Character> listOfVowels = Arrays.asList('a', 'e', 'ı','i', 'o', 'ö','u','ü');
        String consonentsString = "bcçdfgğhjklmnprsştvyz";
        List<Character> listOfConsonents = consonentsString.chars().mapToObj(c -> (char) c).collect(Collectors.toList());
        
        Random random = new Random();
        
        int vowelIndexOfWord, consonentIndexOfWord;
    
        char wordCharArray[] = word.toCharArray();
        
        // remove one vowel
        while(true){
            vowelIndexOfWord = random.nextInt(word.length());
            if( listOfVowels.contains(wordCharArray[vowelIndexOfWord]) ){
                break;
            }
        }
        String wordWithoutOneVowel = word.substring(0, vowelIndexOfWord) + word.substring(vowelIndexOfWord + 1);
        listOfStrings.add(wordWithoutOneVowel);
    
    
        // remove one consonent
//        while(true){
//            consonentIndexOfWord = random.nextInt(word.length());
//            if( listOfConsonents.contains(wordCharArray[consonentIndexOfWord]) ){
//                break;
//            }
//        }
//        String wordWithoutOneConsonent = word.substring(0, consonentIndexOfWord) + word.substring(consonentIndexOfWord + 1);
//        listOfStrings.add(wordWithoutOneConsonent);
        
        
        return listOfStrings;
    }
    
    
    public static List<String> wordEditted(String word){
    
        List<String> listOfStrings = new ArrayList<>();
        
        List<Character> listOfVowels = Arrays.asList('a', 'e', 'ı','i', 'o', 'ö','u','ü');
        String consonentsString = "bcçdfgğhjklmnprsştvyz";
        List<Character> listOfConsonents = consonentsString.chars().mapToObj(c -> (char) c).collect(Collectors.toList());
        
        Random random = new Random();
        
        int vowelIndexOfWord, consonentIndexOfWord;
        
        char wordCharArray[] = word.toCharArray();
        boolean vowelExist = false;
        boolean consonantExist = false;
        
        // remove one vowel
        for(char ch : wordCharArray){
            if( listOfVowels.contains(ch) ){
                vowelExist = true;
                break;
            }
        }
        if( vowelExist ) {
            
            while(true){
              vowelIndexOfWord = random.nextInt(word.length());
              if( listOfVowels.contains(wordCharArray[vowelIndexOfWord]) ){
                  break;
              }
            }
            String wordWithoutOneVowel = word.substring(0, vowelIndexOfWord) + word.substring(vowelIndexOfWord + 1);
            listOfStrings.add(wordWithoutOneVowel);
            
        }
        
    
        // remove one consonent
        for(char ch : wordCharArray){
            if( listOfConsonents.contains(ch) ){
                consonantExist = true;
                break;
            }
        }
        if( consonantExist ){
            
            while(true){
                consonentIndexOfWord = random.nextInt(word.length());
                if( listOfConsonents.contains(wordCharArray[consonentIndexOfWord]) ){
                    break;
                }
            }
            String wordWithoutOneConsonent = word.substring(0, consonentIndexOfWord) + word.substring(consonentIndexOfWord + 1);
            listOfStrings.add(wordWithoutOneConsonent);
            
        }
        
        
    
        // changing a vowel
        if( vowelExist ){
            
            while(true){
                vowelIndexOfWord = random.nextInt(word.length());
                if( listOfVowels.contains(wordCharArray[vowelIndexOfWord]) ){
                    break;
                }
            }
            int vowel2Index;
            while(true){
                vowel2Index = random.nextInt(listOfVowels.size());
                if( listOfVowels.get(vowel2Index) != wordCharArray[vowelIndexOfWord] ){
                    break;
                }
            }
            String wordChangedOneVowel = word.substring(0, vowelIndexOfWord) + Character.toString(listOfVowels.get(vowel2Index)) + word.substring(vowelIndexOfWord + 1);
            listOfStrings.add(wordChangedOneVowel);
    
            
    
            // changing a vowel with star sign
            while(true){
                vowelIndexOfWord = random.nextInt(word.length());
                if( listOfVowels.contains(wordCharArray[vowelIndexOfWord]) ){
                    break;
                }
            }
            int vowel3Index;
            while(true){
                vowel3Index = random.nextInt(listOfVowels.size());
                if( listOfVowels.get(vowel3Index) != wordCharArray[vowelIndexOfWord] ){
                    break;
                }
            }
            String wordChangedStar = word.substring(0, vowelIndexOfWord) + "**" + word.substring(vowelIndexOfWord + 1);
            listOfStrings.add(wordChangedStar);
        
        
        }
        
        
    
    
        // changing a consonent
        if( consonantExist ){
            
            while(true){
                consonentIndexOfWord = random.nextInt(word.length());
                if( listOfConsonents.contains(wordCharArray[consonentIndexOfWord]) ){
                    break;
                }
            }
            int consonent2Index;
            while(true){
                consonent2Index = random.nextInt(listOfConsonents.size());
                if( listOfConsonents.get(consonent2Index) != wordCharArray[consonentIndexOfWord] ){
                    break;
                }
            }
            String wordChangeOneConsonent = word.substring(0, consonentIndexOfWord) + Character.toString(listOfConsonents.get(consonent2Index)) + word.substring(consonentIndexOfWord + 1);
            listOfStrings.add(wordChangeOneConsonent);
        
        }
        
        
        
        return listOfStrings;
    }
    
    
    public static List<String> tagSentence(String sentence){
    
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
//            System.out.println("something wrong");
//            System.out.println("origin:"+tokensOfSentenceOrigin.toString());
//            System.out.println("lowerr:"+tokensOfSentence.toString());
            return null;
        }
        else{
            
            List<String> listOfTaggedStrings = new ArrayList<>();
    
            StringBuilder sentenceBuilder = new StringBuilder();
            StringBuilder sentenceBuilder2 = new StringBuilder();
            StringBuilder sentenceBuilder3 = new StringBuilder();
            StringBuilder sentenceBuilder4 = new StringBuilder();
            StringBuilder sentenceBuilder5 = new StringBuilder();
            StringBuilder sentenceBuilder6 = new StringBuilder();
            StringBuilder sentenceBuilder7 = new StringBuilder();
            StringBuilder sentenceBuilder8 = new StringBuilder();
            StringBuilder sentenceBuilder9 = new StringBuilder();
            StringBuilder sentenceBuilder10 = new StringBuilder();
    
            List<String> listOfSuffix1ForBad = new ArrayList<>();
            listOfSuffix1ForBad.add("a");
            listOfSuffix1ForBad.add("e");
            List<String> listOfSuffix2ForBad = new ArrayList<>();
            listOfSuffix2ForBad.add("de");
            listOfSuffix2ForBad.add("da");
            listOfSuffix2ForBad.add("te");
            listOfSuffix2ForBad.add("ta");
            listOfSuffix2ForBad.add("ye");
            listOfSuffix2ForBad.add("ya");
            List<String> listOfSuffix3ForBad = new ArrayList<>();
            listOfSuffix3ForBad.add("den");
            listOfSuffix3ForBad.add("dan");
            listOfSuffix3ForBad.add("ten");
            listOfSuffix3ForBad.add("tan");
            
            
            
            String fileBads = "data/badsOrderedByNumOfWords.txt";    //creates a new file instance
            List<String> badList = readSentencesFromFile(fileBads);
    
    
            String fileBWords = "data/blackListShortOrderedByNumOfWords.txt";
            List<String> badWordList = readSentencesFromFile(fileBWords);
            
            boolean badExistInSentence = false;
            boolean s6check = false;
            boolean s7check = false;
            boolean s8check = false;
            boolean s9check = false;
            boolean s10check = false;
    
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
                            badExistInSentence = true;
                            sentenceBuilder.append("[BAD "+tokensOfSentenceOrigin.get(i)+"] ");
                            
                            int lengthOfStr = strArray[0].length();
                            if( lengthOfStr >= 5 && lengthOfStr < 10 ){
                                
                                List<String> listOfStringsEditted = wordEditted(tokensOfSentence.get(i));
                                
                                for(int m = 0; m < listOfStringsEditted.size(); ++m){
                                    if( m == 0 ){ // str1
                                        sentenceBuilder6.append("[BAD " + listOfStringsEditted.get(m) + "] ");
                                        
                                        s6check = true;
                                    }
                                    else if( m == 1 ){
                                        sentenceBuilder7.append("[BAD " + listOfStringsEditted.get(m) + "] ");
    
                                        s7check = true;
                                    }
                                    else if( m == 2 ){
                                        sentenceBuilder8.append("[BAD " + listOfStringsEditted.get(m) + "] ");
    
                                        s8check = true;
                                    }
                                    else if( m == 3 ){
                                        sentenceBuilder9.append("[BAD " + listOfStringsEditted.get(m) + "] ");
    
                                        s9check = true;
                                    }
                                    else{
                                        sentenceBuilder10.append("[BAD " + listOfStringsEditted.get(m) + "] ");
    
                                        s10check = true;
                                    }
                                }
                                
    
                                Random rand = new Random();
    
                                //  for suffix1
                                int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                sentenceBuilder2.append("[BAD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix1ForBad.get(randIndex) +"] ");
                                
                                int randIndex2 = (randIndex + 1) % listOfSuffix1ForBad.size();
                                sentenceBuilder3.append("[BAD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix1ForBad.get(randIndex2) +"] ");
                                
                                //  for suffix2
                                int randIndex3 = rand.nextInt(listOfSuffix2ForBad.size());
                                sentenceBuilder4.append("[BAD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix2ForBad.get(randIndex3) +"] ");
                                
                                int randIndex4 = (randIndex3 * 3 + 5) % listOfSuffix2ForBad.size();
                                sentenceBuilder5.append("[BAD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix2ForBad.get(randIndex4) +"] ");
                                
                            }
                            else if( lengthOfStr >= 10 ){
    
                                List<String> listOfStringsEditted = wordEditted(tokensOfSentence.get(i));
    
                                for(int m = 0; m < listOfStringsEditted.size(); ++m){
                                    if( m == 0 ){ // str1
                                        sentenceBuilder6.append("[BAD " + listOfStringsEditted.get(m) + "] ");
            
                                        s6check = true;
                                    }
                                    else if( m == 1 ){
                                        sentenceBuilder7.append("[BAD " + listOfStringsEditted.get(m) + "] ");
            
                                        s7check = true;
                                    }
                                    else if( m == 2 ){
                                        sentenceBuilder8.append("[BAD " + listOfStringsEditted.get(m) + "] ");
            
                                        s8check = true;
                                    }
                                    else if( m == 3 ){
                                        sentenceBuilder9.append("[BAD " + listOfStringsEditted.get(m) + "] ");
    
                                        s9check = true;
                                    }
                                    else{
                                        sentenceBuilder10.append("[BAD " + listOfStringsEditted.get(m) + "] ");
    
                                        s10check = true;
                                    }
                                }
    
                                Random rand = new Random();
    
                                //  for suffix1
                                int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                sentenceBuilder2.append("[BAD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix1ForBad.get(randIndex) +"] ");
    
                                //  for suffix2
                                int randIndex3 = rand.nextInt(listOfSuffix2ForBad.size());
                                sentenceBuilder4.append("[BAD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix2ForBad.get(randIndex3) +"] ");
                                
                                int randIndex4 = (randIndex3 * 3 + 5) % listOfSuffix2ForBad.size();
                                sentenceBuilder5.append("[BAD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix2ForBad.get(randIndex4) +"] ");
    
                                //  for suffix3
                                int randIndex2 = rand.nextInt(listOfSuffix3ForBad.size());
                                sentenceBuilder3.append("[BAD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix3ForBad.get(randIndex2) +"] ");
                                
                            }
                            else{ // length < 5
    
//                                List<String> listOfStringsEditted = wordEdittedShorter5(tokensOfSentence.get(i));
//
//                                for(int m = 0; m < listOfStringsEditted.size(); ++m){
//                                    if( m == 0 ){ // str1
//                                        sentenceBuilder6.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                        s6check = true;
//                                    }
//                                    else if( m == 1 ){
//                                        sentenceBuilder7.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                        s7check = true;
//                                    }
//                                    else if( m == 2 ){
//                                        sentenceBuilder8.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                        s8check = true;
//                                    }
//                                    else if( m == 3 ){
//                                        sentenceBuilder9.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                        s9check = true;
//                                    }
//                                    else{
//                                        sentenceBuilder10.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                        s10check = true;
//                                    }
//                                }
                                
                                Random rand = new Random();
    
                                //  for suffix1
                                int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                sentenceBuilder2.append("[BAD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix1ForBad.get(randIndex) +"] ");
    
                                int randIndex2 = (randIndex + 1) % listOfSuffix1ForBad.size();
                                sentenceBuilder3.append("[BAD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix1ForBad.get(randIndex2) +"] ");
                                
                                // for origin
                                sentenceBuilder4.append("[BAD "+tokensOfSentenceOrigin.get(i)+"] ");
                                sentenceBuilder5.append("[BAD "+tokensOfSentenceOrigin.get(i)+"] ");
                                
                            }
                            
                            
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
                                badExistInSentence = true;
                                StringBuilder strBuild3 = new StringBuilder();
                                strBuild3.append(tokensOfSentenceOrigin.get(i)+" "+tokensOfSentenceOrigin.get(i+1));
                                String strToken2 = strBuild3.toString();
                                sentenceBuilder.append("[BAD "+strToken2+"] ");
    
                                int lengthOfStr = strBad.length();
                                if( lengthOfStr >= 5 && lengthOfStr < 10 ){
    
//                                    List<String> listOfStringsEditted = wordEditted(strToken);
//
//                                    for(int m = 0; m < listOfStringsEditted.size(); ++m){
//                                        if( m == 0 ){ // str1
//                                            sentenceBuilder6.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s6check = true;
//                                        }
//                                        else if( m == 1 ){
//                                            sentenceBuilder7.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s7check = true;
//                                        }
//                                        else if( m == 2 ){
//                                            sentenceBuilder8.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s8check = true;
//                                        }
//                                        else if( m == 3 ){
//                                            sentenceBuilder9.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s9check = true;
//                                        }
//                                        else{
//                                            sentenceBuilder10.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s10check = true;
//                                        }
//                                    }
        
                                    Random rand = new Random();
        
                                    //  for suffix1
                                    int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                    sentenceBuilder2.append("[BAD "+ strToken2 + listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                    int randIndex2 = (randIndex + 1) % listOfSuffix1ForBad.size();
                                    sentenceBuilder3.append("[BAD "+ strToken2 + listOfSuffix1ForBad.get(randIndex2) +"] ");
        
                                    //  for suffix2
                                    int randIndex3 = rand.nextInt(listOfSuffix2ForBad.size());
                                    sentenceBuilder4.append("[BAD "+ strToken2 + listOfSuffix2ForBad.get(randIndex3) +"] ");
        
                                    int randIndex4 = (randIndex3 * 3 + 5) % listOfSuffix2ForBad.size();
                                    sentenceBuilder5.append("[BAD "+ strToken2 + listOfSuffix2ForBad.get(randIndex4) +"] ");
        
                                }
                                else if( lengthOfStr >= 10 ){
    
                                    List<String> listOfStringsEditted = wordEditted(strToken);
    
                                    for(int m = 0; m < listOfStringsEditted.size(); ++m){
                                        if( m == 0 ){ // str1
                                            sentenceBuilder6.append("[BAD " + listOfStringsEditted.get(m) + "] ");
            
                                            s6check = true;
                                        }
                                        else if( m == 1 ){
                                            sentenceBuilder7.append("[BAD " + listOfStringsEditted.get(m) + "] ");
            
                                            s7check = true;
                                        }
                                        else if( m == 2 ){
                                            sentenceBuilder8.append("[BAD " + listOfStringsEditted.get(m) + "] ");
            
                                            s8check = true;
                                        }
                                        else if( m == 3 ){
                                            sentenceBuilder9.append("[BAD " + listOfStringsEditted.get(m) + "] ");
    
                                            s9check = true;
                                        }
                                        else{
                                            sentenceBuilder10.append("[BAD " + listOfStringsEditted.get(m) + "] ");
    
                                            s10check = true;
                                        }
                                    }
        
                                    Random rand = new Random();
        
                                    //  for suffix1
                                    int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                    sentenceBuilder2.append("[BAD "+ strToken2 + listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                    //  for suffix2
                                    int randIndex3 = rand.nextInt(listOfSuffix2ForBad.size());
                                    sentenceBuilder4.append("[BAD "+ strToken2 + listOfSuffix2ForBad.get(randIndex3) +"] ");
        
                                    int randIndex4 = (randIndex3 * 3 + 5) % listOfSuffix2ForBad.size();
                                    sentenceBuilder5.append("[BAD "+ strToken2 + listOfSuffix2ForBad.get(randIndex4) +"] ");
        
                                    //  for suffix3
                                    int randIndex2 = rand.nextInt(listOfSuffix3ForBad.size());
                                    sentenceBuilder3.append("[BAD "+ strToken2 + listOfSuffix3ForBad.get(randIndex2) +"] ");
        
                                }
                                else{ // length < 5
    
//                                    List<String> listOfStringsEditted = wordEdittedShorter5(strToken);
//
//                                    for(int m = 0; m < listOfStringsEditted.size(); ++m){
//                                        if( m == 0 ){ // str1
//                                            sentenceBuilder6.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s6check = true;
//                                        }
//                                        else if( m == 1 ){
//                                            sentenceBuilder7.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s7check = true;
//                                        }
//                                        else if( m == 2 ){
//                                            sentenceBuilder8.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s8check = true;
//                                        }
//                                        else if( m == 3 ){
//                                            sentenceBuilder9.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s9check = true;
//                                        }
//                                        else{
//                                            sentenceBuilder10.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s10check = true;
//                                        }
//                                    }
        
                                    Random rand = new Random();
        
                                    //  for suffix1
                                    int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                    sentenceBuilder2.append("[BAD "+ strToken2 + listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                    int randIndex2 = (randIndex + 1) % listOfSuffix1ForBad.size();
                                    sentenceBuilder3.append("[BAD "+ strToken2 + listOfSuffix1ForBad.get(randIndex2) +"] ");
        
                                    // for origin
                                    sentenceBuilder4.append("[BAD "+ strToken2 +"] ");
                                    sentenceBuilder5.append("[BAD "+ strToken2 +"] ");
        
                                }
                                
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
                                badExistInSentence = true;
                                StringBuilder strBuild3 = new StringBuilder();
                                strBuild3.append(tokensOfSentenceOrigin.get(i)+" "+tokensOfSentenceOrigin.get(i+1)+" "+tokensOfSentenceOrigin.get(i+2));
                                String strToken2 = strBuild3.toString();
                                sentenceBuilder.append("[BAD "+strToken2+"] ");
    
                                int lengthOfStr = strBad.length();
                                if( lengthOfStr >= 5 && lengthOfStr < 10 ){
    
//                                    List<String> listOfStringsEditted = wordEditted(strToken);
//
//                                    for(int m = 0; m < listOfStringsEditted.size(); ++m){
//                                        if( m == 0 ){ // str1
//                                            sentenceBuilder6.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s6check = true;
//                                        }
//                                        else if( m == 1 ){
//                                            sentenceBuilder7.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s7check = true;
//                                        }
//                                        else if( m == 2 ){
//                                            sentenceBuilder8.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s8check = true;
//                                        }
//                                        else if( m == 3 ){
//                                            sentenceBuilder9.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s9check = true;
//                                        }
//                                        else{
//                                            sentenceBuilder10.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s10check = true;
//                                        }
//                                    }
        
                                    Random rand = new Random();
        
                                    //  for suffix1
                                    int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                    sentenceBuilder2.append("[BAD "+ strToken2 + listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                    int randIndex2 = (randIndex + 1) % listOfSuffix1ForBad.size();
                                    sentenceBuilder3.append("[BAD "+ strToken2 + listOfSuffix1ForBad.get(randIndex2) +"] ");
        
                                    //  for suffix2
                                    int randIndex3 = rand.nextInt(listOfSuffix2ForBad.size());
                                    sentenceBuilder4.append("[BAD "+ strToken2 + listOfSuffix2ForBad.get(randIndex3) +"] ");
        
                                    int randIndex4 = (randIndex3 * 3 + 5) % listOfSuffix2ForBad.size();
                                    sentenceBuilder5.append("[BAD "+ strToken2 + listOfSuffix2ForBad.get(randIndex4) +"] ");
        
                                }
                                else if( lengthOfStr >= 10 ){
    
                                    List<String> listOfStringsEditted = wordEditted(strToken);
    
                                    for(int m = 0; m < listOfStringsEditted.size(); ++m){
                                        if( m == 0 ){ // str1
                                            sentenceBuilder6.append("[BAD " + listOfStringsEditted.get(m) + "] ");
            
                                            s6check = true;
                                        }
                                        else if( m == 1 ){
                                            sentenceBuilder7.append("[BAD " + listOfStringsEditted.get(m) + "] ");
            
                                            s7check = true;
                                        }
                                        else if( m == 2 ){
                                            sentenceBuilder8.append("[BAD " + listOfStringsEditted.get(m) + "] ");
            
                                            s8check = true;
                                        }
                                        else if( m == 3 ){
                                            sentenceBuilder9.append("[BAD " + listOfStringsEditted.get(m) + "] ");
    
                                            s9check = true;
                                        }
                                        else{
                                            sentenceBuilder10.append("[BAD " + listOfStringsEditted.get(m) + "] ");
    
                                            s10check = true;
                                        }
                                    }
        
                                    Random rand = new Random();
        
                                    //  for suffix1
                                    int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                    sentenceBuilder2.append("[BAD "+ strToken2 + listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                    //  for suffix2
                                    int randIndex3 = rand.nextInt(listOfSuffix2ForBad.size());
                                    sentenceBuilder4.append("[BAD "+ strToken2 + listOfSuffix2ForBad.get(randIndex3) +"] ");
        
                                    int randIndex4 = (randIndex3 * 3 + 5) % listOfSuffix2ForBad.size();
                                    sentenceBuilder5.append("[BAD "+ strToken2 + listOfSuffix2ForBad.get(randIndex4) +"] ");
        
                                    //  for suffix3
                                    int randIndex2 = rand.nextInt(listOfSuffix3ForBad.size());
                                    sentenceBuilder3.append("[BAD "+ strToken2 + listOfSuffix3ForBad.get(randIndex2) +"] ");
        
                                }
                                else{ // length < 5
    
//                                    List<String> listOfStringsEditted = wordEdittedShorter5(strToken);
//
//                                    for(int m = 0; m < listOfStringsEditted.size(); ++m){
//                                        if( m == 0 ){ // str1
//                                            sentenceBuilder6.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s6check = true;
//                                        }
//                                        else if( m == 1 ){
//                                            sentenceBuilder7.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s7check = true;
//                                        }
//                                        else if( m == 2 ){
//                                            sentenceBuilder8.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s8check = true;
//                                        }
//                                        else if( m == 3 ){
//                                            sentenceBuilder9.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s9check = true;
//                                        }
//                                        else{
//                                            sentenceBuilder10.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s10check = true;
//                                        }
//                                    }
        
                                    Random rand = new Random();
        
                                    //  for suffix1
                                    int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                    sentenceBuilder2.append("[BAD "+ strToken2 + listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                    int randIndex2 = (randIndex + 1) % listOfSuffix1ForBad.size();
                                    sentenceBuilder3.append("[BAD "+ strToken2 + listOfSuffix1ForBad.get(randIndex2) +"] ");
        
                                    // for origin
                                    sentenceBuilder4.append("[BAD "+ strToken2 +"] ");
                                    sentenceBuilder5.append("[BAD "+ strToken2 +"] ");
        
                                }
                                
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
                                badExistInSentence = true;
                                StringBuilder strBuild3 = new StringBuilder();
                                strBuild3.append(tokensOfSentenceOrigin.get(i)+" "+tokensOfSentenceOrigin.get(i+1)+" "+tokensOfSentenceOrigin.get(i+2)+" "+tokensOfSentenceOrigin.get(i+3));
                                String strToken2 = strBuild3.toString();
                                sentenceBuilder.append("[BAD "+strToken2+"] ");
    
                                int lengthOfStr = strBad.length();
                                if( lengthOfStr >= 5 && lengthOfStr < 10 ){
    
//                                    List<String> listOfStringsEditted = wordEditted(strToken);
//
//                                    for(int m = 0; m < listOfStringsEditted.size(); ++m){
//                                        if( m == 0 ){ // str1
//                                            sentenceBuilder6.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s6check = true;
//                                        }
//                                        else if( m == 1 ){
//                                            sentenceBuilder7.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s7check = true;
//                                        }
//                                        else if( m == 2 ){
//                                            sentenceBuilder8.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s8check = true;
//                                        }
//                                        else if( m == 3 ){
//                                            sentenceBuilder9.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s9check = true;
//                                        }
//                                        else{
//                                            sentenceBuilder10.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s10check = true;
//                                        }
//                                    }
        
                                    Random rand = new Random();
        
                                    //  for suffix1
                                    int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                    sentenceBuilder2.append("[BAD "+ strToken2 + listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                    int randIndex2 = (randIndex + 1) % listOfSuffix1ForBad.size();
                                    sentenceBuilder3.append("[BAD "+ strToken2 + listOfSuffix1ForBad.get(randIndex2) +"] ");
        
                                    //  for suffix2
                                    int randIndex3 = rand.nextInt(listOfSuffix2ForBad.size());
                                    sentenceBuilder4.append("[BAD "+ strToken2 + listOfSuffix2ForBad.get(randIndex3) +"] ");
        
                                    int randIndex4 = (randIndex3 * 3 + 5) % listOfSuffix2ForBad.size();
                                    sentenceBuilder5.append("[BAD "+ strToken2 + listOfSuffix2ForBad.get(randIndex4) +"] ");
        
                                }
                                else if( lengthOfStr >= 10 ){
    
                                    List<String> listOfStringsEditted = wordEditted(strToken);
    
                                    for(int m = 0; m < listOfStringsEditted.size(); ++m){
                                        if( m == 0 ){ // str1
                                            sentenceBuilder6.append("[BAD " + listOfStringsEditted.get(m) + "] ");
            
                                            s6check = true;
                                        }
                                        else if( m == 1 ){
                                            sentenceBuilder7.append("[BAD " + listOfStringsEditted.get(m) + "] ");
            
                                            s7check = true;
                                        }
                                        else if( m == 2 ){
                                            sentenceBuilder8.append("[BAD " + listOfStringsEditted.get(m) + "] ");
            
                                            s8check = true;
                                        }
                                        else if( m == 3 ){
                                            sentenceBuilder9.append("[BAD " + listOfStringsEditted.get(m) + "] ");
    
                                            s9check = true;
                                        }
                                        else{
                                            sentenceBuilder10.append("[BAD " + listOfStringsEditted.get(m) + "] ");
    
                                            s10check = true;
                                        }
                                    }
        
                                    Random rand = new Random();
        
                                    //  for suffix1
                                    int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                    sentenceBuilder2.append("[BAD "+ strToken2 + listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                    //  for suffix2
                                    int randIndex3 = rand.nextInt(listOfSuffix2ForBad.size());
                                    sentenceBuilder4.append("[BAD "+ strToken2 + listOfSuffix2ForBad.get(randIndex3) +"] ");
        
                                    int randIndex4 = (randIndex3 * 3 + 5) % listOfSuffix2ForBad.size();
                                    sentenceBuilder5.append("[BAD "+ strToken2 + listOfSuffix2ForBad.get(randIndex4) +"] ");
        
                                    //  for suffix3
                                    int randIndex2 = rand.nextInt(listOfSuffix3ForBad.size());
                                    sentenceBuilder3.append("[BAD "+ strToken2 + listOfSuffix3ForBad.get(randIndex2) +"] ");
        
                                }
                                else{ // length < 5
    
//                                    List<String> listOfStringsEditted = wordEdittedShorter5(strToken);
//
//                                    for(int m = 0; m < listOfStringsEditted.size(); ++m){
//                                        if( m == 0 ){ // str1
//                                            sentenceBuilder6.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s6check = true;
//                                        }
//                                        else if( m == 1 ){
//                                            sentenceBuilder7.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s7check = true;
//                                        }
//                                        else if( m == 2 ){
//                                            sentenceBuilder8.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s8check = true;
//                                        }
//                                        else if( m == 3 ){
//                                            sentenceBuilder9.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s9check = true;
//                                        }
//                                        else{
//                                            sentenceBuilder10.append("[BAD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s10check = true;
//                                        }
//                                    }
        
                                    Random rand = new Random();
        
                                    //  for suffix1
                                    int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                    sentenceBuilder2.append("[BAD "+ strToken2 + listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                    int randIndex2 = (randIndex + 1) % listOfSuffix1ForBad.size();
                                    sentenceBuilder3.append("[BAD "+ strToken2 + listOfSuffix1ForBad.get(randIndex2) +"] ");
        
                                    // for origin
                                    sentenceBuilder4.append("[BAD "+ strToken2 +"] ");
                                    sentenceBuilder5.append("[BAD "+ strToken2 +"] ");
        
                                }
                                
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
                                badExistInSentence = true;
                                sentenceBuilder.append("[BWORD "+tokensOfSentenceOrigin.get(i)+"] ");
    
                                int lengthOfStr = strArray[0].length();
                                if( lengthOfStr >= 5 && lengthOfStr < 10 ){
    
                                    List<String> listOfStringsEditted = wordEditted(tokensOfSentenceOrigin.get(i));
    
                                    for(int m = 0; m < listOfStringsEditted.size(); ++m){
                                        if( m == 0 ){ // str1
                                            sentenceBuilder6.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
            
                                            s6check = true;
                                        }
                                        else if( m == 1 ){
                                            sentenceBuilder7.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
            
                                            s7check = true;
                                        }
                                        else if( m == 2 ){
                                            sentenceBuilder8.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
            
                                            s8check = true;
                                        }
                                        else if( m == 3 ){
                                            sentenceBuilder9.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
    
                                            s9check = true;
                                        }
                                        else{
                                            sentenceBuilder10.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
    
                                            s10check = true;
                                        }
                                    }
        
                                    Random rand = new Random();
        
                                    //  for suffix1
                                    int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                    sentenceBuilder2.append("[BWORD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                    int randIndex2 = (randIndex + 1) % listOfSuffix1ForBad.size();
                                    sentenceBuilder3.append("[BWORD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix1ForBad.get(randIndex2) +"] ");
        
                                    //  for suffix2
                                    int randIndex3 = rand.nextInt(listOfSuffix2ForBad.size());
                                    sentenceBuilder4.append("[BWORD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix2ForBad.get(randIndex3) +"] ");
        
                                    int randIndex4 = (randIndex3 * 3 + 5) % listOfSuffix2ForBad.size();
                                    sentenceBuilder5.append("[BWORD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix2ForBad.get(randIndex4) +"] ");
        
                                }
                                else if( lengthOfStr >= 10 ){
    
                                    List<String> listOfStringsEditted = wordEditted(tokensOfSentenceOrigin.get(i));
    
                                    for(int m = 0; m < listOfStringsEditted.size(); ++m){
                                        if( m == 0 ){ // str1
                                            sentenceBuilder6.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
            
                                            s6check = true;
                                        }
                                        else if( m == 1 ){
                                            sentenceBuilder7.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
            
                                            s7check = true;
                                        }
                                        else if( m == 2 ){
                                            sentenceBuilder8.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
            
                                            s8check = true;
                                        }
                                        else if( m == 3 ){
                                            sentenceBuilder9.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
    
                                            s9check = true;
                                        }
                                        else{
                                            sentenceBuilder10.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
    
                                            s10check = true;
                                        }
                                    }
        
                                    Random rand = new Random();
        
                                    //  for suffix1
                                    int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                    sentenceBuilder2.append("[BWORD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                    //  for suffix2
                                    int randIndex3 = rand.nextInt(listOfSuffix2ForBad.size());
                                    sentenceBuilder4.append("[BWORD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix2ForBad.get(randIndex3) +"] ");
        
                                    int randIndex4 = (randIndex3 * 3 + 5) % listOfSuffix2ForBad.size();
                                    sentenceBuilder5.append("[BWORD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix2ForBad.get(randIndex4) +"] ");
        
                                    //  for suffix3
                                    int randIndex2 = rand.nextInt(listOfSuffix3ForBad.size());
                                    sentenceBuilder3.append("[BWORD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix3ForBad.get(randIndex2) +"] ");
        
                                }
                                else{ // length < 5
    
//                                    List<String> listOfStringsEditted = wordEdittedShorter5(tokensOfSentenceOrigin.get(i));
//
//                                    for(int m = 0; m < listOfStringsEditted.size(); ++m){
//                                        if( m == 0 ){ // str1
//                                            sentenceBuilder6.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s6check = true;
//                                        }
//                                        else if( m == 1 ){
//                                            sentenceBuilder7.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s7check = true;
//                                        }
//                                        else if( m == 2 ){
//                                            sentenceBuilder8.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s8check = true;
//                                        }
//                                        else if( m == 3 ){
//                                            sentenceBuilder9.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s9check = true;
//                                        }
//                                        else{
//                                            sentenceBuilder10.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                            s10check = true;
//                                        }
//                                    }
        
                                    Random rand = new Random();
        
                                    //  for suffix1
                                    int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                    sentenceBuilder2.append("[BWORD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                    int randIndex2 = (randIndex + 1) % listOfSuffix1ForBad.size();
                                    sentenceBuilder3.append("[BWORD "+tokensOfSentenceOrigin.get(i)+ listOfSuffix1ForBad.get(randIndex2) +"] ");
        
                                    // for origin
                                    sentenceBuilder4.append("[BWORD "+tokensOfSentenceOrigin.get(i)+"] ");
                                    sentenceBuilder5.append("[BWORD "+tokensOfSentenceOrigin.get(i)+"] ");
        
                                }
                                
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
                                    badExistInSentence = true;
                                    StringBuilder strBuild3 = new StringBuilder();
                                    strBuild3.append(tokensOfSentenceOrigin.get(i)+" "+tokensOfSentenceOrigin.get(i+1));
                                    String strToken2 = strBuild3.toString();
                                    sentenceBuilder.append("[BWORD "+ strToken2 +"] ");
    
                                    int lengthOfStr = strBad.length();
                                    if( lengthOfStr >= 5 && lengthOfStr < 10 ){
    
//                                        List<String> listOfStringsEditted = wordEditted(strToken);
//
//                                        for(int m = 0; m < listOfStringsEditted.size(); ++m){
//                                            if( m == 0 ){ // str1
//                                                sentenceBuilder6.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s6check = true;
//                                            }
//                                            else if( m == 1 ){
//                                                sentenceBuilder7.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s7check = true;
//                                            }
//                                            else if( m == 2 ){
//                                                sentenceBuilder8.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s8check = true;
//                                            }
//                                            else if( m == 3 ){
//                                                sentenceBuilder9.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s9check = true;
//                                            }
//                                            else{
//                                                sentenceBuilder10.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s10check = true;
//                                            }
//                                        }
        
                                        Random rand = new Random();
        
                                        //  for suffix1
                                        int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                        sentenceBuilder2.append("[BWORD "+ strToken2 + listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                        int randIndex2 = (randIndex + 1) % listOfSuffix1ForBad.size();
                                        sentenceBuilder3.append("[BWORD "+ strToken2 + listOfSuffix1ForBad.get(randIndex2) +"] ");
        
                                        //  for suffix2
                                        int randIndex3 = rand.nextInt(listOfSuffix2ForBad.size());
                                        sentenceBuilder4.append("[BWORD "+ strToken2 + listOfSuffix2ForBad.get(randIndex3) +"] ");
        
                                        int randIndex4 = (randIndex3 * 3 + 5) % listOfSuffix2ForBad.size();
                                        sentenceBuilder5.append("[BWORD "+ strToken2 + listOfSuffix2ForBad.get(randIndex4) +"] ");
        
                                    }
                                    else if( lengthOfStr >= 10 ){
    
                                        List<String> listOfStringsEditted = wordEditted(strToken);
    
                                        for(int m = 0; m < listOfStringsEditted.size(); ++m){
                                            if( m == 0 ){ // str1
                                                sentenceBuilder6.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
            
                                                s6check = true;
                                            }
                                            else if( m == 1 ){
                                                sentenceBuilder7.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
            
                                                s7check = true;
                                            }
                                            else if( m == 2 ){
                                                sentenceBuilder8.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
            
                                                s8check = true;
                                            }
                                            else if( m == 3 ){
                                                sentenceBuilder9.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
    
                                                s9check = true;
                                            }
                                            else{
                                                sentenceBuilder10.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
    
                                                s10check = true;
                                            }
                                        }
        
                                        Random rand = new Random();
        
                                        //  for suffix1
                                        int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                        sentenceBuilder2.append("[BWORD "+ strToken2 + listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                        //  for suffix2
                                        int randIndex3 = rand.nextInt(listOfSuffix2ForBad.size());
                                        sentenceBuilder4.append("[BWORD "+ strToken2 + listOfSuffix2ForBad.get(randIndex3) +"] ");
        
                                        int randIndex4 = (randIndex3 * 3 + 5) % listOfSuffix2ForBad.size();
                                        sentenceBuilder5.append("[BWORD "+ strToken2 + listOfSuffix2ForBad.get(randIndex4) +"] ");
        
                                        //  for suffix3
                                        int randIndex2 = rand.nextInt(listOfSuffix3ForBad.size());
                                        sentenceBuilder3.append("[BWORD "+ strToken2 + listOfSuffix3ForBad.get(randIndex2) +"] ");
        
                                    }
                                    else{ // length < 5
    
//                                        List<String> listOfStringsEditted = wordEdittedShorter5(strToken);
//
//                                        for(int m = 0; m < listOfStringsEditted.size(); ++m){
//                                            if( m == 0 ){ // str1
//                                                sentenceBuilder6.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s6check = true;
//                                            }
//                                            else if( m == 1 ){
//                                                sentenceBuilder7.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s7check = true;
//                                            }
//                                            else if( m == 2 ){
//                                                sentenceBuilder8.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s8check = true;
//                                            }
//                                            else if( m == 3 ){
//                                                sentenceBuilder9.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s9check = true;
//                                            }
//                                            else{
//                                                sentenceBuilder10.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s10check = true;
//                                            }
//                                        }
        
                                        Random rand = new Random();
        
                                        //  for suffix1
                                        int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                        sentenceBuilder2.append("[BWORD "+ strToken2 + listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                        int randIndex2 = (randIndex + 1) % listOfSuffix1ForBad.size();
                                        sentenceBuilder3.append("[BWORD "+ strToken2 + listOfSuffix1ForBad.get(randIndex2) +"] ");
        
                                        // for origin
                                        sentenceBuilder4.append("[BWORD "+ strToken2 +"] ");
                                        sentenceBuilder5.append("[BWORD "+ strToken2 +"] ");
        
                                    }
                                    
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
                                    badExistInSentence = true;
                                    StringBuilder strBuild3 = new StringBuilder();
                                    strBuild3.append(tokensOfSentenceOrigin.get(i)+" "+tokensOfSentenceOrigin.get(i+1)+" "+tokensOfSentenceOrigin.get(i+2));
                                    String strToken2 = strBuild3.toString();
                                    sentenceBuilder.append("[BWORD "+ strToken2 +"] ");
    
                                    int lengthOfStr = strBad.length();
                                    if( lengthOfStr >= 5 && lengthOfStr < 10 ){
    
//                                        List<String> listOfStringsEditted = wordEditted(strToken);
//
//                                        for(int m = 0; m < listOfStringsEditted.size(); ++m){
//                                            if( m == 0 ){ // str1
//                                                sentenceBuilder6.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s6check = true;
//                                            }
//                                            else if( m == 1 ){
//                                                sentenceBuilder7.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s7check = true;
//                                            }
//                                            else if( m == 2 ){
//                                                sentenceBuilder8.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s8check = true;
//                                            }
//                                            else if( m == 3 ){
//                                                sentenceBuilder9.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s9check = true;
//                                            }
//                                            else{
//                                                sentenceBuilder10.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s10check = true;
//                                            }
//                                        }
        
                                        Random rand = new Random();
        
                                        //  for suffix1
                                        int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                        sentenceBuilder2.append("[BWORD "+ strToken2 + listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                        int randIndex2 = (randIndex + 1) % listOfSuffix1ForBad.size();
                                        sentenceBuilder3.append("[BWORD "+ strToken2 + listOfSuffix1ForBad.get(randIndex2) +"] ");
        
                                        //  for suffix2
                                        int randIndex3 = rand.nextInt(listOfSuffix2ForBad.size());
                                        sentenceBuilder4.append("[BWORD "+ strToken2 + listOfSuffix2ForBad.get(randIndex3) +"] ");
        
                                        int randIndex4 = (randIndex3 * 3 + 5) % listOfSuffix2ForBad.size();
                                        sentenceBuilder5.append("[BWORD "+ strToken2 + listOfSuffix2ForBad.get(randIndex4) +"] ");
        
                                    }
                                    else if( lengthOfStr >= 10 ){
    
                                        List<String> listOfStringsEditted = wordEditted(strToken);
    
                                        for(int m = 0; m < listOfStringsEditted.size(); ++m){
                                            if( m == 0 ){ // str1
                                                sentenceBuilder6.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
            
                                                s6check = true;
                                            }
                                            else if( m == 1 ){
                                                sentenceBuilder7.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
            
                                                s7check = true;
                                            }
                                            else if( m == 2 ){
                                                sentenceBuilder8.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
            
                                                s8check = true;
                                            }
                                            else if( m == 3 ){
                                                sentenceBuilder9.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
    
                                                s9check = true;
                                            }
                                            else{
                                                sentenceBuilder10.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
    
                                                s10check = true;
                                            }
                                        }
        
                                        Random rand = new Random();
        
                                        //  for suffix1
                                        int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                        sentenceBuilder2.append("[BWORD "+ strToken2 + listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                        //  for suffix2
                                        int randIndex3 = rand.nextInt(listOfSuffix2ForBad.size());
                                        sentenceBuilder4.append("[BWORD "+ strToken2 + listOfSuffix2ForBad.get(randIndex3) +"] ");
        
                                        int randIndex4 = (randIndex3 * 3 + 5) % listOfSuffix2ForBad.size();
                                        sentenceBuilder5.append("[BWORD "+ strToken2 + listOfSuffix2ForBad.get(randIndex4) +"] ");
        
                                        //  for suffix3
                                        int randIndex2 = rand.nextInt(listOfSuffix3ForBad.size());
                                        sentenceBuilder3.append("[BWORD "+ strToken2 + listOfSuffix3ForBad.get(randIndex2) +"] ");
        
                                    }
                                    else{ // length < 5
    
//                                        List<String> listOfStringsEditted = wordEdittedShorter5(strToken);
//
//                                        for(int m = 0; m < listOfStringsEditted.size(); ++m){
//                                            if( m == 0 ){ // str1
//                                                sentenceBuilder6.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s6check = true;
//                                            }
//                                            else if( m == 1 ){
//                                                sentenceBuilder7.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s7check = true;
//                                            }
//                                            else if( m == 2 ){
//                                                sentenceBuilder8.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s8check = true;
//                                            }
//                                            else if( m == 3 ){
//                                                sentenceBuilder9.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s9check = true;
//                                            }
//                                            else{
//                                                sentenceBuilder10.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s10check = true;
//                                            }
//                                        }
        
                                        Random rand = new Random();
        
                                        //  for suffix1
                                        int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                        sentenceBuilder2.append("[BWORD "+ strToken2 + listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                        int randIndex2 = (randIndex + 1) % listOfSuffix1ForBad.size();
                                        sentenceBuilder3.append("[BWORD "+ strToken2 + listOfSuffix1ForBad.get(randIndex2) +"] ");
        
                                        // for origin
                                        sentenceBuilder4.append("[BWORD "+ strToken2 +"] ");
                                        sentenceBuilder5.append("[BWORD "+ strToken2 +"] ");
        
                                    }
                                    
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
                                    badExistInSentence = true;
                                    StringBuilder strBuild3 = new StringBuilder();
                                    strBuild3.append(tokensOfSentenceOrigin.get(i)+" "+tokensOfSentenceOrigin.get(i+1)+" "+tokensOfSentenceOrigin.get(i+2)+" "+tokensOfSentenceOrigin.get(i+3));
                                    String strToken2 = strBuild3.toString();
                                    sentenceBuilder.append("[BWORD "+ strToken2 +"] ");
    
                                    int lengthOfStr = strBad.length();
                                    if( lengthOfStr >= 5 && lengthOfStr < 10 ){
    
//                                        List<String> listOfStringsEditted = wordEditted(strToken);
//
//                                        for(int m = 0; m < listOfStringsEditted.size(); ++m){
//                                            if( m == 0 ){ // str1
//                                                sentenceBuilder6.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s6check = true;
//                                            }
//                                            else if( m == 1 ){
//                                                sentenceBuilder7.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s7check = true;
//                                            }
//                                            else if( m == 2 ){
//                                                sentenceBuilder8.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s8check = true;
//                                            }
//                                            else if( m == 3 ){
//                                                sentenceBuilder9.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s9check = true;
//                                            }
//                                            else{
//                                                sentenceBuilder10.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s10check = true;
//                                            }
//                                        }
        
                                        Random rand = new Random();
        
                                        //  for suffix1
                                        int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                        sentenceBuilder2.append("[BWORD "+ strToken2 + listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                        int randIndex2 = (randIndex + 1) % listOfSuffix1ForBad.size();
                                        sentenceBuilder3.append("[BWORD "+ strToken2 + listOfSuffix1ForBad.get(randIndex2) +"] ");
        
                                        //  for suffix2
                                        int randIndex3 = rand.nextInt(listOfSuffix2ForBad.size());
                                        sentenceBuilder4.append("[BWORD "+ strToken2 + listOfSuffix2ForBad.get(randIndex3) +"] ");
        
                                        int randIndex4 = (randIndex3 * 3 + 5) % listOfSuffix2ForBad.size();
                                        sentenceBuilder5.append("[BWORD "+ strToken2 + listOfSuffix2ForBad.get(randIndex4) +"] ");
        
                                    }
                                    else if( lengthOfStr >= 10 ){
    
                                        List<String> listOfStringsEditted = wordEditted(strToken);
    
                                        for(int m = 0; m < listOfStringsEditted.size(); ++m){
                                            if( m == 0 ){ // str1
                                                sentenceBuilder6.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
            
                                                s6check = true;
                                            }
                                            else if( m == 1 ){
                                                sentenceBuilder7.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
            
                                                s7check = true;
                                            }
                                            else if( m == 2 ){
                                                sentenceBuilder8.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
            
                                                s8check = true;
                                            }
                                            else if( m == 3 ){
                                                sentenceBuilder9.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
    
                                                s9check = true;
                                            }
                                            else{
                                                sentenceBuilder10.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
    
                                                s10check = true;
                                            }
                                        }
        
                                        Random rand = new Random();
        
                                        //  for suffix1
                                        int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                        sentenceBuilder2.append("[BWORD "+ strToken2 + listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                        //  for suffix2
                                        int randIndex3 = rand.nextInt(listOfSuffix2ForBad.size());
                                        sentenceBuilder4.append("[BWORD "+ strToken2 + listOfSuffix2ForBad.get(randIndex3) +"] ");
        
                                        int randIndex4 = (randIndex3 * 3 + 5) % listOfSuffix2ForBad.size();
                                        sentenceBuilder5.append("[BWORD "+ strToken2 + listOfSuffix2ForBad.get(randIndex4) +"] ");
        
                                        //  for suffix3
                                        int randIndex2 = rand.nextInt(listOfSuffix3ForBad.size());
                                        sentenceBuilder3.append("[BWORD "+ strToken2 + listOfSuffix3ForBad.get(randIndex2) +"] ");
        
                                    }
                                    else{ // length < 5
    
//                                        List<String> listOfStringsEditted = wordEdittedShorter5(strToken);
//
//                                        for(int m = 0; m < listOfStringsEditted.size(); ++m){
//                                            if( m == 0 ){ // str1
//                                                sentenceBuilder6.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s6check = true;
//                                            }
//                                            else if( m == 1 ){
//                                                sentenceBuilder7.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s7check = true;
//                                            }
//                                            else if( m == 2 ){
//                                                sentenceBuilder8.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s8check = true;
//                                            }
//                                            else if( m == 3 ){
//                                                sentenceBuilder9.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s9check = true;
//                                            }
//                                            else{
//                                                sentenceBuilder10.append("[BWORD " + listOfStringsEditted.get(m) + "] ");
//
//                                                s10check = true;
//                                            }
//                                        }
        
                                        Random rand = new Random();
        
                                        //  for suffix1
                                        int randIndex = rand.nextInt(listOfSuffix1ForBad.size());
                                        sentenceBuilder2.append("[BWORD "+ strToken2 + listOfSuffix1ForBad.get(randIndex) +"] ");
        
                                        int randIndex2 = (randIndex + 1) % listOfSuffix1ForBad.size();
                                        sentenceBuilder3.append("[BWORD "+ strToken2 + listOfSuffix1ForBad.get(randIndex2) +"] ");
        
                                        // for origin
                                        sentenceBuilder4.append("[BWORD "+ strToken2 +"] ");
                                        sentenceBuilder5.append("[BWORD "+ strToken2 +"] ");
        
                                    }
                                    
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
                    sentenceBuilder2.append(tokensOfSentenceOrigin.get(i)+" ");
                    sentenceBuilder3.append(tokensOfSentenceOrigin.get(i)+" ");
                    sentenceBuilder4.append(tokensOfSentenceOrigin.get(i)+" ");
                    sentenceBuilder5.append(tokensOfSentenceOrigin.get(i)+" ");
                    
                    sentenceBuilder6.append(tokensOfSentenceOrigin.get(i)+" ");
                    sentenceBuilder7.append(tokensOfSentenceOrigin.get(i)+" ");
                    sentenceBuilder8.append(tokensOfSentenceOrigin.get(i)+" ");
                    sentenceBuilder9.append(tokensOfSentenceOrigin.get(i)+" ");
                    sentenceBuilder10.append(tokensOfSentenceOrigin.get(i)+" ");
                    
                    ++i;
                }
        
        
            }
    
            listOfTaggedStrings.add(sentenceBuilder.toString().trim());
            if( badExistInSentence ){
                listOfTaggedStrings.add(sentenceBuilder2.toString().trim());
                listOfTaggedStrings.add(sentenceBuilder3.toString().trim());
                listOfTaggedStrings.add(sentenceBuilder4.toString().trim());
                listOfTaggedStrings.add(sentenceBuilder5.toString().trim());
    
                if( s6check )
                    listOfTaggedStrings.add(sentenceBuilder6.toString().trim());
                if( s7check )
                    listOfTaggedStrings.add(sentenceBuilder7.toString().trim());
                if( s8check )
                    listOfTaggedStrings.add(sentenceBuilder8.toString().trim());
                if( s9check )
                    listOfTaggedStrings.add(sentenceBuilder9.toString().trim());
                if( s10check )
                    listOfTaggedStrings.add(sentenceBuilder10.toString().trim());
                
            }
            
            
            return listOfTaggedStrings;
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
    
    public static List<NamedEntity> findNamedEntities(PerceptronNer ner, String sentence) throws IOException {
        
        NerSentence result = ner.findNamedEntities(sentence);
        
        List<NamedEntity> namedEntities = result.getNamedEntities();
        
//        for (NamedEntity namedEntity : namedEntities) {
//            System.out.println(namedEntity);
//        }
        
        return namedEntities;
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
            long i = 1, truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0;
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