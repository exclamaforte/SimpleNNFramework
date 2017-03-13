/**
 * @Author: Yuting Liu and Jude Shavlik.
 *
 * Copyright 2017.  Free for educational and basic-research use.
 *
 * The main class for Lab3 of cs638/838.
 *
 * Reads in the image files and stores BufferedImage's for every example.  Converts to fixed-length
 * feature vectors (of doubles).  Can use RGB (plus grey-scale) or use grey scale.
 *
 * You might want to debug and experiment with your Deep ANN code using a separate class, but when you turn in Lab3.java, insert that class here to simplify grading.
 *
 * Some snippets from Jude's code left in here - feel free to use or discard.
 *
 */

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import javax.imageio.ImageIO;

public class Lab3 {

    public static final boolean debug_flag = true;

    private static int     imageSize = 32;
    // Images are imageSize x imageSize.  The provided data is 128x128, but this can be resized by setting this value (or passing in an argument).
    // You might want to resize to 8x8, 16x16, 32x32, or 64x64; this can reduce your network size and speed up debugging runs.
    // ALL IMAGES IN A TRAINING RUN SHOULD BE THE *SAME* SIZE.
    public static enum    Category {
        airplanes(0) , butterfly(1), flower(2), grand_piano(3), starfish(4), watch(5);

        private final int value;
        private Category(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }};  // We'll hardwire these in, but more robust code would not do so.

    private static final Boolean    useRGB = false;
    // If true, FOUR units are used per pixel: red, green, blue, and grey.  If false, only ONE (the grey-scale value).
    private static       int unitsPerPixel = (useRGB ? 4 : 1);
    // If using RGB, use red+blue+green+grey.  Otherwise just use the grey value.

    private static String    modelToUse = "deep";
    // Should be one of { "perceptrons", "oneLayer", "deep" };  You might want to use this if you are trying approaches other than a Deep ANN.
    private static int       inputVectorSize;
    // The provided code uses a 1D vector of input features.  You might want to create a 2D version for your Deep ANN code.
    // Or use the get2DfeatureValue() 'accessor function' that maps 2D coordinates into the 1D vector.
    // The last element in this vector holds the 'teacher-provided' label of the example.

    private static double eta       =    0.1, fractionOfTrainingToUse = 1.00, dropoutRate = 0.50; // To turn off drop out, set dropoutRate to 0.0 (or a neg number).
    private static int    maxEpochs = 150; // Feel free to set to a different value.
    private NeuralNetwork nn;
    
    public static final int Num_Classes  = 6;
    
    public static void main(String[] args) {
        String trainDirectory = "images/trainset/";
        String  tuneDirectory = "images/tuneset/";
        String  testDirectory = "images/testset/";

        if(args.length > 5) {
            System.err.println("Usage error: java Lab3 <train_set_folder_path> <tune_set_folder_path> <test_set_foler_path> <imageSize>");
            System.exit(1);
        }
        if (args.length > 0) { trainDirectory = args[0]; }
        if (args.length > 1) {  tuneDirectory = args[1]; }
        if (args.length > 2) {  testDirectory = args[2]; }
        if (args.length > 3) {  imageSize     = Integer.parseInt(args[3]); }

        // Here are statements with the absolute path to open images folder
        File trainsetDir = new File(trainDirectory);
        File tunesetDir  = new File(tuneDirectory);
        File testsetDir  = new File(testDirectory);

        // create three datasets
        Dataset trainset = new Dataset();
        Dataset tuneset  = new Dataset();
        Dataset testset  = new Dataset();

        loadDataset(trainset, trainsetDir);
        loadDataset(tuneset, tunesetDir);
        loadDataset(testset, testsetDir);
        trainANN(trainset, tuneset, testset);

    }

    public static void loadDataset(Dataset dataset, File dir) {
        for(File file : dir.listFiles()) {
            // check all files
            if(!file.isFile() || file.getName().endsWith(".DS_Store")) {
                continue;
            }
            //String path = file.getAbsolutePath();
            BufferedImage img = null, scaledBI = null;
            try {
                // load in all images
                img = ImageIO.read(file);
                // every image's name is in such format:
                // label_image_XXXX(4 digits) though this code could handle more than 4 digits.
                String name = file.getName();
                int locationOfUnderscoreImage = name.indexOf("_image");

                // Resize the image if requested.  Any resizing allowed, but should really be one of 8x8, 16x16, 32x32, or 64x64 (original data is 128x128).
                if (imageSize != 128) {
                    scaledBI = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_INT_RGB);
                    Graphics2D g = scaledBI.createGraphics();
                    g.drawImage(img, 0, 0, imageSize, imageSize, null);
                    g.dispose();
                }

                Instance instance = new Instance(scaledBI == null ? img : scaledBI, name.substring(0, locationOfUnderscoreImage));

                dataset.add(instance);
            } catch (IOException e) {
                System.err.println("Error: cannot load in the image file");
                System.exit(1);
            }
        }
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////

    private static Category convertCategoryStringToEnum(String name) {
        if ("airplanes".equals(name))   return Category.airplanes; // Should have been the singular 'airplane' but we'll live with this minor error.
        if ("butterfly".equals(name))   return Category.butterfly;
        if ("flower".equals(name))      return Category.flower;
        if ("grand_piano".equals(name)) return Category.grand_piano;
        if ("starfish".equals(name))    return Category.starfish;
        if ("watch".equals(name))       return Category.watch;
        throw new Error("Unknown category: " + name);
    }

    // Return the count of TESTSET errors for the chosen model.
    private static int trainANN(Dataset trainset, Dataset tuneset, Dataset testset) {
        Instance sampleImage = trainset.getImages().get(0); // Assume there is at least one train image!
        inputVectorSize = sampleImage.getWidth() * sampleImage.getHeight() * unitsPerPixel + 1; // The '-1' for the bias is not explicitly added to all examples (instead code should implicitly handle it).  The final 1 is for the CATEGORY.

        // For RGB, we use FOUR input units per pixel: red, green, blue, plus grey.  Otherwise we only use GREY scale.
        // Pixel values are integers in [0,255], which we convert to a double in [0.0, 1.0].
        // The last item in a feature vector is the CATEGORY, encoded as a double in 0 to the size on the Category enum.
        // We do not explicitly store the '-1' that is used for the bias.  Instead code (to be written) will need to implicitly handle that extra feature.

        double[][][][] trainImages = new double[trainset.getSize()][3][trainset.getImageWidth()][trainset.getImageHeight()];
        double[][][][] tuneImages  = new double[tuneset.getSize()][3][tuneset.getImageWidth()][tuneset.getImageHeight()];
        double[][][][] testImages  = new double[testset.getSize()][3][testset.getImageWidth()][testset.getImageHeight()];

        double[][] trainLabels = new double[trainset.getSize()][Num_Classes];
        double[][] tuneLabels = new double[tuneset.getSize()][Num_Classes];
        double[][] testLabels = new double[testset.getSize()][Num_Classes];


        fillInstanceArrays(trainImages, trainLabels,trainset);
        fillInstanceArrays(tuneImages, tuneLabels,  tuneset);
        fillInstanceArrays(testImages, testLabels,  testset);

        
        double[][] testPredictions;
        NeuralNetwork nn;
        
        // =============
        nn = new NeuralNetwork(trainset.getImageWidth(), 3, Category.values().length,false);

        nn.addConvolutionLayer(16, 4, 1);
        nn.addMaxPoolingLayer(2, 3);
        nn.addConvolutionLayer(16, 4, 2);
        nn.addMaxPoolingLayer(1, 2);
        nn.addConvolutionLayer(16, 4, 1);
        nn.addMaxPoolingLayer(1, 2);
        nn.addConvolutionLayer(16, 1, 1);
        nn.addConvolutionLayer(6, 1, 1);
        nn.addOutputLayer();

        runEarlyStopping(nn,trainImages,trainLabels, tuneImages, tuneLabels);

        testPredictions = new double[testLabels.length][Num_Classes];
        nn.predict(testPredictions, testImages);

        System.out.println("Test set 0-1 loss: " + calc01Loss(testPredictions,testLabels));

        nn.printConfusionMatrix(testLabels,testPredictions);
     
        // =================
        nn = new NeuralNetwork(trainset.getImageWidth(), 3, Category.values().length,false);

        nn.addConvolutionLayer(16, 4, 1);
        nn.addMaxPoolingLayer(2, 3);
        nn.addConvolutionLayer(8, 4, 2);
        nn.addMaxPoolingLayer(1, 2);
        nn.addConvolutionLayer(8, 4, 1);
        nn.addMaxPoolingLayer(1, 2);
        nn.addConvolutionLayer(16, 1, 1);
        nn.addConvolutionLayer(6, 1, 1);
        nn.addOutputLayer();

        runEarlyStopping(nn,trainImages,trainLabels, tuneImages, tuneLabels);

        testPredictions = new double[testLabels.length][Num_Classes];
        nn.predict(testPredictions, testImages);

        System.out.println("Test set 0-1 loss: " + calc01Loss(testPredictions,testLabels));

        nn.printConfusionMatrix(testLabels,testPredictions);

        nn = new NeuralNetwork(trainset.getImageWidth(), 3, Category.values().length,false);

        nn.addConvolutionLayer(16, 4, 1);
        nn.addMaxPoolingLayer(2, 3);
        nn.addConvolutionLayer(4, 4, 2);
        nn.addMaxPoolingLayer(1, 2);
        nn.addConvolutionLayer(4, 4, 1);
        nn.addMaxPoolingLayer(1, 2);
        nn.addConvolutionLayer(16, 1, 1);
        nn.addConvolutionLayer(6, 1, 1);
        nn.addOutputLayer();

        runEarlyStopping(nn,trainImages,trainLabels, tuneImages, tuneLabels);

        testPredictions = new double[testLabels.length][Num_Classes];
        nn.predict(testPredictions, testImages);

        System.out.println("Test set 0-1 loss: " + calc01Loss(testPredictions,testLabels));

        nn.printConfusionMatrix(testLabels,testPredictions);

        nn = new NeuralNetwork(trainset.getImageWidth(), 3, Category.values().length,false);

        nn.addConvolutionLayer(16, 4, 1);
        nn.addMaxPoolingLayer(2, 3);
        nn.addConvolutionLayer(4, 4, 2);
        nn.addMaxPoolingLayer(1, 2);
        nn.addConvolutionLayer(4, 4, 1);
        nn.addMaxPoolingLayer(1, 2);
        nn.addConvolutionLayer(8, 1, 1);
        nn.addConvolutionLayer(6, 1, 1);
        nn.addOutputLayer();

        runEarlyStopping(nn,trainImages,trainLabels, tuneImages, tuneLabels);

        testPredictions = new double[testLabels.length][Num_Classes];
        nn.predict(testPredictions, testImages);

        System.out.println("Test set 0-1 loss: " + calc01Loss(testPredictions,testLabels));

        nn.printConfusionMatrix(testLabels,testPredictions);
        
        nn = new NeuralNetwork(trainset.getImageWidth(), 3, Category.values().length,false);

        nn.addConvolutionLayer(16, 4, 1);
        nn.addMaxPoolingLayer(2, 3);
        nn.addConvolutionLayer(16, 4, 2);
        nn.addMaxPoolingLayer(1, 2);
        nn.addConvolutionLayer(16, 4, 1);
        nn.addMaxPoolingLayer(1, 2);
        nn.addConvolutionLayer(16, 1, 1);
        nn.addConvolutionLayer(6, 1, 1);
        nn.addOutputLayer();

        runEarlyStopping(nn,trainImages,trainLabels, tuneImages, tuneLabels);

        testPredictions = new double[testLabels.length][Num_Classes];
        nn.predict(testPredictions, testImages);

        System.out.println("Test set 0-1 loss: " + calc01Loss(testPredictions,testLabels));

        nn.printConfusionMatrix(testLabels,testPredictions);
     
        
        nn = new NeuralNetwork(trainset.getImageWidth(), 3, Category.values().length,true);

        nn.addConvolutionLayer(16, 4, 1);
        nn.addMaxPoolingLayer(2, 3);
        nn.addConvolutionLayer(8, 4, 2);
        nn.addMaxPoolingLayer(1, 2);
        nn.addConvolutionLayer(8, 4, 1);
        nn.addMaxPoolingLayer(1, 2);
        nn.addConvolutionLayer(16, 1, 1);
        nn.addConvolutionLayer(6, 1, 1);
        nn.addOutputLayer();

        runEarlyStopping(nn,trainImages,trainLabels, tuneImages, tuneLabels);

        testPredictions = new double[testLabels.length][Num_Classes];
        nn.predict(testPredictions, testImages);

        System.out.println("Test set 0-1 loss: " + calc01Loss(testPredictions,testLabels));

        nn.printConfusionMatrix(testLabels,testPredictions);

        nn = new NeuralNetwork(trainset.getImageWidth(), 3, Category.values().length,true);

        nn.addConvolutionLayer(16, 4, 1);
        nn.addMaxPoolingLayer(2, 3);
        nn.addConvolutionLayer(4, 4, 2);
        nn.addMaxPoolingLayer(1, 2);
        nn.addConvolutionLayer(4, 4, 1);
        nn.addMaxPoolingLayer(1, 2);
        nn.addConvolutionLayer(16, 1, 1);
        nn.addConvolutionLayer(6, 1, 1);
        nn.addOutputLayer();

        runEarlyStopping(nn,trainImages,trainLabels, tuneImages, tuneLabels);

        testPredictions = new double[testLabels.length][Num_Classes];
        nn.predict(testPredictions, testImages);

        System.out.println("Test set 0-1 loss: " + calc01Loss(testPredictions,testLabels));

        nn.printConfusionMatrix(testLabels,testPredictions);

        nn = new NeuralNetwork(trainset.getImageWidth(), 3, Category.values().length,true);

        nn.addConvolutionLayer(16, 4, 1);
        nn.addMaxPoolingLayer(2, 3);
        nn.addConvolutionLayer(4, 4, 2);
        nn.addMaxPoolingLayer(1, 2);
        nn.addConvolutionLayer(4, 4, 1);
        nn.addMaxPoolingLayer(1, 2);
        nn.addConvolutionLayer(8, 1, 1);
        nn.addConvolutionLayer(6, 1, 1);
        nn.addOutputLayer();

        runEarlyStopping(nn,trainImages,trainLabels, tuneImages, tuneLabels);

        testPredictions = new double[testLabels.length][Num_Classes];
        nn.predict(testPredictions, testImages);

        System.out.println("Test set 0-1 loss: " + calc01Loss(testPredictions,testLabels));

        nn.printConfusionMatrix(testLabels,testPredictions);
        
        return -1;
    }
    public static final int starting_patience = 10;
    public static final float patience_mult = 2;
    public static final int validation_wait = 5;
    public static final double improvement_threshold = 0.995;

    public static void runEarlyStopping(NeuralNetwork net, double[][][][] trainImages, double[][] trainClassLabels, double[][][][] tuneImages, double[][] tuneClassLabels){
        double bestLoss = Double.POSITIVE_INFINITY;
        double[][] predictStorage = new double[tuneClassLabels.length][Num_Classes];
        double[][] trainingStorage = new double[trainClassLabels.length][Num_Classes];
        int patience = starting_patience;
        int epoch = 0;
        while(epoch < maxEpochs){
            if(debug_flag)
                System.out.println("Epoch " + (epoch +1));
            shuffle(trainImages,trainClassLabels);
            net.train(trainImages,trainClassLabels);
            if((epoch + 1)% validation_wait == 0){

                double loss = calc01Loss(net,tuneImages,tuneClassLabels,predictStorage);
                if(debug_flag) {
                    System.out.println("Current 01 tune loss: " + loss);
                    System.out.println("Current 01 train loss" + calc01Loss(net,trainImages,trainClassLabels,trainingStorage));
                }
                if(loss < (bestLoss)){
                    if(loss < (bestLoss * improvement_threshold))
                        patience = (int) Math.max(patience, epoch * patience_mult);
                    bestLoss = loss;
                    net.cacheCurrentBestTuneWeights();
                }
                //I've run out of patience
                else if(epoch > patience){
                    break;
                }
            }
            epoch ++;
        }
        //Set best weights to best based off tuning set
        net.resetToBestTuningWeights();
    }

    private static double calc01Loss(NeuralNetwork nnet, double[][][][] images, double[][] classLabels, double[][] predictStorage){
        nnet.predict(predictStorage,images);
        return calc01Loss(classLabels,predictStorage);
    }

    private static double calc01Loss(double[][] predicted, double[][] actual){
        double correct = 0;
        for(int i = 0; i < actual.length; i ++){
            for(int j = 0; j < Num_Classes; j ++){
                correct += actual[i][j] * predicted[i][j];
            }
        }
        return 1- correct / actual.length;
    }

    private static double[][] convertDouble(int[][] in) {
    	double[][] ret = new double[in.length][in[0].length];
    	for (int i = 0; i < in.length; i++) {
    		for (int j = 0; j < in[i].length; j++) {
    			ret[i][j] = ((double)in[i][j]) / 255.0;
    		}
    	}
    	return ret;
    }

    private static void fillInstanceArrays(double[][][][] images, double[][] labelsOnehot, Dataset d) {
        ArrayList<Instance> trainInstances = d.getImages();
        for(int i = 0; i < trainInstances.size(); i ++){
            Instance instance = trainInstances.get(i);
        	images[i][0] = convertDouble(instance.getRedChannel());
        	images[i][1] = convertDouble(instance.getGreenChannel());
        	images[i][2] = convertDouble(instance.getBlueChannel());
        }



        for(int i = 0; i < trainInstances.size(); i ++){
            Category category = convertCategoryStringToEnum(trainInstances.get(i).getLabel());

            labelsOnehot[i][category.getValue()] = 1;
        }
    }

    public static void shuffle(double[][][][] trainImages, double[][] trainClassLabels){
        assert(trainClassLabels.length == trainImages.length);
        for(int i = 0; i < trainClassLabels.length-1; i ++){
            int swapIndex = (int)(i + Math.random()*(trainClassLabels.length-i));
            double[] swapClass = trainClassLabels[swapIndex];
            double[][][] swapImage = trainImages[swapIndex];
            trainImages[swapIndex] = trainImages[i];
            trainClassLabels[swapIndex] = trainClassLabels[i];
            trainImages[i] = swapImage;
            trainClassLabels[i] = swapClass;
        }
    }
}
