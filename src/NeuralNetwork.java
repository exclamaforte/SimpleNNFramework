import java.util.ArrayList;
import java.util.Locale;

public class NeuralNetwork {
    public static final double LEARNING_RATE  = 0.001;
    private int inputDepth;
    private int inputWidth;
    private OutputLayer output;
    private ArrayList<Layer> layers;
    private int outputClasses;
    private double learningRate = 0.1;
    public NeuralNetwork(int inputWidth, int inputDepth, int outputClasses) {
        this.layers = new ArrayList<Layer>();
        this.inputDepth = inputDepth;
        this.inputWidth = inputWidth;
        this.outputClasses = outputClasses;
    }
    /*
     * Converts a dataset object to the correct feature vector
     */
    public static double[][][] convertDataset(Dataset d) {
    	return null;
    }

    /*
     * Converts a dataset object to the correct class vector
     */
    public static double[] convertClass(Dataset d) {
    	return null;
    }

    public static double Convolution_Initial_Radius = 0.1;

    // Only set width (not height) because we're assuming it's a square
    public void addConvolutionLayer(int numKernels, int kernelWidth, int step) {
        int previousWidth;
        int previousDepth;
        if(layers.size() > 0) {
            Layer previousLayer = layers.get(layers.size() - 1);
            previousDepth = previousLayer.previousDepth;
            previousWidth = previousLayer.previousWidth;
        }
        else {
            previousWidth = inputWidth;
            previousDepth = inputDepth;
        }
        layers.add(new ConvolutionLayer(previousWidth,previousDepth,numKernels,kernelWidth,step,Convolution_Initial_Radius));
    }
    public void addMaxPoolingLayer(int step, int poolingWidth) {
        int previousWidth = -1;
        int previousDepth = -1;
        if(layers.size() == 0) {
            previousWidth = inputWidth;
            previousDepth= inputDepth;
        }
        else {
            Layer input = layers.get(layers.size() - 1);
            previousWidth = input.getOutputWidth();
            previousDepth = input.getOutputDepth();
        }
        layers.add(new MaxPoolingLayer(previousWidth, previousDepth,step,poolingWidth));
    }
    public void addFullyConnectedLayer(int numHU) {
        int previousWidth = -1;
        int previousDepth = -1;
        if(layers.size() == 0) {
            previousWidth = inputWidth;
            previousDepth= inputDepth;
        }
        else {
            Layer input = layers.get(layers.size() - 1);
            previousWidth = input.getOutputWidth();
            previousDepth = input.getOutputDepth();
        }
        layers.add(new FullyConnectedLayer(previousWidth, numHU));
    }
    public void addOutputLayer() {
        int previousDepth = -1;
        if (layers.size() == 0) {
            previousDepth = inputDepth;
        } else {
            Layer input = layers.get(layers.size() - 1);
            previousDepth = input.getOutputDepth();
        }
        this.output = new OutputLayer(previousDepth, outputClasses);
    }
    public void train(double[][][][] train, double[][] trainClass) {

        //initialize forward and backward arrays
        double[][][][] forward = new double[this.layers.size() + 2][][][];
        double[][][][] backward = new double[this.layers.size() + 2][][][];
        forward[0] = new double[inputDepth][inputWidth][inputWidth];
        backward[0] = new double[inputDepth][inputWidth][inputWidth];
        for(int i = 0; i < layers.size(); i ++){
            Layer layer = layers.get(i);
            forward[i+1] = new double[layer.outputDepth][layer.outputWidth][layer.outputWidth];
            backward[i+1] = new double[layer.outputDepth][layer.outputWidth][layer.outputWidth];
        }
        forward[forward.length -1] = new double[outputClasses][1][1];
        backward[forward.length -1] = new double[outputClasses][1][1];
        //**********************************

        //Run epochs
        for (int instidx = 0; instidx < train.length; instidx++) {
        	forward[0] = train[instidx];

            int i = 1;
            for (Layer l : this.layers) {
                l.forward(i,forward, trainClass[instidx]);
                i++;
            }
            output.forward(i, forward,  trainClass[instidx]);


            output.backwards(i,forward,backward, learningRate, trainClass[instidx]);
            i--;
            for(int layer_idx = layers.size() -1; layer_idx >= 0; layer_idx --){
                Layer l = layers.get(layer_idx);
                l.backwards(i,forward,backward,LEARNING_RATE);
                i--;
            }
        }
    }

    public void test(double[][][] test, double[][][] testClass) {

    }

    // Takes two arrays, then calculates and prints a confusion matrix
    // First array of 2D vector is an array of instance classifications. 2nd array is the 1-hot classification
    // @param dataset is a Dataset object in order to retrieve proper class labels
    public void printConfusionMatrix(double[][] actualOutputClasses, double[][] expectedOutputClasses) {

        assert(expectedOutputClasses.length == actualOutputClasses.length);
        assert(expectedOutputClasses[0].length == actualOutputClasses[0].length);

        int numClasses = expectedOutputClasses[0].length;
        int numInstances = expectedOutputClasses.length;

        // First array index is PREDICTED, second index is CORRECT (according to confusion matrix picture from lab3.ppt)
        int[][] confusionMatrix = new int[numClasses][numClasses];

        // Iterate through instances (index 'i' points to each instance)
        for(int i = 0; i < numInstances; i++) {
            double[] expectedClass = expectedOutputClasses[i];
            double[] actualClass = actualOutputClasses[i];

            // Iterate through predicted class
            for(int k = 0; k < expectedClass.length; k++) {

                // Iterate through actual class
                for(int j = 0; j < actualClass.length; j++) {

                    // Confusion determination logic
                    if(expectedClass[k] == 1 && actualClass[j] == 1) confusionMatrix[k][j]++;
                    else if(expectedClass[k] == 0 && actualClass[j] == 1) confusionMatrix[k][j]++;
                    else if(expectedClass[k] == 1 && actualClass[j] == 1) confusionMatrix[k][j]++;
                }
            }
        }

        // Print the label for each column (CORRECT CATEGORY part of confusion matrix picture)
        String[] stringOfLabels = spaghettiLabels();
        System.out.print("\t");
        for(String label : stringOfLabels) System.out.print(label + "\t");

        // TODO: Print the matrix
        for(int j = 0; j < confusionMatrix.length; j++) {
            System.out.print(stringOfLabels[j] + "\t");
            for(int k = 0; k < confusionMatrix.length; k++) {
                System.out.print(confusionMatrix[k][j]+"\t");
            }
            System.out.print("\n");
        }


        // The sole purpose of the following code is to guarantee that each row & each column adds to the correct sum
        // (which is the number of total instances) -- again, refer to confusion matrix picture from lab3.ppt
        int rowSum = 0;
        int colSum = 0;

        // Iterate through the confusion matrix row-by-row
        // j is CORRECT
        for(int j = 0; j < numClasses; j++) {
            // k is PREDICTED
            for(int k = 0; k < numClasses; k++) {
                rowSum += confusionMatrix[k][j];
            }
            assert(rowSum == numInstances);
        }

        // k is PREDICTED
        for(int k = 0; k < numClasses; k++) {
            // j is CORRECT
            for(int j = 0; j < numClasses; j++) {
                colSum += confusionMatrix[k][j];
            }
            assert(colSum == numInstances);
        }
    }

    private static String[] spaghettiLabels() {
        String[] labels = new String[6];
        labels[0] = Lab3.Category.airplanes.name();
        labels[1] = Lab3.Category.butterfly.name();
        labels[2] = Lab3.Category.flower.name();
        labels[5] = Lab3.Category.grand_piano.name();
        labels[3] = Lab3.Category.starfish.name();
        labels[4] = Lab3.Category.watch.name();
        return labels;
    }

}
