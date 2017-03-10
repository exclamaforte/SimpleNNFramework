import java.util.ArrayList;

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
    public void train(double[][][][] train, double[] trainClass) {

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


            output.backwards(i,forward,backward, learningRate, trainClass);
            i--;
            for (Layer l : this.layers) {
                l.backwards(i,forward,backward,LEARNING_RATE);
                i--;
            }
        }
    }

    public void test(double[][][] test, double[][][] testClass) {

    }
}
