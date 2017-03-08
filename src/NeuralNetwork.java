import java.util.ArrayList;

public class NeuralNetwork {
    private int inputDepth;
    private int inputWidth;
    private OutputLayer output;
    private ArrayList<Layer> layers;
    private int outputClasses;
    public NeuralNetwork(int inputWidth, int inputDepth, int outputClasses) {
        this.layers = new ArrayList<Layer>();
        this.inputDepth = inputDepth;
        this.inputWidth = inputWidth;
        this.outputClasses = outputClasses;
        this.output = new OutputLayer(outputClasses);
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
    public void addConvolutionLayer(int side, int numHU, int imgWidth, int imgHeight) {
        layers.add(new ConvolutionLayer(side, numHU));
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
        layers.add(new FullyConnectedLayer(numHU));
    }
    public void train(double[][][][] train, double[] trainClass) {
        double[][][][] forward = new double[this.layers.size() + 2][][][];
        double[][][][] backward = new double[this.layers.size() + 2][][][];
        for (int instidx = 0; instidx < train.length; instidx++) {
        	forward[0] = train[instidx];

            int i = 1;
            for (Layer l : this.layers) {
                l.forward(i,forward, trainClass[instidx]);
                i++;
            }
            output.forward(i, forward, trainClass[instidx]);


            output.backwards(i,forward,backward);
            i--;
            for (Layer l : this.layers) {
                l.backwards(i,forward,backward);
                i--;
            }
        }
    }

    public void test(double[][][] test, double[][][] testClass) {

    }
}
