import java.util.ArrayList;

public class NeuralNetwork {
    private InputLayer input;
    private OutputLayer output;
    private ArrayList<Layer> layers;
    private int inputSize;
    private int outputSize;
    public NeuralNetwork(int inputSize, int outputSize) {
        this.layers = new ArrayList<Layer>();
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.input = new InputLayer(inputSize);
        this.output = new OutputLayer(outputSize);
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
    public void addMaxPoolingLayer(int side, int numHU, int imgWidth, int imgHeight) {
        layers.add(new MaxPoolingLayer(side, numHU));
    }
    public void addFullyConnectedLayer(int numHU) {
        layers.add(new FullyConnectedLayer(numHU));
    }
    public void train(double[][][] train, double[] trainClass) {
        double[][][][] forward = new double[this.layers.size() + 2][][][];
        double[][][][] backward = new double[this.layers.size() + 1][][][];
        for (int instidx = 0; instidx < train.length; instidx++) {
        	double[][][] hold = new double[1][][];
        	hold[0] = train[instidx];
            int i = 1;
            forward[0] = input.forward(hold, trainClass[instidx]);
            for (Layer l : this.layers) {
                forward[i] = l.forward(forward[i - 1], trainClass[instidx]);
                i++;
            }
            forward[i] = output.forward(forward[i - 1], trainClass[instidx]);

            i--;
            backward[i] = output.backwards(forward[i]);
            for (Layer l : this.layers) {
                backward[i - 1] = l.backwards(backward[i]);
                i--;
            }
        }
    }
    public void test(double[][][] test, double[][][] testClass) {

    }
}
