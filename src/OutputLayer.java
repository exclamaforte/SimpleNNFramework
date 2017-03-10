//Implements softmax for classification


public class OutputLayer extends Layer {
	private int outputSize;
	public OutputLayer(int previousSize, int outputSize) {
		super(previousSize, 1, outputSize, 1);
		this.outputSize = outputSize;
		this.expStorage = new double[outputSize];
	}
	private double[] expStorage;

	@Override
	public void forward(int layer, double[][][][] forwardData, double cls) {
		assert(this.previousDepth == 1);
		assert(this.previousWidth == outputSize);
		//only one row
		assert(forwardData[layer-1][0].length == 1);
		assert(forwardData[layer-1][0][0].length == 1);

		double C = 0;
		for(int i = 0; i < outputSize; i ++ ){
			expStorage[i] = Math.exp(forwardData[layer-1][i][0][0]);
			C += expStorage[i];
		}
		for(int i = 0; i < outputSize; i ++){
			forwardData[layer][i][0][0] = expStorage[i] / C;
		}
	}

	@Override
	public void backwards(int layer, double[][][][] forwardData, double[][][][] backwardData, double learningRate) {
		assert(false);
	}
	public void backwards(int layer, double[][][][] forwardData, double[][][][] backwardData, double learningRate, double[] target) {
		for (int i = 0 ; i < outputSize; i++) {
			backwardData[layer - 1][i][0][0] = forwardData[layer][i][0][0] - target[i];
		}
	}
	@Override
	public void randomInit() {
	}

}
