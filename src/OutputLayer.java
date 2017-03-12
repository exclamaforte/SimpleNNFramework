//Implements softmax for classification


import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class OutputLayer extends Layer {
	private int outputSize;
	public OutputLayer(int previousSize, int outputSize) {
		super(previousSize, 1, outputSize, 1);
		this.outputSize = outputSize;
		this.expStorage = new double[outputSize];
	}
	private double[] expStorage;

	@Override
	public void forward(int layer, double[][][][] forwardData) {
		assert(this.previousDepth == 1);
		assert(this.previousWidth == outputSize);
		//only one row
		assert(forwardData[layer-1][0].length == 1);
		assert(forwardData[layer-1][0][0].length == 1);

		double max = -Double.MAX_VALUE;
		for (int i = 0; i < outputSize; i++) {
			if (forwardData[layer-1][i][0][0] > max) {
				max = forwardData[layer-1][i][0][0];
			}
		}
		double C = 0;
		for(int i = 0; i < outputSize; i ++ ){
			expStorage[i] = Math.exp(forwardData[layer-1][i][0][0] - max);
			C += expStorage[i];
		}
		for(int i = 0; i < outputSize; i ++){
			forwardData[layer][i][0][0] = expStorage[i] / C;
		}
	}

	@Override
	public void forwardDropout(int layer, double[][][][] forwardData, boolean isTraining) {
		forward(layer,forwardData);
	}

	@Override
	public void backwards(int layer, double[][][][] forwardData, double[][][][] backwardData, double learningRate) {
		throw new NotImplementedException();
	}

	public void backwards(int layer, double[][][][] forwardData, double[][][][] backwardData, double learningRate, double[] target) {
		for (int i = 0 ; i < outputSize; i++) {
			backwardData[layer - 1][i][0][0] = forwardData[layer][i][0][0] - target[i];
		}
	}

	@Override
	public void randomInit() {
	}

	@Override
	public void cacheBestWeights() {
		//do nothing
	}
	@Override
	public void resetToBestWeights(){
		//do nothing
	}

}
