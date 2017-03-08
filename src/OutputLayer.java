//Implements softmax for classification


public class OutputLayer extends Layer {
	private int outputSize;
	public OutputLayer(int outputSize) {

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

		double C = 0;
		for(int i = 0; i < outputSize; i ++ ){
			expStorage[i] = Math.exp(forwardData[layer-1][0][0][i]);
			C += expStorage[i];
		}
		for(int i = 0; i < outputSize; i ++){
				forwardData[layer][0][0][i] = expStorage[i] / C;
		}
	}

	@Override
	public void backwards(int layer, double[][][][] forwardVals, double[][][][] backwardsVals ) {
		// TODO Auto-generated method stub

	}

	@Override
	public void randomInit() {
		// TODO Auto-generated method stub
		
	}

}
