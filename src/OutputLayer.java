
public class OutputLayer extends Layer {
	private int outputSize;
	public OutputLayer(int outputSize) {
		this.outputSize = outputSize;
	}

	@Override
	public double[][][] forward(double[][][] data, double cls) {
		return data;
	}

	@Override
	public double[][][] backwards(double[][][] errs) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void randomInit() {
		// TODO Auto-generated method stub
		
	}

}
