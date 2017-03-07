public class FullyConnectedLayer extends Layer {
    private int numHU;
    private double[][] weights;
    private double radius = 0.1;

    public FullyConnectedLayer(int numHU) {
        this.numHU = numHU;
    }
    
	@Override
	public double[][][] forward(double[][][] data, double cls) {
		// only uses first index of data
		double[][][] ret = new double[1][data[0].length][data[0][0].length];
		
		return ret;
	}

	public void setRadius(double rad) {
		this.radius = rad;
	}
	
	@Override
	public double[][][] backwards(double[][][] errs) {
		return null;
	}
	
	@Override
	public void randomInit() {
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[0].length; j++) {
				this.weights[i][j] = (Math.random() - 0.5) * radius;
			}
		}
	}
}
