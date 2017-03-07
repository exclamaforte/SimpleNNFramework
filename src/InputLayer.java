public class InputLayer extends Layer {
    private int featureVectorSize;
    public InputLayer(int featureVectorSize) {
        this.featureVectorSize = featureVectorSize;
    }

    @Override
    public double[][][] forward(double[][][] data, double cls) {
        return data;
    }

    @Override
    public double[][][] backwards(double[][][] errs) {
        return errs;
    }

	@Override
	public void randomInit() {
	}
}
