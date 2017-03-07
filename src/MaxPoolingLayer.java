
public class MaxPoolingLayer extends Layer {
    private int side;
    private int[][] backIndex;

    public MaxPoolingLayer(int side, int numHU) {
        this.side = side;
        this.backIndex = new int[side][side];
    }

    @Override
    public double[][][] forward(double[][][] data, double cls) {
        assert((data[0].length % side) == 0);
        
        int newSideLength = data.length / side;

        double[][][] ret = new double[data.length][newSideLength][newSideLength];
        for (int imageNum = 0; imageNum < data.length; imageNum++) {
	        for (int i = 0; i < newSideLength; i++) {
	            for (int j = 0; j < newSideLength; j++) {
	                ret[imageNum][i][j] = -Double.MAX_VALUE;
	                for (int k = 0; k < side; k++) {
	                    for (int l = 0; l < side; l++) {
	                        if (data[imageNum][i * side + k][j * side + l] > ret[imageNum][i][j]) {
	                            ret[imageNum][i][j] = data[imageNum][i * side + k][j * side + l];
	                        }
	                    }
	                }
	            }
	        }
        }
        return ret;
    }

    @Override
    public double[][][] backwards(double[][][] errs) {
        return null;
    }

	@Override
	public void randomInit() {
	}

}
