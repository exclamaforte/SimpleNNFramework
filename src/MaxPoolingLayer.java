
public class MaxPoolingLayer extends Layer {
    private int poolingWidth;
    private int step;
    private int[][][] maxIndexes;


    public MaxPoolingLayer(int previousWidth, int previousDepth, int step , int poolingWidth) {
        this.poolingWidth = poolingWidth;
        this.step = step;
        this.previousWidth = previousWidth;
        this.previousDepth = previousDepth;

        //Get output dimensions
        int c = previousWidth + step - poolingWidth;
        assert(c % step == 0);
        this.outputWidth = c / step;
        this.outputDepth = this.previousDepth;

        this.maxIndexes = new int[outputDepth][poolingWidth][poolingWidth];
    }

    @Override
    public void forward(int layer, double[][][][] forwardData, double cls) {
        double[][][] data = forwardData[layer];
        assert(data.length == previousDepth);
        assert(data[0].length == previousWidth);

        double[][][] ret = forwardData[layer + 1];
        for (int imageNum = 0; imageNum < previousDepth; imageNum++) {
	        for (int i = 0; i < outputWidth; i++) {
	            for (int j = 0; j < outputWidth; j++) {
	                ret[imageNum][i][j] = -Double.MAX_VALUE;
	                for (int k = 0; k < poolingWidth; k++) {
	                    for (int l = 0; l < poolingWidth; l++) {
	                        if (data[imageNum][i * step + k][j * step + l] > ret[imageNum][i][j]) {
	                            ret[imageNum][i][j] = data[imageNum][i * step + k][j * step + l];
                                maxIndexes[imageNum][i][j] = k* poolingWidth + l;
	                        }
	                    }
	                }
	            }
	        }
        }
    }

    @Override
    public void backwards(int layer, double[][][][] forwardValues, double[][][][] backwardValues) {
        for (int imageNum = 0; imageNum < previousDepth; imageNum++) {
            for (int i = 0; i < outputWidth; i++) {
                for (int j = 0; j < outputWidth; j++) {
                    int maxIndex = maxIndexes[imageNum][i][j];
                    int maxK = maxIndex / poolingWidth;
                    int maxL = maxIndex % poolingWidth;
                    for (int k = 0; k < poolingWidth; k++) {
                        for (int l = 0; l < poolingWidth; l++) {
                            double deriv = 0;
                            if(k == maxK && l == maxL){
                                deriv = backwardValues[layer][imageNum][i][j];
                            }
                            backwardValues[layer-1][imageNum][i*step + k][j* step + l] = deriv;
                        }
                    }
                }
            }
        }
    }

	@Override
	public void randomInit() {
	}

}
