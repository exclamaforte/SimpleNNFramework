
public class MaxPoolingLayer extends Layer {
    private int poolingWidth;
    private int step;
    private int[][][] maxIndexes;


    public MaxPoolingLayer(int previousWidth, int previousDepth,int step , int poolingWidth) {
        super(previousWidth,previousDepth,(previousWidth + step - poolingWidth) / step, previousDepth);
        this.poolingWidth = poolingWidth;
        this.step = step;
        this.previousWidth = previousWidth;
        this.previousDepth = previousDepth;

        //Assert step and pooling width works
        int c = previousWidth + step - poolingWidth;
        assert(c % step == 0);


        this.maxIndexes = new int[outputDepth][outputWidth][outputWidth];
    }

    @Override
    public void forward(int layer, double[][][][] forwardData, double cls) {
        double[][][] data = forwardData[layer-1];
        assert(data.length == previousDepth);
        assert(data[0].length == previousWidth);

        double[][][] ret = forwardData[layer ];
        for (int imageNum = 0; imageNum < previousDepth; imageNum++) {
	        for (int output_i = 0; output_i < outputWidth; output_i++) {
	            for (int output_j = 0; output_j < outputWidth; output_j++) {
	                ret[imageNum][output_i][output_j] = Double.NEGATIVE_INFINITY;
	                for (int pooling_i = 0; pooling_i < poolingWidth; pooling_i++) {
	                    for (int pooling_j = 0; pooling_j < poolingWidth; pooling_j++) {
	                        if (data[imageNum][output_i * step + pooling_i][output_j * step + pooling_j] > ret[imageNum][output_i][output_j]) {
	                            ret[imageNum][output_i][output_j] = data[imageNum][output_i * step + pooling_i][output_j * step + pooling_j];
                                maxIndexes[imageNum][output_i][output_j] = pooling_i* poolingWidth + pooling_j;
	                        }
	                    }
	                }
	            }
	        }
        }
    }

    @Override
    public void forwardDropout(int layer, double[][][][] forwardData, double cls, boolean isTraining) {
        forward(layer,forwardData,cls);
    }

    @Override
    public void backwards(int layer, double[][][][] forwardValues, double[][][][] backwardValues,double learningRate) {
        for (int imageNum = 0; imageNum < previousDepth; imageNum++) {
            for (int output_i = 0; output_i < outputWidth; output_i++) {
                for (int output_j = 0; output_j < outputWidth; output_j++) {
                    int maxIndex = maxIndexes[imageNum][output_i][output_j];
                    int maxK = maxIndex / poolingWidth;
                    int maxL = maxIndex % poolingWidth;
                    for (int k = 0; k < poolingWidth; k++) {
                        for (int l = 0; l < poolingWidth; l++) {
                            double deriv = 0;
                            if(k == maxK && l == maxL){
                                deriv = backwardValues[layer][imageNum][output_i][output_j];
                            }
                            backwardValues[layer-1][imageNum][output_i*step + k][output_j* step + l] = deriv;
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
