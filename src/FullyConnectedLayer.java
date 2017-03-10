public class FullyConnectedLayer extends Layer {
    private double[][] weights;
    private double bias[]; 
    private double radius = 0.1;

    public FullyConnectedLayer(int previousWidth, int numHU) {
    	super(previousWidth, 1, numHU, 1);
    	weights = new double[numHU][previousWidth];
    	bias = new double[numHU];
    }

    @Override
    public void randomInit() {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                this.weights[i][j] = (Math.random() - 0.5) * radius;
            }
        }
    }
	@Override
	public void forward(int layer, double[][][][] forwardData, double cls) {
		/*for(int outIndex = 0; outIndex < outputWidth; outIndex++){
			double sum = 0;
			for(int depth = 0; k < previousDepth; k ++) {
				for(int l = 0; l < kernelWidth; l ++) {
					for(int m = 0; m < kernelWidth; m ++) {
						sum += forwardData[layer -1][k][i * step + l][i * step + m] * kernels[kernel][k][l][m];
					}
				}
			}
			sum += biases[kernel];
			forwardData[layer][kernel][i][j] = Math.max(0, sum);
		}*/
 
		for (int outIndex = 0; outIndex < outputWidth; outIndex++) {
			double sum = 0.0;
			for (int prevIndex = 0; prevIndex < previousWidth; prevIndex++) {
				sum += forwardData[layer - 1][prevIndex][0][0] * weights[outIndex][prevIndex];
			}
			sum += bias[outIndex];
			forwardData[layer][outIndex][0][0] = Math.max(0, sum);
		}
	}

	@Override
	public void backwards(int layer, double[][][][] forwardData, double[][][][] backwardData, double learningRate) {
        for(int i = 0; i < backwardData[layer-1].length; i ++) {
            for(int j = 0; j < backwardData[layer-1][0].length; j ++) {
                for(int k = 0; k < backwardData[layer-1][0][0].length; k ++) {
                    backwardData[layer-1][i][j][k] = 0;
                }
            }
        }

        for (int currNodeNum = 0; currNodeNum < outputWidth; currNodeNum++) {
        	// set the deltas
        	if (forwardData[layer][currNodeNum][0][0] > 0) {
    			for (int nextNodeNum = 0; nextNodeNum < backwardData[layer].length; nextNodeNum++) {
    				backwardData[layer - 1][currNodeNum][0][0] += 
    						weights[currNodeNum][nextNodeNum] * backwardData[layer][nextNodeNum][0][0];
            	}
            	for (int nextNodeNum = 0; nextNodeNum < backwardData[layer].length; nextNodeNum++) {
            		weights[currNodeNum][nextNodeNum] += 
            				learningRate * forwardData[layer - 1][currNodeNum][0][0] * backwardData[layer][nextNodeNum][0][0];
            	}
            	bias[currNodeNum] -= learningRate * backwardData[layer][currNodeNum][0][0];
    		}
        }
	}
}
