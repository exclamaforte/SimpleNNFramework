public class FullyConnectedLayer extends Layer {
    private double[][] weights;
    private double bias[]; 
    private double radius = 0.1;
	private double[] dropoutMultiplier;

    public FullyConnectedLayer(int previousDepth, int numHU) {
    	super(1, previousDepth, 1, numHU);
    	weights = new double[numHU][previousDepth];
    	bias = new double[numHU];
		dropoutMultiplier = new double[numHU];
		for(int i = 0; i < numHU; i ++){
			dropoutMultiplier[i] = 1;
		}
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
		for (int outIndex = 0; outIndex < outputDepth; outIndex++) {
			double sum = 0.0;
			for (int prevIndex = 0; prevIndex < previousDepth; prevIndex++) {
				sum += forwardData[layer - 1][prevIndex][0][0] * weights[outIndex][prevIndex];
			}
			sum += bias[outIndex];
			forwardData[layer][outIndex][0][0] = dropoutMultiplier[outIndex]*Math.max(0, sum);
		}
	}

	@Override
	public void forwardDropout(int layer, double[][][][] forwardData, double cls, boolean isTraining) {
		for(int i = 0; i < dropoutMultiplier.length; i ++){
			if(isTraining)
				dropoutMultiplier[i] = (Math.random() < DropoutRate ? 0 : 1);
			else
				dropoutMultiplier[i] = 1-DropoutRate;
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

        for (int currNodeNum = 0; currNodeNum < outputDepth; currNodeNum++) {
        	// set the deltas
        	if (forwardData[layer][currNodeNum][0][0] > 0) {
    			for (int nextNodeNum = 0; nextNodeNum < previousDepth; nextNodeNum++) {
    				backwardData[layer - 1][currNodeNum][0][0] += 
    						weights[currNodeNum][nextNodeNum] * backwardData[layer][nextNodeNum][0][0];
            	}
            	for (int nextNodeNum = 0; nextNodeNum < previousDepth; nextNodeNum++) {
            		weights[currNodeNum][nextNodeNum] += 
            				learningRate * forwardData[layer - 1][currNodeNum][0][0] * backwardData[layer][nextNodeNum][0][0];
            	}
            	bias[currNodeNum] -= learningRate * backwardData[layer][currNodeNum][0][0];
    		}
        }
	}
}
