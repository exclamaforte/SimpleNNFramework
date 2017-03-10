public class FullyConnectedLayer extends Layer {
    private double[][] weights;
    private double bias; 
    private double radius = 0.1;

    public FullyConnectedLayer(int previousWidth, int numHU) {
    	super(previousWidth, 1, numHU, 1);
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
			sum += bias;
			forwardData[layer][outIndex][0][0] = Math.max(0, sum);
		}
	}
	@Override
	public void backwards(int layer, double[][][][] forwardData, double[][][][] backwardData, double learningRate) {
        for(int i = 0; i < backwardData[layer-1].length; i ++){
            for(int j = 0; j < backwardData[layer-1][0].length; j ++){
                for(int k = 0; k < backwardData[layer-1][0][0].length; k ++){
                    backwardData[layer-1][i][j][k] = 0;
                }
            }
        }
        
        for(int kernel_i = 0; kernel_i < outputDepth; kernel_i++) {
            double biasDeltaSum = 0;
            for(int kernel_j = 0; kernel_j < previousDepth; kernel_j++) {
                for (int kernel_k = 0; kernel_k < kernelWidth; kernel_k++) {
                    for (int kernel_l = 0; kernel_l < kernelWidth; kernel_l++) {
                        double kernelElementDeltaSum = 0;
                        for (int output_i = 0; output_i < outputWidth; output_i++) {
                            for (int output_j = 0; output_j < outputWidth; output_j++) {
                                //check if derivative is 0
                                if(forwardData[layer][kernel_i][output_i][output_j] == 0)
                                    continue;

                                double derivOut = backwardData[layer][kernel_i][output_i][output_j];

                                //only calculate bias term once
                                //Im sorry if this code makes you mad
                                if(kernel_j == 0 && kernel_k == 0 && kernel_l == 0)
                                    biasDeltaSum += derivOut;

                                kernelElementDeltaSum +=
                                    derivOut*
                                    forwardData[layer -1][kernel_j][output_i * step + kernel_k][output_j * step +kernel_l];
                                backwardData[layer-1][kernel_j][output_i*step + kernel_k][output_j*step + kernel_l]
                                    += derivOut*kernels[kernel_i][kernel_j][kernel_k][kernel_l];
                            }
                        }
                        kernels[kernel_i][kernel_j][kernel_k][kernel_l] += kernelElementDeltaSum* learningRate;
                    }
                }
                biases[kernel_i] += biasDeltaSum * learningRate;
            }
        }
		
	}
}
