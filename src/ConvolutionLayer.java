public class ConvolutionLayer extends Layer {

    private int kernelWidth;
    private int step;
    public double[][][][] kernels;
    public double[] biases;
    private double initRad;

    public ConvolutionLayer(int previousWidth, int previousDepth, int numKernels, int kernelWidth, int step, double initRad) {
        super(previousWidth,previousDepth,(previousWidth + step - kernelWidth) / step, numKernels);
        this.initRad = initRad;
        this.kernelWidth = kernelWidth;
        this.step = step;

        //Make sure step and kernel width work
        assert((previousWidth + step - kernelWidth) % step == 0);

        kernels = new double[numKernels][previousDepth][kernelWidth][kernelWidth];
        biases = new double[numKernels];



    }

    public void randomInit() {
        for (int i = 0; i < kernels.length; i++) {
            for (int j = 0; j < kernels[0].length; j++) {
                for (int k = 0; k < kernels[0][0].length; k++) {
                    for(int l = 0; l < kernels[0][0][0].length; l ++)
                    kernels[i][j][k][l] = (Math.random() - 0.5) * initRad;
                }
            }
        }
    }

    @Override
    public void forward(int layer, double[][][][] forwardData, double cls) {
        for(int kernel = 0; kernel < outputDepth; kernel ++){
            for(int i = 0; i < outputWidth; i ++){
                for(int j = 0; j < outputWidth; j ++){
                    double sum = 0;
                    for(int k = 0; k < previousDepth; k ++){
                        for(int l = 0; l < kernelWidth; l ++){
                            for(int m = 0; m < kernelWidth; m ++){
                                sum += forwardData[layer -1][k][i * step + l][i * step + m] * kernels[kernel][k][l][m];
                            }
                        }
                    }
                    sum += biases[kernel];
                    forwardData[layer][kernel][i][j] = Math.max(0, sum);
                }
            }
        }
    }

    @Override
    public void backwards(int layer, double[][][][] forwardData, double[][][][] backwardData, double learningRate) {
        //We need to first initialize backwards values for previous layer since we reuse the backwardData array
        for(int i = 0; i < backwardData[layer-1].length; i ++){
            for(int j = 0; j < backwardData[layer-1][0].length; j ++){
                for(int k = 0; k < backwardData[layer-1][0][0].length; k ++){
                    backwardData[layer-1][i][j][k] = 0;
                }
            }
        }


        for(int kernel_i = 0; kernel_i < outputDepth; kernel_i ++){
            double biasDeltaSum = 0;
            for(int kernel_j = 0; kernel_j < previousDepth; kernel_j ++) {
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
