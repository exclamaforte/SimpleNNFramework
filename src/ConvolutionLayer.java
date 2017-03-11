import java.util.Random;

public class ConvolutionLayer extends Layer {

    private int kernelWidth;
    private int step;
    public double[][][][] kernels;
    public double[] biases;
    private double initRad;
    private double[][][] dropoutMultiplier;

    public double[][][][] bestKernels;
    public double[] bestBiases;

    public ConvolutionLayer(int previousWidth, int previousDepth, int numKernels, int kernelWidth, int step, double initRad) {
        super(previousWidth, previousDepth, (previousWidth + step - kernelWidth) / step, numKernels);
        this.initRad = initRad;
        this.kernelWidth = kernelWidth;
        this.step = step;

        //Make sure step and kernel width work
        assert((previousWidth + step - kernelWidth) % step == 0);

        kernels = new double[numKernels][previousDepth][kernelWidth][kernelWidth];
        bestKernels = new double[numKernels][previousDepth][kernelWidth][kernelWidth];
        biases = new double[numKernels];
        bestBiases = new double[numKernels];
        dropoutMultiplier = new double[outputDepth][outputWidth][outputWidth];

        for(int i = 0; i < outputDepth; i ++){
            for(int j = 0; j < outputWidth; j ++){
                for(int k = 0; k < outputWidth; k ++){
                    dropoutMultiplier[i][j][k] = 1;
                }
            }
        }

    }

    public void randomInit() {
        for (int i = 0; i < kernels.length; i++) {
            for (int j = 0; j < kernels[0].length; j++) {
                for (int k = 0; k < kernels[0][0].length; k++) {
                    for(int l = 0; l < kernels[0][0][0].length; l ++)
                    	kernels[i][j][k][l] = (gen.nextDouble() - 0.5) * initRad;
                }
            }
        }
        for(int i = 0; i < biases.length; i ++){
            biases[i] = 5;
        }
    }

    @Override
    public void cacheBestWeights() {
        for(int i = 0; i < kernels.length; i ++){
            for(int j = 0; j < kernels[0].length; j ++){
                for(int k = 0; k < kernels[0][0].length; k ++){
                    for(int l = 0; l < kernels[0][0][0].length; l ++){
                        bestKernels[i][j][k][l] = kernels[i][j][k][l];
                    }
                }
            }
        }
        for(int i = 0; i < biases.length; i ++){
            bestBiases[i] = biases[i];
        }
    }

    @Override
    public void resetToBestWeights(){
        for(int i = 0; i < kernels.length; i ++){
            for(int j = 0; j < kernels[0].length; j ++){
                for(int k = 0; k < kernels[0][0].length; k ++){
                    for(int l = 0; l < kernels[0][0][0].length; l ++){
                        kernels[i][j][k][l] = bestKernels[i][j][k][l];
                    }
                }
            }
        }
        for(int i = 0; i < biases.length; i ++){
            biases[i] = bestBiases[i];
        }
    }

    @Override
    public void forward(int layer, double[][][][] forwardData, double[] cls) {
        for(int output_i = 0; output_i < outputDepth; output_i ++){
            for(int output_j = 0; output_j < outputWidth; output_j ++){
                for(int output_k = 0; output_k < outputWidth; output_k ++){
                    double sum = 0;
                    if(dropoutMultiplier[output_i][output_j][output_k] != 0) {
                        for (int kernel_i = 0; kernel_i < previousDepth; kernel_i++) {
                            for (int kernel_j = 0; kernel_j < kernelWidth; kernel_j++) {
                                for (int kernel_k = 0; kernel_k < kernelWidth; kernel_k++) {
                                    sum += forwardData[layer - 1][kernel_i][output_j * step + kernel_j][output_k * step + kernel_k] * kernels[output_i][kernel_i][kernel_j][kernel_k];
                                }
                            }
                        }
                        sum += biases[output_i];
                    }
                    forwardData[layer][output_i][output_j][output_k] = dropoutMultiplier[output_i][output_j][output_k] *
                            Math.max(0.01 * sum, sum);
                }
            }
        }
    }

    @Override
    public void forwardDropout(int layer, double[][][][] forwardData, double[] cls, boolean isTraining) {
        for(int i = 0; i < outputDepth; i ++){
            for(int j = 0;  j < outputWidth;  j ++){
                for(int k = 0; k < outputWidth; k ++){
                    if(isTraining){
                        dropoutMultiplier[i][j][k] = (Math.random() < DropoutRate ? 0 : 1);
                    }
                    else{
                        dropoutMultiplier[i][j][k] = 1 - DropoutRate;
                    }
                }
            }
        }
        forward(layer,forwardData,cls);
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


        for(int kernel_i = 0; kernel_i < outputDepth; kernel_i++) {
            double biasDeltaSum = 0;
            for(int kernel_j = 0; kernel_j < previousDepth; kernel_j++) {
                for (int kernel_k = 0; kernel_k < kernelWidth; kernel_k++) {
                    for (int kernel_l = 0; kernel_l < kernelWidth; kernel_l++) {
                        double kernelElementDeltaSum = 0;
                        for (int output_i = 0; output_i < outputWidth; output_i++) {
                            for (int output_j = 0; output_j < outputWidth; output_j++) {

                                if(dropoutMultiplier[kernel_i][output_i][output_j]!= 0) {
                                    double multiplier = 1;
                                    // check if leaky
                                    if (forwardData[layer][kernel_i][output_i][output_j] <= 0)
                                        multiplier = 0.01;

                                    double derivOut = backwardData[layer][kernel_i][output_i][output_j];

                                    //only calculate bias term once
                                    //Im sorry if this code makes you mad
                                    if (kernel_j == 0 && kernel_k == 0 && kernel_l == 0)
                                        biasDeltaSum += derivOut * multiplier;

                                    kernelElementDeltaSum +=
                                            derivOut * multiplier *
                                                    forwardData[layer - 1][kernel_j][output_i * step + kernel_k][output_j * step + kernel_l];
                                    // is it forwardData[layer-1] or forwardData[layer]
                                    backwardData[layer - 1][kernel_j][output_i * step + kernel_k][output_j * step + kernel_l]
                                            += derivOut * kernels[kernel_i][kernel_j][kernel_k][kernel_l] * multiplier;
                                }
                            }
                        }
                        kernels[kernel_i][kernel_j][kernel_k][kernel_l] -= kernelElementDeltaSum* learningRate;
                    }
                }
            }
            biases[kernel_i] -= biasDeltaSum * learningRate;
        }
    }

    public static int Seed = (int)(Math.random()*100);
    public static Random gen = new Random(Seed);


}
