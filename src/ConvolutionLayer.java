import java.util.Random;

public class ConvolutionLayer extends Layer {

    private int kernelWidth;
    private int step;
    public double[][][][] kernels;
    public double[] biases;
    private double initRad;
    private double[][][] dropoutMultiplier;

    public ConvolutionLayer(int previousWidth, int previousDepth, int numKernels, int kernelWidth, int step, double initRad) {
        super(previousWidth, previousDepth, (previousWidth + step - kernelWidth) / step, numKernels);
        this.initRad = initRad;
        this.kernelWidth = kernelWidth;
        this.step = step;

        //Make sure step and kernel width work
        assert((previousWidth + step - kernelWidth) % step == 0);

        kernels = new double[numKernels][previousDepth][kernelWidth][kernelWidth];
        biases = new double[numKernels];
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
    public void forward(int layer, double[][][][] forwardData, double cls) {
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
    public void forwardDropout(int layer, double[][][][] forwardData, double cls, boolean isTraining) {
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

    public static void main(String[] args){










        int inputDepth =1;
        int inputWidth = 4;
        int outputDepth = 1;
        int step = 2;
        int kernelWidth = 2;
        int kernelDepth = inputDepth;
        int outputWidth = (inputWidth + step - kernelWidth) / step;

        int maxStep = 2;
        int maxDepth = outputDepth;
        int poolingWidth = 2;
        int maxWidth = (outputWidth + maxStep - poolingWidth)/maxStep;

        MaxPoolingLayer maxLayer = new MaxPoolingLayer(outputWidth,outputDepth,maxStep,poolingWidth);

        //System.out.println(maxLayer.getOutputDepth());


        ConvolutionLayer layer = new ConvolutionLayer(inputWidth,inputDepth,outputDepth,kernelWidth,step,1);



        boolean random = true;


        for(int d = 0; d < outputDepth; d ++) {
            layer.biases[d] = 0;
            for (int i = 0; i < kernelDepth; i++) {
                for (int j = 0; j < kernelWidth; j++) {
                    for (int k = 0; k < kernelWidth; k++) {
                        layer.kernels[d][i][j][k] = j %2 + k %2;
                    }
                }
            }
        }



        System.out.println(layer.getOutputWidth());
        double[][][][] forward = new double[3][][][];
        forward[0] = new double[inputDepth][inputWidth][inputWidth];
        forward[1] = new double[outputDepth][outputWidth][outputWidth];
        forward[2] = new double[maxDepth][maxWidth][maxWidth];

        double[][][][] backward = new double[3][][][];
        backward[0] = new double [inputDepth][inputWidth][inputWidth];
        backward[1] = new double[outputDepth][outputWidth][outputWidth];
        backward[2] = new double[maxDepth][maxWidth][maxWidth];

        // do analytic;
        for(int i = 0; i < inputDepth; i ++){
            for(int j = 0; j <inputWidth; j ++){
                for(int k = 0; k < inputWidth; k ++){
                    if(random)
                        forward[0][i][j][k] = gen.nextDouble();
                    else
                        forward[0][i][j][k] = ((i+j+k) == 0 ? 1 : 0);
                }
            }
        }





        int check_i = 0;
        int check_j = 0;
        int check_k = 0;



        for(int i = 0; i < outputDepth; i ++){
            for(int j = 0; j < maxWidth; j ++){
                for(int k = 0; k < maxWidth; k ++){
                    backward[2][i][j][k] = 1;
                }
            }
        }

        double eps = 0.0000000001;








        for(int i = 0; i < 10000; i ++) {

            double startKernelValue  = layer.kernels[0][0][0][0];
            layer.kernels[0][0][0][0] += eps;

            layer.forward(1,forward,0);
            maxLayer.forward(2,forward,0);
            double kernelRight = sum(forward[2]);

            layer.kernels[0][0][0][0] -= 2 * eps;
            layer.forward(1, forward, 0);
            maxLayer.forward(2, forward, 0);
            double kernelLeft = sum(forward[2]);

            double kernelNumerical = (kernelRight - kernelLeft) / (2 * eps);

            layer.kernels[0][0][0][0] = startKernelValue;

            layer.forward(1, forward, 0);
            maxLayer.forward(1, forward, 0);
            backward[2][0][0][0] = -(forward[2][0][0][0]- 12);

            maxLayer.backwards(2, forward, backward, 0.001);
            layer.backwards(1, forward, backward, 0.001);

            double kernelAnalytical = (layer.kernels[0][0][0][0] - startKernelValue);

            System.out.println("Kernel numerical " + kernelNumerical);
            System.out.println("kernel analytical " + kernelAnalytical);
            System.out.println("output " + forward[2][0][0][0]);
            System.out.println("******************\n");
        }








        forward[0][check_i][check_j][check_k] += eps;

        layer.forward(1,forward,0);
        maxLayer.forward(2,forward,0);
        double right = sum(forward[2]);
        forward[0][check_i][check_j][check_k] -= 2 * eps;
        layer.forward(1,forward,0);
        maxLayer.forward(2,forward,0);
        double left = sum(forward[2]);

        double numericalDeriv = (right - left)/(2*eps);

        //analytical

        forward[0][check_i][check_j][check_k] += eps;

        layer.forward(1,forward,0);
        maxLayer.forward(2,forward,0);

        for(int i = 0; i < outputDepth; i ++){
            for(int j = 0; j < maxWidth; j ++){
                for(int k = 0; k < maxWidth; k ++){
                    backward[2][i][j][k] = 1;
                }
            }
        }

        maxLayer.backwards(2,forward,backward,0.001);
        layer.backwards(1,forward,backward,0.001);

        double analyticalDeriv = backward[0][check_i][check_j][check_k];

        System.out.println("Numerical Deriv Input" + numericalDeriv);
        System.out.println("Analytical Deriv Input" + analyticalDeriv);

        System.out.println("Abs diff " + Math.abs(analyticalDeriv - numericalDeriv));

        System.out.println(layer.biases[0]);
        System.out.println(layer.kernels[0][0][0][0]);




    }

    public static double sum(double[][][] arr){
        double sum = 0;
        for(int i = 0; i < arr.length; i ++){
            for(int j = 0; j < arr[0].length; j ++){
                for(int k = 0; k < arr[0][0].length; k ++){
                    sum += arr[i][j][k];
                }
            }
        }
        return sum;
    }
}
