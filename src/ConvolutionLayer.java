import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class ConvolutionLayer extends Layer {
    public int side;
    public int numHU;
    public double[][][] kernels;
    public double initRad;
    public ConvolutionLayer(int side, int numHU) {
        this.side = side;
        this.numHU = numHU;
        this.kernels = new double[numHU][side][side];
    }
    public void setInitRadius(double ir) {
        this.initRad = ir;
    }

    public void randomInit() {
        for (int i = 0; i < kernels.length; i++) {
            for (int j = 0; j < kernels.length; j++) {
                for (int k = 0; k < kernels.length; k++) {
                    kernels[i][j][k] = (Math.random() - 0.5) * initRad;
                }
            }
        }
    }

    @Override
    public void forward(int layer, double[][][][] forwardData, double cls) {

    }

    @Override
    public void backwards(int layer, double[][][][] forwardData, double[][][][] backwardData) {

    }

}
