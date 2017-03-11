import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public abstract class Layer {

    public static final double DropoutRate = 0.5;

    /* outputs new weights for the forward pass */
    public abstract void forward(int layer, double[][][][] forwardData, double cls);
    public abstract void forwardDropout(int layer, double[][][][] forwardData, double cls, boolean isTraining);
    /* takes in the propagated errors, and returns the new propagated errors.
       Presumably updates the weights within the layer as well */
    public abstract void backwards(int layer,
                                   double[][][][] forwardData,
                                   double[][][][] backwardData,
                                   double learningRate);
    public abstract void randomInit();

    public Layer(int previousWidth, int previousDepth, int outputWidth, int outputDepth) {
        this.previousWidth = previousWidth;
        this.previousDepth = previousDepth;
        this.outputDepth = outputDepth;
        this.outputWidth= outputWidth;
    }

    protected int outputWidth;
    protected int outputDepth;
    protected int previousWidth;
    protected int previousDepth;

    public int getOutputWidth(){
        return outputWidth;
    }
    public int getOutputDepth(){
        return outputDepth;
    }
    public int getPreviousWidth(){
        return previousWidth;
    }
    public int getPreviousDepth(){
        return previousDepth;
    }
}
