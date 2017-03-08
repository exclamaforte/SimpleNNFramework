import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public abstract class Layer {
    /* outputs new weights for the forward pass */
    public abstract void forward(int layer, double[][][][] forwardData, double cls);
    /* takes in the propagated errors, and returns the new propagated errors. Presumably updates the weights within the layer as well */
    public abstract void backwards(int layer, double[][][][] forwardData, double[][][][] backwardData);
    public abstract void randomInit();
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
