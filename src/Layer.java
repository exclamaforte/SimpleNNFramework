import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public abstract class Layer {
    /* outputs new weights for the forward pass */
    public abstract double[][][] forward(double[][][] data, double cls);
    /* takes in the propagated errors, and returns the new propagated errors. Presumably updates the weights within the layer as well */
    public abstract double[][][] backwards(double[][][] errs);
    public abstract void randomInit();
}
