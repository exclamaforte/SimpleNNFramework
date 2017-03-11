/**
 * @Author: Yuting Liu
 * This is the dataset class that holds in the whole dataset
 *
 */
import java.util.ArrayList;

public class Dataset {
	// the list of all instances
	private ArrayList<Instance> instances;

	public Dataset() {
		this.instances = new ArrayList<Instance>();
	}

	// get the size of the dataset
	public int getSize() {
		return instances.size();
	}
	public int getImageWidth() {
		return instances.get(0).getWidth();
	}
	public int getImageHeight() {
		return instances.get(0).getHeight();
	}
	public ArrayList<Instance> getImages() {
		return instances;
	}
	// add instance into the data set
	public void add(Instance inst) {
		instances.add(inst);
	}
}
