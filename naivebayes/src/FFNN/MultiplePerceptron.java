/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package FFNN;

import java.util.ArrayList;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Mujahid Suriah
 */
public class MultiplePerceptron extends AbstractClassifier {
    ArrayList<Node> listNodeHidden;
    ArrayList<Node> listNodeOutput;
    int itteration;
    double learningRate;
    int numHiddenLayer;
    public MultiplePerceptron(int itt, double learn, int numHLayer) {
        listNodeHidden = new ArrayList<>();//inisialisasis listNodeHidden
        listNodeOutput = new ArrayList<>();
        itteration = itt;
        learningRate = learn;
        numHiddenLayer = numHLayer;
    }
    
    @Override 
    public void buildClassifier(Instances i){
        for (int hiddenLayer=0; hiddenLayer<numHiddenLayer; hiddenLayer++) {//buat neuron untuk hidden layer
            listNodeHidden.add(new Node(i.numAttributes()));
        }
        
        for (int numInstance = 0; numInstance<i.numClasses(); numInstance++) {//buat neuron untuk output
            listNodeOutput.add(new Node(numHiddenLayer));
        }
        ArrayList<Double> value = new ArrayList<>();
        for (int itt = 0; itt<itteration; itt++) {
            for (int numInstances=0; numInstances<i.numInstances()-1;numInstances++) {
                
                double sigma;
                
            }
        }
    }
    
    
    
    @Override 
    public double classifyInstance(Instance i) {
        return 0;
    }
}
