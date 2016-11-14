/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes.pkg2.ai;

import java.util.ArrayList;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author Mujahid Suriah
 */
public class MultilayerPerceptron extends AbstractClassifier {
    private ArrayList<ArrayList<Double>> model;
    
    public MultilayerPerceptron() {
        model = new ArrayList<>();
    }
    
    private boolean isSame(double[] in1, double[] in2) {
        
        return false;
    }
    
    private double sigmoid(double input) {
        return 1/(1+Math.pow(Math.E,input*(-1)));
    }
    @Override
    public void buildClassifier(Instances inst) throws Exception {
            for (int i = 0; i < inst.numInstances(); i ++) {
                ArrayList<Double> tempList = new ArrayList<>();
                Double [] temp1 = new Double[inst.instance(i).numValues()];
                for (int j=0; j < inst.instance(i).numValues();j++) {
                    temp1[j] = inst.get(i).value(j);
                    tempList.add(inst.get(i).value(j));
                }
                Double [] temp2 = new Double[inst.instance(i).numValues()+1];
                for (int j=0; j<inst.instance(i).numValues();j++) {
                    if (j<inst.instance(i).numValues()) {
                        if (i == 0) {
                            temp2[j] = 0.0;
                            tempList.add(0.0);
                        }
                        else {
                            
                        }
                    }
                }
                for (int j=0;j<1;j++) {
                    double sigma=0;
                    sigma += temp1[j] * temp2[j+1];
                    tempList.add(sigma);
                    tempList.add(sigmoid(sigma));
                }
                model.add(tempList);
            }
        for(ArrayList<Double> inner : model) {
            for (double d : inner) {
                System.out.print(d+" ");
            }
            System.out.println();
        }
    }
    
    public static void main(String [] args) throws Exception {
        MultilayerPerceptron mp = new MultilayerPerceptron();
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(("D:\\Program Files\\Weka-3-8\\data\\iris.arff"));
        Instances train = source.getDataSet();
        train.setClassIndex(train.numAttributes()-1);
        mp.buildClassifier(train);
    }
}
