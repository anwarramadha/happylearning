/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ANN_single2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author Mujahid Suriah
 */
public class SinglelayerPerceptron extends AbstractClassifier {
    ArrayList<Node> listOutput;
    double[] listDoubleinstance;
    double learningRate;
    int itteration;
    double threshold;
    static int fold = 0;
    public SinglelayerPerceptron (int itt, double rate, double thres) {
        itteration = itt;
        learningRate = rate;
        threshold = thres;
    } 
    
    @Override
    public void buildClassifier(Instances i) {
        listOutput = new ArrayList<>();
        for (int idx = 0; idx < i.numClasses(); idx++) {
            listOutput.add(new Node(i.numAttributes()));
        }
        
        //mengubah class menjadi numeric (diambil indexnya)
        listDoubleinstance = new double[i.numInstances()];
        for (int numIns=0; numIns<i.numInstances();numIns++) {
            listDoubleinstance[numIns] = i.instance(numIns).toDoubleArray()[i.classIndex()];
        }
        
        double error = 0;
        for (int iter = 0; iter < itteration; iter++) {
            double errorThres = 0;
            for (int idxInstance = 0; idxInstance < i.numInstances(); idxInstance++) {
                
                //buat list input
                ArrayList<Double> listInput = new ArrayList<>();
                listInput.add(1.0); //ini bias
                for (int idx=0; idx < i.numAttributes()-1; idx++) {
                    listInput.add(i.get(idxInstance).value(idx));
                }
                
                //Hitung output rumus = sigmoid dari sigma
                for(int idxOut=0; idxOut < listOutput.size(); idxOut++) {
                    output(listInput, idxOut);
                }
                
                //Hitung error
                calculateError(idxInstance);
                //update bobot
                updateBobot(listInput);
                
            }
            for (int idxOut=0; idxOut < listOutput.size(); idxOut++) {
                errorThres += Math.pow(listOutput.get(idxOut).getError(), 2)/2;
            }
            if (errorThres <= threshold) break;
//            System.out.println(errorThres);
        }
//        fold++;
//        for (int idx =0; idx < i.numInstances(); idx++) {
//            for (int idxOut=0; idxOut < listOutput.size(); idxOut++) {
//                error += Math.pow(listOutput.get(idxOut).getError(), 2)/2;
//            }
//        }
//        System.out.println("Fold " + fold);
//        System.out.println("error " + error);
    }
    
    @Override
    public double classifyInstance(Instance i) {
        ArrayList<Double> listInput = new ArrayList<>();
        listInput.add(1.0); //ini bias
        for (int idx=0; idx < i.numAttributes()-1; idx++) {
            listInput.add(i.value(idx));
        }

        //Hitung output rumus = sigmoid dari sigma
        for(int idxOut=0; idxOut < listOutput.size(); idxOut++) {
            output(listInput, idxOut);
        }
        return maxValue();
    }
    
    private void calculateError(int idx) {
        //Hitung error pakai rumus target - output
        double [] target = target(idx);
        for (int i=0; i < listOutput.size(); i++) {
            listOutput.get(i).setError(target[i] - listOutput.get(i).getValue());
//            System.out.println(i+" "+listOutput.get(i).getError());
        }
    }
    
    //Mengubah kelas numeric menjadi 1 atau 0
    //Misal [iris-setosa, iris-versicolor, iris-virginica] -> [0,1,2] (numeric)
    //Jika nilai kelas pada setiap instance sama dengan salah satu nilai pada array kelas numeric
    //maka nilai pada index tersebut di assign 1, else 0
    private double[] target(int idx) {
        double [] result = new double[listOutput.size()];
        for (int i = 0; i < listOutput.size(); i++) {
            if (i == listDoubleinstance[idx]) {
                result[i] = 1;
            }else result[i] = 0;
        }
        return result;
    }
    
    private void updateBobot(ArrayList<Double> input) {
//        System.out.println(listOutput.size()+" "+ input.size());
        for (int i=0; i < listOutput.size(); i++) {
            for (int idxweight=0; idxweight < listOutput.get(i).getWeightSize(); idxweight++) {
                listOutput.get(i).setWeight(idxweight, listOutput.get(i).getWeightFromList(idxweight) + 
                        learningRate*listOutput.get(i).getError()*input.get(idxweight));
//                System.out.println(listOutput.get(i).getWeightFromList(idxweight));
            }
        }
    }
    private void output(ArrayList<Double> input, int idx) {
        double sigma = 0;
        for (int i=0; i < listOutput.get(idx).getWeightSize(); i++) {
            sigma += listOutput.get(idx).getWeightFromList(i)*input.get(i);
        }
        listOutput.get(idx).setValue(sigmoid(sigma));
    }
    
    private double sigmoid(double input) {
        return 1/(1+Math.pow(Math.E,input*(-1)));
    }
    
    private double maxValue() {
        double result=0, max = listOutput.get(0).getValue();
        for(int i =0; i < listOutput.size(); i++) {
            if (max < listOutput.get(i).getValue()) {
                result = i;
                max = listOutput.get(i).getValue();
            }
        }
        return result;
    }
    
    public static void main (String [] args) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(("D:\\Program Files\\Weka-3-8\\data\\Team.arff"));
        Instances train = source.getDataSet();
        Normalize nm = new Normalize();
        nm.setInputFormat(train);
        train = Filter.useFilter(train, nm);
        train.setClassIndex(train.numAttributes()-1);
        for (int i=100; i<3000; i+=100) {
            for (double j=0.01; j < 1; j+=0.01) {
                System.out.println(i + " "+ j);
                SinglelayerPerceptron slp = new SinglelayerPerceptron(i, j, 0.00);
                slp.buildClassifier(train);
                Evaluation eval = new Evaluation(train);
//                eval.crossValidateModel(slp, train,10, new Random(1));
                eval.evaluateModel(slp, train);
                System.out.println(eval.toSummaryString());
                System.out.println(eval.toMatrixString());
            }
        }
    }
}
