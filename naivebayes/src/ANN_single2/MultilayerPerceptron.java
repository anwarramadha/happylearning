/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ANN_single2;

import java.util.ArrayList;
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
public class MultilayerPerceptron extends AbstractClassifier {
    ArrayList<Node> listHidden;
    ArrayList<Node> listOutput;
    double learningRate;
    double threshold;
    int numHiden;
    double[] listDoubleinstance;
    static int fold = 0;
    
    public MultilayerPerceptron(Instances i, int numHide, double rate, double thres) {
        learningRate = rate;
        threshold = thres;
        numHiden = numHide;
        //inisialisasi array hidden
        listHidden = new ArrayList<>();
        for (int idx=0; idx < numHiden; idx++) {
            listHidden.add(new Node(i.numAttributes())); //1 untuk bias
        }
        
        //inialisasi array output
        listOutput = new ArrayList<>();
        for (int idx=0; idx < i.numClasses(); idx++) {
            listOutput.add(new Node(listHidden.size()));
        }
    }
    
    @Override 
    public void buildClassifier(Instances i) {
        
        //mengubah class menjadi numeric (diambil indexnya)
        listDoubleinstance = new double[i.numInstances()];
        for (int numIns=0; numIns<i.numInstances();numIns++) {
            listDoubleinstance[numIns] = i.instance(numIns).toDoubleArray()[i.classIndex()];
        }
        int cnt=0;
        for (int itt =0; itt < 10000; itt ++) {
            for (int idxInstance=0; idxInstance < i.numInstances(); idxInstance++) {
                //buat list input
                ArrayList<Double> listInput = new ArrayList<>();
                listInput.add(1.0); //ini untuk bias
                for(int ins=0; ins < i.get(idxInstance).numAttributes()-1; ins++) {
                    listInput.add(i.get(idxInstance).value(ins));
                }
                
                ArrayList<Double> listHide = new ArrayList<>();
                listHide.add(1.0);
                //Hitung output hidden layer
                for (int idxHidden =1; idxHidden < listHidden.size(); idxHidden++) {
                    output(listHidden, listInput, idxHidden);
                    listHide.add(listHidden.get(idxHidden).getValue());
                }
                
                //Hitung ouput output lyer
                for (int idxOutput=0; idxOutput < listOutput.size(); idxOutput++) {
                    output(listOutput, listHide, idxOutput);
                }
                
                //Hitung error
                calculateError(idxInstance);
                //update bobot
                updateBobot(listInput);
            }
            //Hitung seluruh error untuk menentukan kapan harus berhenti
//            double error = 0;
//            for (int idx =0; idx < i.numInstances(); idx++) {
//                for (int idxOut=0; idxOut < listOutput.size(); idxOut++) {
//                    error += Math.pow(listOutput.get(idxOut).getError(), 2)/2;
//                }
//            }
//            cnt++;
//            if (cnt==1000) {
//                System.out.println("error " + error);
//                cnt=0;
//            }
//            if (error <= threshold) break;
        }
        double error = 0;
        fold++;
        for (int idx =0; idx < i.numInstances(); idx++) {
            for (int idxOut=0; idxOut < listOutput.size(); idxOut++) {
                error += Math.pow(listOutput.get(idxOut).getError(), 2)/2;
            }
        }
        System.out.println("Fold " + fold);
        System.out.println("error " + error);
    
    }
    
    @Override
    public double classifyInstance(Instance i) {
        ArrayList<Double> listInput = new ArrayList<>();
        listInput.add(1.0); //ini untuk bias
        for(int ins=0; ins < i.numAttributes()-1; ins++) {
            listInput.add(i.value(ins));
        }

        ArrayList<Double> listHide = new ArrayList<>();
        listHide.add(1.0);
        //Hitung output hidden layer
        for (int idxHidden =1; idxHidden < listHidden.size(); idxHidden++) {
            output(listHidden, listInput, idxHidden);
            listHide.add(listHidden.get(idxHidden).getValue());
        }

        //Hitung ouput output lyer
        for (int idxOutput=0; idxOutput < listOutput.size(); idxOutput++) {
            output(listOutput, listHide, idxOutput);
        }
        return maxValue();
    }
    private void calculateError(int idx) {
        //Hitung error output layer
        double [] target = target(idx);
        double [] output = outputTo1Or0(idx);
        for (int i=0; i < listOutput.size(); i++) {
            double error = listOutput.get(i).getValue() * (1-listOutput.get(i).getValue()) *
                    (target[i] - output[i]);
            listOutput.get(i).setError(error);
//            System.out.println(i+" "+target[i] + " - "+output[i]+ " " +listOutput.get(i).getError());
        }
        
        //Hitung error hidden
        for (int i =0; i < listHidden.size(); i++) {
            double error = listHidden.get(i).getValue() * (1-listHidden.get(i).getValue());
            double sigma=0;
            for (int j = 0; j < listOutput.size(); j ++) {
                sigma += listOutput.get(j).getError() * listOutput.get(j).getWeightFromList(i);
            }
            listHidden.get(i).setError(sigma*error);
        }
    }
    
    private void updateBobot(ArrayList<Double> input) {
        //update bobot hidden
        for (int i=0; i < listHidden.size(); i++) {
            for (int j=0; j < listHidden.get(i).getWeightSize(); j++) {
                double newWeight = listHidden.get(i).getWeightFromList(j) + 
                        learningRate*listHidden.get(i).getError()*input.get(j);
                listHidden.get(i).setWeight(j, newWeight);
            }
        }
        
        //update bobot output
        for (int i=0; i < listOutput.size(); i++) {
            for (int j=0; j < listOutput.get(i).getWeightSize(); j++) {
                double newWeight = listOutput.get(i).getWeightFromList(j) + 
                        learningRate*listOutput.get(i).getError()*listHidden.get(j).getValue();
                listOutput.get(i).setWeight(j, newWeight);
            }
        }
    }
    
    public double[] outputTo1Or0(int idx) {
        double [] result = new double[listOutput.size()];
        for (int i=0; i < listOutput.size(); i++) {
            if (i == maxValue()) {
                result[i] = 1;
            }
            else result[i] = 0;
        }
        return result;
    }
    
    private void output(ArrayList<Node> listNode, ArrayList<Double> input, int idx) {
        double sigma = 0;
        for (int i=0; i < listNode.get(idx).getWeightSize(); i++) {
            sigma += listNode.get(idx).getWeightFromList(i)*input.get(i);
        }
        listNode.get(idx).setValue(sigmoid(sigma));
    }
    
    private double sigmoid(double input) {
        return 1/(1+Math.pow(Math.E,input*(-1)));
    }
    
    private double[] target(int idx) {
        double [] result = new double[listOutput.size()];
        for (int i = 0; i < listOutput.size(); i++) {
            if (i == listDoubleinstance[idx]) {
                result[i] = 1;
            }else result[i] = 0;
        }
        return result;
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
    
    public static void main (String[] args) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(("D:\\Program Files\\Weka-3-8\\data\\Team.arff"));
        Instances train = source.getDataSet();
        Normalize nm = new Normalize();
        nm.setInputFormat(train);
        train = Filter.useFilter(train, nm);
        train.setClassIndex(train.numAttributes()-1);
        MultilayerPerceptron slp = new MultilayerPerceptron(train, 13, 0.1, 0.5);
//        slp.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
                eval.crossValidateModel(slp, train,10, new Random(1));
//        eval.evaluateModel(slp, train);
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
    }
}
