/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ANN;

import java.util.ArrayList;
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
    ArrayList<Node> listOutput;
    ArrayList<Node> listHidden;
    double learningRate;
    int itteration;
    private final double[] listDoubleinstance;
    public MultilayerPerceptron(Instances i, double rate, int itter, int numHidden) {
        learningRate = rate;
        listHidden = new ArrayList<>();
        
        for (int num =0; num<numHidden+1; num++) {
            listHidden.add(new Node(i.numAttributes()));
        }
        
        listOutput = new ArrayList<>();
        for (int num =0; num<i.numClasses(); num++) {
            listOutput.add(new Node(listHidden.size()));
        }
        itteration = itter;
        listDoubleinstance = new double[i.numInstances()];
        for (int numIns=0; numIns<i.numInstances();numIns++) {
            listDoubleinstance[numIns] = i.instance(numIns).toDoubleArray()[i.classIndex()];
        }
    }
    
    @Override 
    public void buildClassifier(Instances i) {
//       System.out.println(listOutput.get(0).getWeightSize() + " "+ listHidden.size());
       int cnt = 0;
        while (true) {//ulang iterasi
//            System.out.println();
//            System.out.println("iterasi "+itt);
            for (int idxInstance=0; idxInstance < i.numInstances(); idxInstance++){
                 //buat list input
                ArrayList<Double> listInput = new ArrayList<>();
                listInput.add(1.0);
                for (int idx=0; idx < i.numAttributes()-1; idx++) {
                    listInput.add(i.get(idxInstance).value(idx));
                }
                
                //hitung output hidden
                ArrayList<Double> hiddenOutput = new ArrayList<>();
                hiddenOutput.add(1.0);
                for (int idxOutput=1; idxOutput < listHidden.size(); idxOutput++) {
                    output(listHidden, listInput, idxOutput);
                    hiddenOutput.add(listHidden.get(idxOutput).getValue());
//                    System.out.println(outputVal);
                }
                //hitung output layer
                for (int idxOutput=0; idxOutput < listOutput.size(); idxOutput++) {
                    output(listOutput, hiddenOutput, idxOutput);
//                    System.out.println(outputVal);
                }
                
                //hitung error
                calculateError(idxInstance);
                //update bobot
                updateWeight(listInput);
            }
            double error = 0;
            for (int idxErr=0; idxErr < i.numInstances(); idxErr++) {
                for (int idx=0; idx < listOutput.size(); idx++) {
                    error += Math.pow(listOutput.get(idx).getError(), 2)/2;
//                    System.out.println(listOutput.get(idx).getError());
                }
//                System.out.println(error);
            }
            if(cnt == 1000) {
                System.out.println(error);
                System.out.println();
                cnt = 0;
            }
            cnt++;
            if (error <= 0.3) break;
        }
//        for (int idx=0;idx<listOutput.size();idx++) {
//            System.out.println("Output value "+listOutput.get(idx).getValue());
//            System.out.println("Output error "+listOutput.get(idx).getError());
//            for (int idx2=0; idx2<listOutput.get(idx).getWeightSize();idx2++)
//                System.out.println("Output weight"+listOutput.get(idx).getWeightFromList(idx2));
//        }
    }
    
    @Override
    public double classifyInstance(Instance i) {
        //buat list input
       ArrayList<Double> listInput = new ArrayList<>();
       listInput.add(1.0);
       for (int idx=0; idx < i.numAttributes()-1; idx++) {
           listInput.add(i.value(idx));
       }

       //hitung output layer
        ArrayList<Double> hiddenOutput = new ArrayList<>();
        hiddenOutput.add(1.0);
        for (int idxOutput=1; idxOutput < listHidden.size(); idxOutput++) {
            output(listHidden, listInput, idxOutput);
            hiddenOutput.add(listHidden.get(idxOutput).getValue());
//                        System.out.println(outputVal);
        }
        //hitung output layer
        for (int idxOutput=0; idxOutput < listOutput.size(); idxOutput++) {
            output(listOutput, hiddenOutput, idxOutput);
        }
       double result = maxIdxValue(listOutput);
       return result;
    }
    
    private void calculateError(int idxInstance) {
//        System.out.println(Arrays.toString(maxIdx));
        
        //hitung error output
        double[] target = targetToArray1Or0(idxInstance);
        for (int i=0; i < listOutput.size(); i++) {
            double result = listOutput.get(i).getValue()*(1-listOutput.get(i).getValue())*(target[i] - listOutput.get(i).getValue());
            listOutput.get(i).setError(result);
//            System.out.println(i+ " " +result);
//            System.out.print(target[i]+" - " + listOutput.get(i).getValue()+ " = "+result+" ");
        }
//        System.out.println();

        //hitung error hidden
        for (int i=0; i < listHidden.size(); i++) {
            double result = listHidden.get(i).getValue()*(1-listHidden.get(i).getValue());
//            System.out.println(result);
            double sigma = 0;
            
//                for (int k = 0; k < listOutput.size(); k++) {
//                    double error = listOutput.get(k).getError();
//                    sigma += error * listOutput.get(k).getWeightFromList(i);
//                }
            for (int j = 0; j < listOutput.size(); j++) {
                double error = listOutput.get(j).getError();
                sigma += error * listOutput.get(j).getWeightFromList(i);
//                System.out.println(sigma);
            }
//            System.out.println(result);
            listHidden.get(i).setError(result*sigma);
        }
    }
    
    private void updateWeight(ArrayList<Double> input) {
        //update weight hidden
        for (int idxOutput=0; idxOutput < listHidden.size(); idxOutput++) {
            for (int idxweight = 0; idxweight < listHidden.get(idxOutput).getWeightSize(); idxweight++) {
                double newWeight = listHidden.get(idxOutput).getWeightFromList(idxweight) + 
                        learningRate*listHidden.get(idxOutput).getError()*input.get(idxweight);
                listHidden.get(idxOutput).setWeight(idxweight, newWeight);
//                System.out.println(listHidden.get(idxOutput).getError());
            }
        }
        
        //update weight output
        for (int idxOutput=0; idxOutput < listOutput.size(); idxOutput++) {
            for (int idxweight = 0; idxweight < listOutput.get(idxOutput).getWeightSize(); idxweight++) {
                double newWeight = listOutput.get(idxOutput).getWeightFromList(idxweight) + 
                        learningRate*listOutput.get(idxOutput).getError()*listHidden.get(idxweight).getValue();
                listOutput.get(idxOutput).setWeight(idxweight, newWeight);
//                System.out.println(newWeight);
            }
//            System.out.println(listOutput.get(idxOutput).getError());
        }
    }
    
    public void output(ArrayList<Node> listNode, ArrayList<Double> listInput, int idx) {
        double sigma = 0;
//        System.out.println(listOutput.get(0).getWeightFromList(0));
        for (int ih = 0; ih < listNode.get(idx).getWeightSize(); ih++)  {
            System.out.println(listNode.get(idx).getWeightFromList(ih)+"*"+listInput.get(ih));
            sigma += listNode.get(idx).getWeightFromList(ih)*listInput.get(ih);//input value = 1, value
        }        
//        System.out.println(sigma+" "+sigmoid(sigma));
        listNode.get(idx).setValue(sigmoid (sigma)); 
    }
    
    private double sigmoid(double input) {
        return 1/(1+Math.pow(Math.E,input*(-1)));
    }
    
    private int maxIdxValue (ArrayList<Node> input) {
        double result = 0, max = input.get(0).getValue();
        for (int i=1; i < input.size(); i++) {
            if (max < input.get(i).getValue()) {
                result = i;
                max = input.get(i).getValue();
            }
        }
//        System.out.println(result);
        return (int)result;
    }
    
    private double[] targetToArray1Or0(int idxInstance) {
        double res[] = new double[listOutput.size()];
        for (int i=0;i<listOutput.size();i++) {
            if (i == listDoubleinstance[idxInstance])
                res[i] = 1;
            else res[i] = 0;
        }
        return res;
    }
    
    public static void main (String [] args) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(("D:\\Program Files\\Weka-3-8\\data\\iris.arff"));
        Instances train = source.getDataSet();
        Normalize nm = new Normalize();
        nm.setInputFormat(train);
        train = Filter.useFilter(train, nm);
        train.setClassIndex(train.numAttributes()-1);
                System.out.println();
//                System.out.println(i + " "+0.8);
                MultilayerPerceptron slp = new MultilayerPerceptron(train, 0.1, 5000, 14);
                slp.buildClassifier(train);
                Evaluation eval = new Evaluation(train);
                eval.evaluateModel(slp, train);
                System.out.println(eval.toSummaryString());
                System.out.print(eval.toMatrixString());
    }
}
