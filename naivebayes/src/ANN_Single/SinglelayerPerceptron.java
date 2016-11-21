/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ANN_Single;

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
public class SinglelayerPerceptron extends AbstractClassifier {
    ArrayList<Node> listOutput;
    double learningRate;
    int itteration;
    private final double[] listDoubleinstance;
    private static int fold =0;
    public SinglelayerPerceptron(Instances i, double rate, int itter) {
        learningRate = rate;
//        listOutput = new ArrayList<>();
//        for (int num =0; num<i.numClasses(); num++) {
//            listOutput.add(new Node(i.numAttributes()));
//        }
        itteration = itter;
        listDoubleinstance = new double[i.numInstances()];
        for (int numIns=0; numIns<i.numInstances();numIns++) {
            listDoubleinstance[numIns] = i.instance(numIns).toDoubleArray()[i.classIndex()];
        }
    }
    
    @Override 
    public void buildClassifier(Instances i) {
       listOutput = new ArrayList<>();
        for (int num =0; num<i.numClasses(); num++) {
            listOutput.add(new Node(i.numAttributes()));
        }
        while (true) {//ulang iterasi
//            System.out.println();
//            System.out.println("iterasi "+itt);
            for (int idxInstance=0; idxInstance < i.numInstances(); idxInstance++){
                 //buat list input
//                 System.out.print(idxInstance+" ");
                ArrayList<Double> listInput = new ArrayList<>();
                listInput.add(1.0);
                for (int idx=0; idx < i.numAttributes()-1; idx++) {
                    listInput.add(i.get(idxInstance).value(idx));
                }
                
                //hitung output layer
                for (int idxOutput=0; idxOutput < listOutput.size(); idxOutput++) {
                    output(listInput, idxOutput);
//                    listOutput.get(idxOutput).setValue(outputVal);
//                    System.out.print(listOutput.get(idxOutput).getValue()+" ");
                }
//                System.out.println();
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
            System.out.println(error);
            System.out.println();
            if (error <= 0) break;
        }
        fold++;
        System.out.println("Fold ke-"+fold);
        double error = 0;
        for (int idxErr=0; idxErr < i.numInstances(); idxErr++) {
           for (Node listOutput1 : listOutput) {
                error += Math.pow(listOutput1.getError(), 2)/2;
//                    System.out.println(listOutput1.getError());
           }
//                System.out.println(error);
        }
        System.out.println("error "+error);
        for (int idx=0;idx<listOutput.size();idx++) {
            System.out.println("Output value "+listOutput.get(idx).getValue());
            System.out.println("Output error "+listOutput.get(idx).getError());
            for (int idx2=0; idx2<listOutput.get(idx).getWeightSize();idx2++)
                System.out.println("Output weight"+listOutput.get(idx).getWeightFromList(idx2));
        }
    }
    
    @Override
    public double classifyInstance(Instance i) {
        //buat list input
        System.out.println("classify");
       ArrayList<Double> listInput = new ArrayList<>();
       listInput.add(1.0);
       for (int idx=0; idx < i.numAttributes()-1; idx++) {
           listInput.add(i.value(idx));
       }

       //hitung output layer
       for (int idxOutput=0; idxOutput < listOutput.size(); idxOutput++) {
           outputC(listInput, idxOutput);
//           listOutput.get(idxOutput).output(listInput);
//           listOutput.get(idxOutput).setValue(outputVal);
//           System.out.println(idxOutput+" "+outputVal);
//            System.out.print(listOutput.get(idxOutput).getValue()+" ");
       }
       System.out.println();
       double result = maxIdxValue(listOutput);
//       System.out.println(result+" "+i.stringValue(i.classIndex()));
        System.out.println("end c");
       return result;
    }
    
    private void calculateError(int idxInstance) {
        double[] maxIdx = maxValueToArray(listOutput);
//        System.out.println("output "+Arrays.toString(maxIdx));
        double[] target = targetToArray1Or0(idxInstance);
//        System.out.println("Target "+Arrays.toString(target));
//        int beda = 0;
        for (int i=0; i < listOutput.size(); i++) {
            double result = target[i] - maxIdx[i];
//            if (result != 0) beda++;
            listOutput.get(i).setError(result);
//            System.out.println(result);
        }
//        System.out.println(beda);
    }
    
    private void updateWeight(ArrayList<Double> input) {
        for (int idxOutput=0; idxOutput < listOutput.size(); idxOutput++) {
            for (int idxweight = 0; idxweight < listOutput.get(idxOutput).getWeightSize(); idxweight++) {
                double newWeight = listOutput.get(idxOutput).getWeightFromList(idxweight) + 
                        learningRate*listOutput.get(idxOutput).getError()*input.get(idxweight);
                listOutput.get(idxOutput).setWeight(idxweight, newWeight);
//                System.out.println(newWeight+ " " +listOutput.get(idxOutput).getWeightFromList(idxweight));
            }
        }
    }
    
    public void output(ArrayList<Double> listInput, int idx) {
        double sigma = 0;
//        System.out.println(listOutput.get(0).getWeightFromList(0));
        for (int ih = 0; ih < listOutput.get(idx).getWeightSize(); ih++)  {
//            System.out.println(listOutput.get(idx).getWeightFromList(ih)+"*"+listInput.get(ih));
            sigma += listOutput.get(idx).getWeightFromList(ih)*listInput.get(ih);//input value = 1, value
        }        
//        System.out.println(sigma+" "+sigmoid(sigma));
        listOutput.get(idx).setValue(sigmoid (sigma)); 
    }
    
    public void outputC(ArrayList<Double> listInput, int idx) {
        double sigma = 0;
//        System.out.println(listOutput.get(0).getWeightFromList(0));
        for (int ih = 0; ih < listOutput.get(idx).getWeightSize(); ih++)  {
            System.out.println(listOutput.get(idx).getWeightFromList(ih)+"*"+listInput.get(ih));
            sigma += listOutput.get(idx).getWeightFromList(ih)*listInput.get(ih);//input value = 1, value
        }        
        System.out.println(sigma+" "+sigmoid(sigma));
        listOutput.get(idx).setValue(sigmoid (sigma)); 
    }
    
    
    private double sigmoid(double input) {
        return 1/(1+Math.pow(Math.E,input*(-1)));
    }
    
    private double[] maxValueToArray(ArrayList<Node> input) {
        double [] result = new double[input.size()];
        for (int i=0; i < listOutput.size(); i++) {
            if (i == maxIdxValue(input)) {
                result[i] = 1;
            }else result[i] = 0;
        }
        return result;
    }
    
    private int maxIdxValue (ArrayList<Node> input) {
        double result = 0, max = input.get(0).getValue();
        for (int i=1; i < input.size(); i++) {
            if (max < input.get(i).getValue()) {
                result = i;
                max = input.get(i).getValue();
            }
//            System.out.println(input.get(i).getValue());
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
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(("D:\\Program Files\\Weka-3-8\\data\\diabetes.arff"));
        Instances train = source.getDataSet();
        Normalize nm = new Normalize();
        nm.setInputFormat(train);
        train = Filter.useFilter(train, nm);
        train.setClassIndex(train.numAttributes()-1);
                System.out.println();
//                System.out.println(i + " "+0.8);
                SinglelayerPerceptron slp = new SinglelayerPerceptron(train, 0.1, 5000);
                slp.buildClassifier(train);
                Evaluation eval = new Evaluation(train);
//                eval.crossValidateModel(slp, train, 10, new Random(1));
                eval.evaluateModel(slp, train);
                System.out.println(eval.toSummaryString());
                System.out.print(eval.toMatrixString());
    }
}
