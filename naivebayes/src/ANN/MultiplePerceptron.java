/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ANN;

import ANN.Node;
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
public class MultiplePerceptron extends AbstractClassifier{
    private ArrayList<Node> listNodeHidden;
    private ArrayList<Node> listNodeOutput;
    private final double learningRate;
    private final double[] listDoubleinstance;
    
    public MultiplePerceptron(Instances i, int numNode, double rate) {
        listNodeHidden = new ArrayList<>();
        for (int num=0;num<numNode+1;num++) {
            listNodeHidden.add(new Node(i.numAttributes()));
        }
        
        listNodeOutput = new ArrayList<>();
        for (int num=0;num<i.numClasses();num++) {
            listNodeOutput.add(new Node(listNodeHidden.size()));
        }
        
        listDoubleinstance = new double[i.numInstances()];
        for (int numIns=0; numIns<i.numInstances();numIns++) {
            listDoubleinstance[numIns] = i.instance(numIns).toDoubleArray()[i.classIndex()];
        }
        learningRate = rate;
    }
    
    @Override
    public void buildClassifier(Instances i) {
//        System.out.println(listNodeHidden.get(0).getWeightSize()+" "+listNodeOutput.get(0).getWeightSize());
        for (int itt = 0;itt<5000;itt++) {
            for (int idxInstance=0; idxInstance<i.numInstances();idxInstance++) {
                ArrayList<Double> listInput = new ArrayList<>();
                listInput.add(1.0);
                for(int idxInstanceVal=0; idxInstanceVal<i.numAttributes()-1;idxInstanceVal++) {
                    listInput.add(i.get(idxInstance).value(idxInstanceVal));
                }

                ArrayList<Double> listOutputHidden = new ArrayList<>();
                listOutputHidden.add(1.0);
                
                //set output hidden layer
//                System.out.println("Hidden layer\n");
                for(int idxNodeHidden=1; idxNodeHidden<listNodeHidden.size();idxNodeHidden++) {
                    double outputVal = listNodeHidden.get(idxNodeHidden).output(listInput);
                    listNodeHidden.get(idxNodeHidden).setValue(outputVal);
                    listOutputHidden.add(outputVal);
//                    System.out.println(outputVal);
                }
                
//                System.out.println("Output layer\n");
                //set output layer
                for(int idxNodeHidden=0; idxNodeHidden<listNodeOutput.size();idxNodeHidden++) {
                    double outputVal = listNodeOutput.get(idxNodeHidden).output(listOutputHidden);
                    listNodeOutput.get(idxNodeHidden).setValue(outputVal);
//                    System.out.println(outputVal);
                }

                //calculate error (back propagation)
                calculateError(idxInstance);
                //re-calculate weight
                calculateWeight(i.instance(idxInstance));
            }
        }
        for (int idx=0;idx<listNodeHidden.size();idx++) {
                System.out.println("Hidden value "+listNodeHidden.get(idx).getValue());
                System.out.println("Hidden error "+listNodeHidden.get(idx).getError());
                for (int idx2=0; idx2<listNodeHidden.get(idx).getWeightSize();idx2++)
                    System.out.println("Hidden weight"+listNodeHidden.get(idx).getWeightFromList(idx2));
            }
            System.out.println();
            for (int idx=0;idx<listNodeOutput.size();idx++) {
                System.out.println("Output value "+listNodeOutput.get(idx).getValue());
                System.out.println("Output error "+listNodeOutput.get(idx).getError());
                for (int idx2=0; idx2<listNodeOutput.get(idx).getWeightSize();idx2++)
                    System.out.println("Output weight"+listNodeOutput.get(idx).getWeightFromList(idx2));
            }
    }
    
    @Override
    public double classifyInstance(Instance i) {
        ArrayList<Double> listInput = new ArrayList<>();
        listInput.add(1.0);
        for(int idxInstanceVal=0; idxInstanceVal<i.numAttributes()-1;idxInstanceVal++) {
            listInput.add(i.value(idxInstanceVal));
        }

        ArrayList<Double> listOutputHidden = new ArrayList<>();
        listOutputHidden.add(1.0);
        //set output hidden layer
        for(int idxNodeHidden=1; idxNodeHidden<listNodeHidden.size();idxNodeHidden++) {
            double outputVal = listNodeHidden.get(idxNodeHidden).output(listInput);
            listNodeHidden.get(idxNodeHidden).setValue(outputVal);
            listOutputHidden.add(outputVal);
        }

        //set output layer
        for(int idxNodeHidden=0; idxNodeHidden<listNodeOutput.size();idxNodeHidden++) {
            double outputVal = listNodeOutput.get(idxNodeHidden).output(listOutputHidden);
            listNodeOutput.get(idxNodeHidden).setValue(outputVal);
//            System.out.printf("%f ", outputVal);
        }
//        System.out.println();
        return getIdxMax(listNodeOutput);
    }
    public void calculateError(int idxInstance) {
        //error output
        double []  output = maxValueToArray();
        double [] target = targetToArray1Or0(idxInstance);
//        System.out.println(listNodeOutput.size());
        for (int idxOutputNode=0; idxOutputNode<listNodeOutput.size();idxOutputNode++) {
            double error = listNodeOutput.get(idxOutputNode).getValue()*
                    (1-listNodeOutput.get(idxOutputNode).getValue());//dari fungsi sigmoid
//            System.out.println(error);
            double delta = target[idxOutputNode] - listNodeOutput.get(idxOutputNode).getError();
            error *= delta;
            listNodeOutput.get(idxOutputNode).setError(error);
        }
        
        //set error hidden input
        for (int idxHiddenNode=0;idxHiddenNode<listNodeHidden.size(); idxHiddenNode++) {
             double error = listNodeHidden.get(idxHiddenNode).getValue()*
                    (1-listNodeHidden.get(idxHiddenNode).getValue());//dari fungsi sigmoid
             double sum = 0;
             for (int idxOutputNode=0; idxOutputNode<listNodeOutput.size();idxOutputNode++) {
                 sum += listNodeOutput.get(idxOutputNode).getError()*
                         listNodeOutput.get(idxOutputNode).getWeightFromList(idxHiddenNode);
//                 System.out.println(listNodeOutput.get(idxOutputNode).getWeightFromList(idxHiddenNode));
             }
             error *= sum;
             listNodeHidden.get(idxHiddenNode).setError(error);
        }
    }
    
    public void calculateWeight(Instance i) {
        
        ArrayList<Double> listInput = new ArrayList<>();
        listInput.add(1.0);
        for(int idxInstanceVal=0; idxInstanceVal<i.numAttributes();idxInstanceVal++) {
            listInput.add(i.value(idxInstanceVal));
        }
        
        //set weight hidden
//        for (int index = 0; index < listNodeHidden.size();index++) {
//            for (int indexDalem = 0 ; indexDalem < listNodeHidden.get(index).getWeightSize();indexDalem++) {
//                double delta = learningRate*listNodeHidden.get(index).getError()*listInput.get(indexDalem);
//                double newWeight = delta + listNodeHidden.get(index).getWeightFromList(indexDalem);
//                listNodeHidden.get(index).setWeight(indexDalem, newWeight);
////                System.out.println(index+" "+indexDalem+" "+newWeight);
//            }
//        }
        for (int idxHidden=0; idxHidden < listNodeHidden.size(); idxHidden++) {
            for (int idxweight=0;idxweight<listNodeHidden.get(idxHidden).getWeightSize();idxweight++) {
                double oldVal = listNodeHidden.get(idxHidden).getWeightFromList(idxweight);
//                System.out.println(oldVal);
                oldVal += learningRate*listInput.get(idxweight) * listNodeHidden.get(idxHidden).getError();
                listNodeHidden.get(idxHidden).setWeight(idxweight, oldVal);
//                System.out.println(listNodeHidden.get(idxHidden).getWeightFromList(idxweight));
            }
        }
//        System.out.println(listNodeOutput.get(0).getWeightSize()+" "+listOutputHidden);
        //set weight output
        for (int index = 0; index < listNodeOutput.size();index++) {
            for (int indexDalem = 0 ; indexDalem < listNodeHidden.size();indexDalem++) {
                double delta = learningRate*listNodeOutput.get(index).getError()*listNodeHidden.get(indexDalem).getValue();
                double newWeight = delta + listNodeOutput.get(index).getWeightFromList(indexDalem);
                listNodeOutput.get(index).setWeight(indexDalem, newWeight);
//                System.out.println(newWeight);
            }
        }
//        for (int idxOutput=0; idxOutput < listNodeOutput.size(); idxOutput++) {
//            for (int idxweight=0;idxweight<listNodeOutput.get(idxOutput).getWeightSize();idxweight++) {
//                double oldVal = listNodeOutput.get(idxOutput).getWeightFromList(idxweight);
//                oldVal += learningRate*listOutputHidden.get(idxweight) * listNodeOutput.get(idxOutput).getError();
//                listNodeOutput.get(idxOutput).setWeight(idxweight, oldVal);
//            }
//        }
            
    }
    
    private double getIdxMax(ArrayList<Node> in) {
        double max = in.get(0).getValue(), idx=0;
        for (int i = 0; i < in.size(); i++) {
            if (max<in.get(i).getValue()) {
                max = in.get(i).getValue();
                idx = i;
            }
        }
        return idx;
    }
    
    private double[] maxValueToArray() {
        double maxTemp = 0, max= listNodeOutput.get(0).getValue(), num = listNodeOutput.size();
        double[] result = new double[(int)num];
        for (int i=1; i < listNodeOutput.size(); i++) {
            if (max <listNodeOutput.get(i).getValue()) {
                maxTemp = i;
                max = listNodeOutput.get(i).getValue();
            }
        }
        for (int i=0; i < listNodeOutput.size(); i++) {
            if (i == maxTemp) {
                result[i] = 1;
            }else result[i] = 0;
        }
        return result;
    }
    
    private double[] targetToArray1Or0(int idxInstance) {
        double res[] = new double[listNodeOutput.size()];
        for (int i=0;i<listNodeOutput.size();i++) {
            if (i == listDoubleinstance[idxInstance])
                res[i] = 1;
            else res[i] = 0;
        }
        return res;
    }
    
    public static void main(String [] args) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(("D:\\Program Files\\Weka-3-8\\data\\iris.arff"));
        Instances train = source.getDataSet();
        Normalize nm = new Normalize();
        nm.setInputFormat(train);
        train = Filter.useFilter(train, nm);
        train.setClassIndex(train.numAttributes()-1);
        MultiplePerceptron mlp = new MultiplePerceptron(train, 20, 0.3);
        mlp.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(mlp, train);
        System.out.println(eval.toSummaryString());
        System.out.print(eval.toMatrixString());
    }
}
