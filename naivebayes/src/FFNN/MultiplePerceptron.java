/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package FFNN;

import java.util.ArrayList;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

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
    ArrayList<Integer> target;
    double [] instancesToDouble;
    public MultiplePerceptron(int itt, double learn, int numHLayer, Instances i) {
        listNodeHidden = new ArrayList<>();//inisialisasis listNodeHidden
        listNodeOutput = new ArrayList<>();
        itteration = itt;
        learningRate = learn;
        numHiddenLayer = numHLayer;
        for (int hiddenLayer=0; hiddenLayer<numHiddenLayer+1; hiddenLayer++) {//buat neuron untuk hidden layer
            //ditambah 1 untuk neuron bias
            listNodeHidden.add(new Node(i.numAttributes()));
            
        }
        
        for (int numInstance = 0; numInstance<i.numClasses(); numInstance++) {//buat neuron untuk output
            listNodeOutput.add(new Node(numHiddenLayer));
        }
        target = new ArrayList<>();
        instancesToDouble = new double[i.numInstances()];
        for (int numIns=0; numIns<i.numInstances();numIns++) {
            instancesToDouble[numIns] = i.instance(numIns).toDoubleArray()[i.classIndex()];
        }
    }
    
    @Override 
    public void buildClassifier(Instances i){
        //iterasi
        for (int itt = 0;itt<itteration;itt++) {
//            System.out.println("Iterasi ke "+ itt);
            for (int indexInstance = 0; indexInstance < i.numInstances();indexInstance++) {
                ArrayList<Double> listInput = new ArrayList<>();

                //mengisi nilai listInput dengan nilai di instances
                listInput.add(1.0);//ini bias input
                for (int index = 0 ; index < i.numAttributes()-1;index++)
                    listInput.add(i.get(indexInstance).value(index));

                ArrayList<Double> listOutputHidden = new ArrayList<>();

                //menghitung output hidden layer
                for (int index = 0; index < listNodeHidden.size();index++) {//output bias tidak boleh ganti
                    double value = listNodeHidden.get(index).output(listInput);
                    listOutputHidden.add(value);
                    if (index!=0)
                        listNodeHidden.get(index).setValue(value);
                }


                //menghitung output output layer
                for (int index = 0; index < listNodeOutput.size();index++) {
                    double value = listNodeOutput.get(index).output(listOutputHidden); 
//                    System.out.println(value);
                    listNodeOutput.get(index).setValue(value);
                    
                }
                
                calculateError(i.instance(indexInstance), indexInstance);

                updateBobot(i.instance(indexInstance));
            }
        }
    }
    
    public void updateBobot(Instance i) {
        
        ArrayList<Double> listInput = new ArrayList<>(); 
        
        //mengisi nilai listInput dengan nilai di instances
        listInput.add(1.0);
        for (int index = 0 ; index < i.numAttributes()-1;index++)
            listInput.add(i.value(index));
        
        //bobot hidden
        for (int index = 0; index < listNodeHidden.size();index++) {
            for (int indexDalem = 0 ; indexDalem < listInput.size();indexDalem++) {
                double delta = learningRate*listNodeHidden.get(index).getError()*listInput.get(indexDalem);
                double newWeight = delta + listNodeHidden.get(index).getWeightFromList(indexDalem);
                listNodeHidden.get(index).setWeight(indexDalem, newWeight);
            }
        }
        
        //bobot output
        for (int index = 0; index < listNodeOutput.size();index++) {
            for (int indexDalem = 0 ; indexDalem < listNodeHidden.size();indexDalem++) {
                double delta = learningRate*listNodeOutput.get(index).getError()*listNodeHidden.get(indexDalem).getValue();
                double newWeight = delta + listNodeOutput.get(index).getWeightFromList(indexDalem);
                listNodeOutput.get(index).setWeight(indexDalem, newWeight);
            }
        }
        
        
    }
    
    public void calculateError(Instance i, int insIdx) {
        
        i.attribute(i.classIndex());
        for (int index = 0 ; index < listNodeOutput.size() ; index++) {
            double outputVal = listNodeOutput.get(index).getValue();
            //System.out.println("real "+outputVal+"expect "+realVal);
            double errorVal;
            if ((double)index == getTargetValue(insIdx))//cek jika index == target, maka nilai target = 1, else 0
                errorVal=outputVal*(1-outputVal)*(1-outputVal);
            else errorVal=outputVal*(1-outputVal)*(0-outputVal); //nilai target = 0
            listNodeOutput.get(index).setError(errorVal);
//            System.out.printf("%.3f %.3f ", outputVal, errorVal);
        }
//        System.out.println();
        //set error layer hidden
        for (int index = 0; index < listNodeHidden.size(); index++) {
            double outputVal = listNodeHidden.get(index).getValue();
            double errorVal = outputVal*(1-outputVal);
            double sigma = 0;
            for (int indexDalem = 0; indexDalem < listNodeOutput.size();indexDalem++){
                sigma += listNodeOutput.get(indexDalem).getWeightFromList(index) * listNodeOutput.get(indexDalem).getError();
            }
            errorVal *= sigma;
            listNodeHidden.get(index).setError(errorVal);
        }
        
        //beres
    }
    
    private double getTargetValue(int index) {
        return instancesToDouble[index];
    }
    
    private double maxValue(ArrayList<Node> in) {
        double result=0, max=in.get(0).getValue();
        for (int idx = 0; idx<in.size();idx++) {
            if (max<in.get(idx).getValue()) {
                max = in.get(idx).getValue();
                result=(double)idx;
            }
        }
//        System.out.println(result);
        return result;
    }
    
    @Override 
    public double classifyInstance(Instance i) {
        ArrayList<Double> listInput = new ArrayList<>();

        //mengisi nilai listInput dengan nilai di instances
        listInput.add(1.0);
        for (int index = 0 ; index < i.numAttributes()-1;index++)
            listInput.add(i.value(index));

        ArrayList<Double> listOutputHidden = new ArrayList<>();

        //menghitung output hidden layer
        for (int index = 0; index < listNodeHidden.size();index++) {//dari 1 karena node 0 ada bias
            double value = listNodeHidden.get(index).output(listInput);
            listOutputHidden.add(value);
            if (index!=0)
                listNodeHidden.get(index).setValue(value);
        }


        //menghitung output output layer
        for (int index = 0; index < listNodeOutput.size();index++) {
            double value = listNodeOutput.get(index).output(listOutputHidden); 
            listNodeOutput.get(index).setValue(value);

        }
        
        return maxValue(listNodeOutput);
    }
    
    public static void main(String args[]) throws Exception{
//        System.out.println("input jumlah layer 0/1 :");
//        Scanner input = new Scanner(System.in);
//        int layer = input.nextInt();
//        System.out.println("input learning rate");
//        double rate = input.nextDouble();
//        int hidden = 0;
//        if(layer==1){
//            System.out.println("input jumlah neuron di hidden layer");
//            hidden = input.nextInt();
//        }
//        
//        System.out.print("Masukkan nama file : ");
//        String filename = input.next();
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(("D:\\Program Files\\Weka-3-8\\data\\iris.arff"));
        Instances train = source.getDataSet();
//        Normalize nm = new Normalize();
//        nm.setInputFormat(train);
//        train = Filter.useFilter(train, nm);
        for (int i=0; i < train.numAttributes();i++)
            System.out.println(i+". "+train.attribute(i).name());
        System.out.print("Masukkan indeks kelas : ");
        //int classIdx = input.nextInt();
        train.setClassIndex(train.numAttributes()-1);
        MultiplePerceptron mlp = new MultiplePerceptron(10000, 0.005 , 50, train);
        mlp.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(mlp, train);
        System.out.println(eval.toSummaryString());
    }
}
