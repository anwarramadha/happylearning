/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package FFNN;

import java.util.ArrayList;
import java.util.Scanner;
import weka.classifiers.AbstractClassifier;
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
    
    public MultiplePerceptron(int itt, double learn, int numHLayer, Instances i) {
        listNodeHidden = new ArrayList<>();//inisialisasis listNodeHidden
        listNodeOutput = new ArrayList<>();
        itteration = itt;
        learningRate = learn;
        numHiddenLayer = numHLayer;
        for (int hiddenLayer=0; hiddenLayer<numHiddenLayer; hiddenLayer++) {//buat neuron untuk hidden layer
            listNodeHidden.add(new Node(i.numAttributes()));
            
        }
        
        for (int numInstance = 0; numInstance<i.numClasses(); numInstance++) {//buat neuron untuk output
            listNodeOutput.add(new Node(numHiddenLayer));
        }
        target = new ArrayList<>();
    }
    
    @Override 
    public void buildClassifier(Instances i){
        //iterasi
        for (int indexInstance = 0; indexInstance < i.numInstances();indexInstance++) {
            ArrayList<Double> listInput = new ArrayList<>();
            
            //mengisi nilai listInput dengan nilai di instances
            for (int index = 0 ; index < i.numAttributes();index++)
                listInput.add(i.get(indexInstance).value(index));
            
            ArrayList<Double> listOutputHidden = new ArrayList<>();
            
            //menghitung output hidden layer
            for (int index = 0; index < listNodeHidden.size();index++) {
                double value = listNodeHidden.get(index).output(listInput);
                listOutputHidden.add(value);
                listNodeHidden.get(index).setValue(value);
                System.out.println("Hidden layer outpunt" + value);
            }
            
            
            ArrayList<Double> listOutputOutput = new ArrayList<>();
            
            //menghitung output output layer
            for (int index = 0; index < listNodeOutput.size();index++) {
                double value = listNodeOutput.get(index).output(listOutputHidden);
                listOutputOutput.add(value);   
                listNodeOutput.get(index).setValue(value);
            }
            
            calculateError(i.instance(indexInstance));
            
            updateBobot(i.instance(indexInstance));
            
            
            
            
            
            
            
        }
        
        

    }
    
    public void updateBobot(Instance i) {
        
        ArrayList<Double> listInput = new ArrayList<>(); 
        
        //mengisi nilai listInput dengan nilai di instances
        for (int index = 0 ; index < i.numAttributes();index++)
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
                double delta = learningRate*listNodeOutput.get(0).getError()*listNodeHidden.get(0).getValue();
                double newWeight = delta + listNodeOutput.get(index).getWeightFromList(indexDalem);
                listNodeOutput.get(index).setWeight(indexDalem, newWeight);
            }
        }
        
        
    }
    
    public void calculateError(Instance i) {
        
        i.attribute(i.classIndex());
        
        //ceritanya ngisi target (BELUM BERES)
        for (int index = 0; index < i.numClasses() ; index++)
            target.add(1);
        
        //set error layer output
        for (int index = 0 ; index < listNodeOutput.size() ; index++) {
            double outputVal = listNodeOutput.get(index).getValue();
            double errorVal = outputVal*(1-outputVal)*(target.get(index)-outputVal);
            listNodeOutput.get(index).setError(errorVal);
        }
        
        //set error layer hidden
        for (int index = 0; index < listNodeHidden.size(); index++) {
            double outputVal = listNodeHidden.get(index).getValue();
            double errorVal = outputVal*(1-outputVal);
            double sigma = 0;
            for (int indexDalem = 0 ; indexDalem < listNodeOutput.get(index).getWeightSize(); indexDalem++)
                sigma += listNodeOutput.get(index).getWeightFromList(indexDalem)*listNodeOutput.get(index).getError();
            
            errorVal *= sigma;
            
            listNodeHidden.get(index).setError(errorVal);
        }
        
        //beres
    }
    
    
    
    @Override 
    public double classifyInstance(Instance i) {
        return 0;
    }
    
    public static void main(String args[]) throws Exception{
        System.out.println("input jumlah layer 0/1 :");
        Scanner input = new Scanner(System.in);
        int layer = input.nextInt();
        System.out.println("input learning rate");
        double rate = input.nextDouble();
        if(layer==1){
            System.out.println("input jumlah neuron di hidden layer");
            int hidden = input.nextInt();
        }
        
        System.out.print("Masukkan nama file : ");
        String filename = input.next();
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(("C:\\Program Files\\Weka-3-8\\data\\"+filename));
        Instances train = source.getDataSet();
        for (int i=0; i < train.numAttributes();i++)
            System.out.println(i+". "+train.attribute(i).name());
        System.out.print("Masukkan indeks kelas : ");
        int classIdx = input.nextInt();
        train.setClassIndex(classIdx);
        MultiplePerceptron mlp = new MultiplePerceptron(1, rate , layer, train);
        mlp.buildClassifier(train);
    }
}
