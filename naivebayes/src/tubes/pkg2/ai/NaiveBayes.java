/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes.pkg2.ai;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Debug;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;
/**
 *
 * @author Mujahid Suriah
 */
public class NaiveBayes extends AbstractClassifier{
    private Instances datatrain;
    private final ArrayList<Atribut> listAtribut;
    
    public NaiveBayes () throws Exception {
        listAtribut = new ArrayList<>();
    }
    
    public Instances filter(Instances data) throws Exception {
        Instances newdata = data;
        NumericToNominal convert= new NumericToNominal();
        convert.setInputFormat(newdata);
        newdata = Filter.useFilter(newdata, convert);
        return newdata;
    }
    
    
    public ArrayList<Atribut> getList() {
        return listAtribut;
    }
    
    public void toSummaryString() {
        System.out.println("                    class");
        System.out.print("Attribute         ");
        for (int i = 0; i < datatrain.attribute(datatrain.classIndex()).numValues(); i++) {
            System.out.print(datatrain.attribute(datatrain.classIndex()).value(i)+"  ");
        }
        System.out.println();
        System.out.println("==============================================================");
        for(Atribut attr : listAtribut) {
            System.out.println(attr.getName());
            for (Nilai value : attr.getListAtribut()) {
                System.out.print(value.getName() + "                     ");
                for (Kelas valKelas : value.getListKelas()) {
                    System.out.print(valKelas.getFrekuensi()+ "                ");
                }
                System.out.println();
            }
            System.out.println();
        }
    }
    /**
     * @param args the command line arguments
     * @throws java.io.IOException
     */
    public static void main(String[] args) throws IOException, Exception {
        System.out.print("1. Buat Model \n");
        System.out.print("2. Load Model\n");
        System.out.print("Masukkan pilihan : ");
        Scanner sc = new Scanner(System.in);
        int pil = sc.nextInt();
        DataSource source = new DataSource(("D:\\Program Files\\Weka-3-8\\data\\iris.2D.arff"));
        Instances train = source.getDataSet();
        train.setClassIndex(train.numAttributes() - 1);
        NaiveBayes tb = new NaiveBayes();
        train = tb.filter(train);
            // setup filter
         Filter f = new Discretize();
         f.setInputFormat(train);
         Evaluation eval = new Evaluation(train);
         // apply filter
        train = Filter.useFilter(train, f);
        switch(pil) {
            case 1 : 
                tb.buildClassifier(train);
                //tb.toSummaryString();
                eval.crossValidateModel(tb, train ,10, new Random(1));
                System.out.println(eval.toSummaryString());
                saveModel(tb);
                break;
            default :
                tb = loadModel();
                train = tb.filter(train);
                 f.setInputFormat(train);
                train = Filter.useFilter(train, f);

                train.setClassIndex(train.numAttributes() - 1);
                tb.buildClassifier(train);
                //tb.toSummaryString();
                eval.crossValidateModel(tb, train ,10, new Random(1));
                System.out.println(eval.toSummaryString());
        }
    }

    @Override
    public void buildClassifier(Instances i) throws Exception {
        datatrain = i;
        for (int j = 0; j < datatrain.numAttributes()-1; j++) {
            listAtribut.add(new Atribut(datatrain, j, i.classIndex()));
        }
    }
    
    @Override
    public double classifyInstance(Instance last) {
        double prob[] = new double[last.classAttribute().numValues()];
        int [] countEachClass = getNumEachClass();
        int idx = 0;
        for (int classIndex = 0; classIndex < last.attribute(last.numAttributes() - 1).numValues(); classIndex ++ ) {
            double temp = 1;
            //System.out.println(last.classAttribute().value(classIndex));
            for (int i=0; i < last.numAttributes()-1; i++) {
                for (Atribut attr : getList()) {
                    temp = temp * attr.getFrekuensiNilai(last.toString(i), 
                            last.attribute(last.numAttributes() - 1).value(classIndex))/countEachClass[idx];
                }
                
            }
            double res;
            res = countEachClass[idx]/last.numAttributes() * temp;
            prob[classIndex] = res;
            idx++;
        }
        return maxIndex(prob);
    }
    
    public static NaiveBayes loadModel () throws IOException, ClassNotFoundException {
        Scanner sc = new Scanner (System.in);
        System.out.print("Nama file model : ");
        String filename = sc.nextLine();
        
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename)))
        {
            NaiveBayes cls = (NaiveBayes) ois.readObject();
            ois.close();
            return cls;
        }
    }
    
    public static void saveModel (Classifier c) throws IOException {
        Scanner sc = new Scanner (System.in);
        System.out.print("Nama file model : ");
        String filename = sc.nextLine();
        
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(c);
            oos.flush();
            oos.close();
            System.out.println("Save success!");
        }
    }
    
    private int[] getNumEachClass() {
        int [] countEachClass = new int[datatrain.numAttributes()];
        for (int i=0; i < datatrain.numAttributes()-1; i++) {
            for (Atribut attr : getList()) {
                for (Nilai val : attr.getListAtribut()) {
                    countEachClass[i] = (int)val.getNumClass();
                }
            }
        }
        return countEachClass;
    }
    
    private static int maxIndex (double [] val) {
        double max = 0;
        int maxIndex = 0;
        int i =0;
        for (double x : val) {
            if (x > max) {
              max = x;
              maxIndex = i;
            }
            i++;
        }
        return maxIndex;
    }
    
    public String classify(Instances init) {
        int nAttributes = init.numAttributes();
        Instance ins = new DenseInstance(nAttributes);
        Instances newData = init;
        Scanner s = new Scanner(System.in);
        Double in;
        System.out.println("Jumlah Atribut : " + (nAttributes - 1));
        
        for (int i = 1; i <= nAttributes - 1; i++) {
            //Attribute a = train.attribute(i - 1);
            System.out.print("Attribute "+ i + " : ");
            in = s.nextDouble();
            ins.setValue(i, in);
            //newIns[i] = in;
        }
        newData.add(ins);
        double nomorKelas = classifyInstance(newData.lastInstance());
        return init.attribute(init.numAttributes()-1).value((int)nomorKelas);
    }
}
