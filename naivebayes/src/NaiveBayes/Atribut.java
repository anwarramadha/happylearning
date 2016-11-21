/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NaiveBayes;

import java.io.Serializable;
import java.util.ArrayList;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

/**
 *
 * @author Mujahid Suriah
 */
public class Atribut implements Serializable {
    private final String name;
    private final ArrayList<Nilai> listNilai;
    public Atribut(Instances ints, int i, int classindex) throws Exception {
        if (ints.attribute(i).isNumeric()) {
            Instances newData = new Instances(ints);
            Discretize f = new Discretize();
            f.setInputFormat(newData);
            newData = Filter.useFilter(newData, f);
            name = ints.attribute(i).name();listNilai = new ArrayList<>();
            for (int j = 0; j < newData.attribute(i).numValues(); j++) {
                listNilai.add(new Nilai(ints, i, j, classindex));
            }
        }
        else {
            name = ints.attribute(i).name().replaceAll("\\s+", "");
//            System.out.println(name);
            listNilai = new ArrayList<>();
            for (int j = 0; j < ints.attribute(i).numValues(); j++) {
                listNilai.add(new Nilai(ints, i, j, classindex));
            }
        }
        
    }
    
    
    public String getName() {
        return name;
    }
    
    public ArrayList<Nilai> getListAtribut() {
        return listNilai;
    }
   
    public double getFrekuensiNilai(String className, String valStr, double val, boolean isNumeric) {
        //System.out.println(nama);
        for (Nilai valNilai : listNilai) {
//                System.out.println(valNilai.getName());
            //System.out.println(listNilai.size());
        //System.out.println(name.replaceAll("\\s+", "")+ "="+nama.replaceAll("\\s+", ""));
            if (isNumeric) {
                if (val >= valNilai.getLower() && val < valNilai.getUpper()){
                    return valNilai.getFrekuensiNilai(className);
                }
            }
            else {
//                System.out.println(valStr+"="+valNilai.getName());
                if (valNilai.getName().replaceAll("\\s", "").equalsIgnoreCase(valStr.replaceAll("\\s", "")))
                    return valNilai.getFrekuensiNilai(className);
            } 

        }
        return 1;
    }
}
