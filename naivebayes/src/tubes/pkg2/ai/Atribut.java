/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes.pkg2.ai;

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
        Instances newData = new Instances(ints);
        Discretize f = new Discretize();
        f.setInputFormat(newData);
        newData = Filter.useFilter(newData, f);
        name = ints.attribute(i).name();
        listNilai = new ArrayList<>();
        for (int j = 0; j < newData.attribute(i).numValues(); j++) {
            listNilai.add(new Nilai(ints, i, j, classindex));
        }
    }
    
    
    public String getName() {
        return name;
    }
    
    public ArrayList<Nilai> getListAtribut() {
        return listNilai;
    }
   
    public double getFrekuensiNilai(String nama, String className, double val) {
        for (Nilai valNilai : listNilai) {
            if (nama.equalsIgnoreCase(name)) {
                if (val >= valNilai.getLower() && val < valNilai.getUpper()){
                    return valNilai.getFrekuensiNilai(className);
                }
            }
        }
        return 1;
    }
}
