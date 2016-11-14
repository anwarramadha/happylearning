/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes.pkg2.ai;

import java.io.Serializable;
import java.util.ArrayList;
import weka.core.Instances;

/**
 *
 * @author Mujahid Suriah
 */
public class Atribut implements Serializable {
    private final String name;
    private final int index;
    private final int num;
    private final ArrayList<Nilai> listNilai;
    public Atribut(Instances ints, int i, int classindex) {
        name = ints.attribute(i).name();
        index = ints.attribute(i).index();
        num = ints.attribute(i).numValues();
        listNilai = new ArrayList<>();
        for (int j = 0; j < ints.attribute(i).numValues(); j++) {
            listNilai.add(new Nilai(ints, i, j, classindex));
        }
    }
    
    public String getName() {
        return name;
    }
    
    public int getIndex() {
        return index;
    }
    
    public ArrayList<Nilai> getListAtribut() {
        return listNilai;
    }
   
    public int getNumValues() {
        return num;
    }
    
    public double getFrekuensiNilai(String attrName, String className) {
        for (Nilai valNilai : listNilai) {
            //System.out.println(attrName);
            if (valNilai.getName().equalsIgnoreCase(attrName)) {
                return valNilai.getFrekuensiNilai(className);
            }
        }
        return 1;
    }
}
