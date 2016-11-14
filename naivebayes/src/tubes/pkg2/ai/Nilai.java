/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubes.pkg2.ai;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.jar.Attributes;
import weka.core.Attribute;
import weka.core.Instances;

/**
 *
 * @author Mujahid Suriah
 */
public class Nilai implements Serializable {
    private final String name;
    private final int index;
    private final ArrayList<Kelas> kelas;
    private double numEachClass;
    public Nilai (Instances inst, int i, int j, int classindex) {
        name = inst.attribute(i).value(j);
        kelas = new ArrayList<>();
        index = 0;
        for (int k = 0; k < inst.attribute(classindex).numValues(); k++) { //buat nama kelas size = 3
            double cnt = 1;
            double cntClass = 0;
            for(int m = 0; m < inst.instance(i).numValues(); m++) { //buat akses elemen baris ke-m size = 5
                for (int l = 0; l < inst.numInstances(); l++) { //jumlah seluruh instances
                    if (inst.attribute(classindex).value(k).equalsIgnoreCase(inst.get(l).toString(classindex)) && m == i) cntClass += 1;
                    if (inst.attribute(i).value(j).equalsIgnoreCase(inst.get(l).toString(m)) && 
                            inst.attribute(classindex).value(k).equalsIgnoreCase(inst.get(l).toString(classindex)) && m == i) {//jika ada nilai yang sama pada atribut 
                            //dan kelas yang sama
                        cnt+=1;
                    }
                }
            }
            numEachClass = cntClass;
            kelas.add(new Kelas(inst.attribute(classindex).value(k),cnt, cntClass));
        }
    }
    
    public String getName() {
        return name;
    }
    
    public ArrayList<Kelas> getListKelas() {
        return kelas;
    }
    
    public double getFrekuensiNilai (String attrName) {
        for (Kelas valKelas : kelas) {
            if (valKelas.getName().equalsIgnoreCase(attrName)) {
                return valKelas.getFrekuensi();
            }
        }
        return 1;
    }
    
    public double getNumClass() {
        return numEachClass;
    }
}
