/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NaiveBayes;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

/**
 *
 * @author Mujahid Suriah
 */
public class Nilai implements Serializable {
    private final String name;
    private  double lower;
    private  double upper;
    private final ArrayList<Kelas> kelas;
    private final int [] numClass;
    public Nilai (Instances inst, int i, int j, int classindex) throws Exception {
        Instances newData = new Instances(inst);
        numClass = NaiveBayes.getNumEachClass(newData);
        lower = 0;
        upper = 0;
        kelas = new ArrayList<>();
        //if(newData.instance(i).isMissing(j)) newData.instance(i).setValue(i, "b");
        if (newData.attribute(i).isNumeric()) {
            Discretize f = new Discretize();
            f.setInputFormat(newData);
            newData = Filter.useFilter(newData, f);
            name = newData.attribute(i).value(j);
            if (f.getCutPoints(i)!=null) {
                if (j == 0) {
                    lower = Double.NEGATIVE_INFINITY;
                    upper = f.getCutPoints(i)[j];
                }
                else {
                    if (j != newData.attribute(0).numValues()-1) {
                        lower = f.getCutPoints(i)[j-1];
                        upper = f.getCutPoints(i)[j];
                    }
                    else {
                        lower = f.getCutPoints(i)[j-1];
                        upper = Double.POSITIVE_INFINITY;
                    }
                }
            }else 
            {
                lower = Double.NEGATIVE_INFINITY;
                upper = Double.POSITIVE_INFINITY;
            }
            for (int k = 0; k < inst.attribute(classindex).numValues(); k++) { //buat nama kelas
                double cnt = 1;
                int countClass = 0;
                for (int l = 0; l < inst.numInstances(); l++) { //jumlah seluruh instances
                    double val = inst.get(l).value(i);
                    if (countClass <= numClass[k]) {
                        if (inst.attribute(classindex).value(k).
                                equalsIgnoreCase(inst.get(l).toString(classindex).replaceAll("'", ""))) {/*nama kelasnya*/
                            if (val >= lower && val < upper) {//jika ada nilai yang sama pada atribut 
                                //dan kelas yang sama dan nilai dari atribut lebih besar sama dengan lower
                                cnt+=1;
                            }
                            countClass++;
                        }
                    }
                    else break;
                }
                kelas.add(new Kelas(newData.attribute(classindex).value(k),cnt));
            }
        }
        else {
            //System.out.println(newData.attribute(i).value(j).replaceAll("\\s+", ""));
            name = newData.attribute(i).value(j).replaceAll("\\s", "");
            //System.out.println(name);
            //System.out.println(name);
            for (int k = 0; k < inst.attribute(classindex).numValues(); k++) { //buat nama kelas
                double cnt = 1;
                int countClass = 0;
                for (int l = 0; l < inst.numInstances(); l++) { //jumlah seluruh instances
                    if (countClass <= numClass[k]) {
                        //System.out.println("with whitespace "+inst.attribute(i).value(j)+"without "+inst.attribute(i).value(j).replaceAll("\\s", "")+"p");
//                        System.out.println(inst.get(l).toString(classindex));
                        //System.out.println(inst.attribute(classindex).value(k));
                        if (inst.attribute(classindex).value(k).replaceAll("\\s", "").
                                equalsIgnoreCase(inst.get(l).toString(classindex).replaceAll("\\s", ""))//nama kelas
                                && inst.attribute(i).value(j).replaceAll("\\s", "").//cek nama atribut
                                equalsIgnoreCase(inst.get(l).toString(i).replaceAll("\\s", ""))) {
                             //jika ada nilai yang sama pada atribut 
                                //dan kelas yang sama dan nilai dari atribut lebih besar sama dengan lower
                            cnt+=1;
                            countClass++;
                        }
                    }
                    else break;
                }
                kelas.add(new Kelas(newData.attribute(classindex).value(k).replaceAll("\\s+", ""),cnt));
            }
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
            if (valKelas.getName().replace("\\s+", "").equalsIgnoreCase(attrName.replaceAll("\\s+", ""))) {
//            System.out.println(name);
                return valKelas.getFrekuensi();
            }
        }
        return 1;
    }
    
    public double getLower() {
        return lower;
    }
    
    public double getUpper() {
        return upper;
    }
}
