/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NaiveBayes;

import java.io.Serializable;
import java.util.ArrayList;
import weka.core.Attribute;


/**
 *
 * @author Mujahid Suriah
 */
public class Kelas implements Serializable {
    private final String name;
    private int index;
    private final double frekuensi;
    public Kelas(String str, double i) {
        name = str;
        frekuensi = i;
    }
    
    public String getName() {
        return name;
    }
    
    public int getIndex() {
        return index;
    }
    
    public double getFrekuensi() {
        return frekuensi;
    }
}
