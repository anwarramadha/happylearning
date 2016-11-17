/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package FFNN;

import java.util.ArrayList;

/**
 *
 * @author Mujahid Suriah
 */
public class Node {
    double value;
    double error;
    ArrayList<Double> weight;
    
    public Node (int numWeight) {
        weight = new ArrayList<>(numWeight);
        for (int num=0;num<numWeight;num++) weight.add(0.0);
    }
    
    public void setValue(double val) {
        value = val;
    }
    
    public void setError(double val) {
        error = val;
    }
    
    public double getError() {
        return error;
    }
    
    public double getValue() {
        return value;
    }
    
    public void setWeight(int index, double w) {
        weight.add(w);
        weight.set(index, w);
    }
    
    public double getWeightFromList(int index) {
        return weight.get(index);
    }
    
    public int getWeightSize() {
        return weight.size();
    }
    
    
    public double output(ArrayList<Double> listInput) {
        double sigma = 0;
        System.out.println("size"+weight.size());
        for (int ih = 0; ih< weight.size(); ih++)  {
            sigma+= weight.get(ih)*listInput.get(ih);
        }        
        return sigmoid (sigma); 
    }
    
    
    private double sigmoid(double input) {
        return 1/(1+Math.pow(Math.E,input*(-1)));
    }
    
}
