/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ANN_single2;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

/**
 *
 * @author Mujahid Suriah
 */
public class Node implements Serializable{
    private double value=1;
    private double error=10;
    private ArrayList<Double> weight;
    
    public Node (int numWeight) {
        value=1;
        weight = new ArrayList<>();
        for (int num=0;num<numWeight;num++) weight.add(Math.random());
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
        weight.set(index, w);
//        System.out.println(weight.get(index));
    }
    
    public double getWeightFromList(int index) {
        return weight.get(index);
    }
    
    public ArrayList<Double> getWeight() {
        return weight;
    }
    public int getWeightSize() {
        return weight.size();
    }
    
    
    public void output(ArrayList<Double> listInput) {
        double sigma = 0;
        for (int ih = 0; ih < weight.size(); ih++)  {
            System.out.println(weight.get(ih)+"*"+listInput.get(ih));
            sigma += weight.get(ih)*listInput.get(ih);//input value = 1, value
        }        
//        System.out.println(sigma+" "+sigmoid(sigma));
        value = sigmoid (sigma); 
    }
    
    
    private double sigmoid(double input) {
        return 1/(1+Math.pow(Math.E,input*(-1)));
    }
    
}
