package com.evolvingstuff.neuralnet;

public abstract class Neuron {
	abstract public double Activate(double x);
	abstract public double Derivative(double x);
}
