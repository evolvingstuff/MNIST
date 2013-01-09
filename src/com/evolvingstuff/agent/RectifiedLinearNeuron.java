package com.evolvingstuff.agent;

public class RectifiedLinearNeuron extends Neuron {

	private double slope;
	
	public RectifiedLinearNeuron(double slope) {
		this.slope = slope;
	}
	
	@Override
	public double Activate(double x) {
		if (x >= 0) {
			return x;
		}
		else {
			return x * slope;
		}
	}

	@Override
	public double Derivative(double x) {
		if (x >= 0) {
			return 1;
		}
		else {
			return slope;
		}
	}
}
