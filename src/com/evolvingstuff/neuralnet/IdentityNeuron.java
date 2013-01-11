package com.evolvingstuff.neuralnet;

public class IdentityNeuron extends Neuron
{
	@Override
	public double Activate(double x) {
		return x;
	}

	@Override
	public double Derivative(double x) {
		return 1;
	}
}

