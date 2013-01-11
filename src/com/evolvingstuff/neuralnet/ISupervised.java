package com.evolvingstuff.neuralnet;

public interface ISupervised {
	double[] Next(double[] input, double[] target_output) throws Exception;
	double[] Next(double[] input) throws Exception;
}
