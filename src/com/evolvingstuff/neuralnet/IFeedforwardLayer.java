package com.evolvingstuff.neuralnet;

public interface IFeedforwardLayer {
	int GetInputDimension();
	int GetOutputDimension();
	double[] Forward(double[] input);
	double[] Backprop(double[] delta) throws Exception;
}
