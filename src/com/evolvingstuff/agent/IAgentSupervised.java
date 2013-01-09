package com.evolvingstuff.agent;

public interface IAgentSupervised {
	double[] Next(double[] input, double[] target_output) throws Exception;
	double[] Next(double[] input) throws Exception;
}
