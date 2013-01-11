package com.evolvingstuff.neuralnet;

import java.util.Random;

public class Layer implements IFeedforwardLayer {
	
	int input_dimension;
	int bias_indx;
	int output_dimension;
	Neuron neuron;
	double learning_rate;
	public double[][] weights;
	double[] input_acts;
	double[] output_acts;
	double[] output_sums;
	Random r;
	
	double DELTA_WARNING_SIZE = 1000;
	
	public Layer(Random r, int input_dimension, int output_dimension, Neuron neuron, double init_weight_range, double learning_rate) {
		this.r = r;
		this.input_dimension = input_dimension;
		this.output_dimension = output_dimension;
		this.neuron = neuron;
		this.learning_rate = learning_rate;
		double fan_in_factor = 1;
		fan_in_factor =  1.0 / Math.sqrt(input_dimension+1);
		this.learning_rate *= fan_in_factor;
		bias_indx = input_dimension;
		weights = new double[output_dimension][input_dimension+1];
		for (int k = 0; k < output_dimension; k++) {
			for (int i = 0; i < input_dimension+1; i++) {
				weights[k][i] = r.nextGaussian() * init_weight_range * fan_in_factor;
			}
		}
		input_acts = new double[input_dimension];
		output_sums = new double[output_dimension];
		output_acts = new double[output_dimension];
	}

	public int GetInputDimension() {
		return input_dimension;
	}

	public int GetOutputDimension() {
		return output_dimension;
	}

	public double[] Forward(double[] input) {
		input_acts = input;
		for (int k = 0; k < output_dimension; k++) {
			output_sums[k] = 0;
			output_acts[k] = 0;
			for (int i = 0; i < input_dimension; i++) {
				output_sums[k] += weights[k][i] * input[i];
			}
			output_sums[k] += weights[k][bias_indx];
			output_acts[k] = neuron.Activate(output_sums[k]);
		}
		return output_acts;
	}

	public double[] Backprop(double[] delta) throws Exception {
		double[] delta_at_input = new double[input_dimension];
		for (int k = 0; k < output_dimension; k++) {
			if (Math.abs(delta[k]) > DELTA_WARNING_SIZE) {
				throw new Exception("WARNING: delta["+k+"] > DELTA_WARNING_SIZE: " + delta[k] + ". Suggest using a smaller initial weight range if hidden neurons are unbounded activation.");
			}
			double delta_pre_nonlinearity = neuron.Derivative(output_sums[k]) * delta[k];
			for (int i = 0; i < input_dimension; i++) {
				delta_at_input[i] += delta_pre_nonlinearity * weights[k][i];
				weights[k][i] += input_acts[i] * delta_pre_nonlinearity * learning_rate;
			}
			weights[k][bias_indx] += delta_pre_nonlinearity * learning_rate;
		}
		return delta_at_input;
	}
}
