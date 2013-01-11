package com.evolvingstuff;

import java.util.Random;

import com.evolvingstuff.neuralnet.FlatJumpNetwork;
import com.evolvingstuff.neuralnet.Neuron;
import com.evolvingstuff.neuralnet.RectifiedLinearNeuron;
import com.evolvingstuff.task.MNIST;

public class App {
	public static void main(String[] args) throws Exception {
		
		System.out.println("MNIST");
		
		final Random r = new Random(5532122);//54324
		final String saved_progress_path = "saved-progress/";
		final String mnist_data_path = "mnist-data/";
		final MNIST task = new MNIST(mnist_data_path);
		final int epoches = 500;
		final int hidden_per_layer = 6000;
		final Neuron neuron = new RectifiedLinearNeuron(0.01);
		final double init_weight_range = 0.2;
		final double learning_rate = 0.3;
		final FlatJumpNetwork neural_network = new FlatJumpNetwork(r, task.GetObservationDimension(), task.GetActionDimension(), hidden_per_layer, neuron, init_weight_range, learning_rate);
		
		final boolean load_initially = false;
		final boolean save_on_improvement = true;
		
		if (load_initially) {
			try {
				System.out.println("Attempting to load previously saved agent from "+saved_progress_path+".");
				neural_network.Load(saved_progress_path);
			}
			catch (Exception e) {
				System.out.println("Failed to load. Possible cause: "+saved_progress_path+" folder might be empty?");
			}
		}
		
		int low_at = 0;
		double low = Double.POSITIVE_INFINITY;
		for (int t = 0; t < epoches; t++)
		{
			System.out.println("\n-----------------------------------------------------------------");
			System.out.println("[epoch "+(t+1)+"]:");
			
			task.SetValidationMode(false);
			task.EvaluateFitnessSupervised(neural_network);
			task.SetValidationMode(true);
			double validation = task.EvaluateFitnessSupervised(neural_network);

			if (1 - validation < low) {
				low = 1 - validation;
				low_at = t;
				
				if (save_on_improvement) {			
					System.out.println("Lowest test error so far. Saving (this will overwrite contents of "+saved_progress_path+" folder.)");
					neural_network.Save(saved_progress_path);
				}
			}
			System.out.println("lowest test error @ epoch " + (low_at+1));
		}
		System.out.println("\n\ndone.");
	}
}
