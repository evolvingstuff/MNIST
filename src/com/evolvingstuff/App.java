package com.evolvingstuff;

import java.util.Random;

import com.evolvingstuff.agent.FlatJumpNetwork;
import com.evolvingstuff.agent.Neuron;
import com.evolvingstuff.agent.RectifiedLinearNeuron;
import com.evolvingstuff.task.MNIST;

public class App {
	public static void main(String[] args) throws Exception {
		
		System.out.println("MNIST");
		
		final Random r = new Random(5532122);//54324
		final MNIST task = new MNIST();
		final int epoches = 500;
		final int hidden_per_layer = 6000;
		final Neuron neuron = new RectifiedLinearNeuron(0.01);
		final double init_weight_range = 0.2;
		final double learning_rate = 0.3;
		final FlatJumpNetwork agent = new FlatJumpNetwork(r, task.GetObservationDimension(), task.GetActionDimension(), hidden_per_layer, neuron, init_weight_range, learning_rate);
		final String path = "data/";
		final boolean load_initially = false;
		final boolean save_on_improvement = true;
		
		if (load_initially) {
			try {
				System.out.println("Attempting to load previously saved agent from "+path+".");
				agent.Load(path);
			}
			catch (Exception e) {
				System.out.println("Failed to load. Possible cause: "+path+" folder might be empty?");
			}
		}
		
		int low_at = 0;
		double low = Double.POSITIVE_INFINITY;
		for (int t = 0; t < epoches; t++)
		{
			System.out.println("\n-----------------------------------------------------------------");
			System.out.println("[epoch "+(t+1)+"]:");
			
			task.SetValidationMode(false);
			task.EvaluateFitnessSupervised(agent);
			task.SetValidationMode(true);
			double validation = task.EvaluateFitnessSupervised(agent);

			if (1 - validation < low) {
				low = 1 - validation;
				low_at = t;
				
				if (save_on_improvement) {			
					System.out.println("Lowest test error so far. Saving (this will overwrite contents of "+path+" folder.)");
					agent.Save(path);
				}
			}
			System.out.println("lowest test error @ epoch " + (low_at+1));
		}
		System.out.println("\n\ndone.");
	}
}
