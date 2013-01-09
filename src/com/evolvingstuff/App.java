package com.evolvingstuff;

import java.util.Random;

import com.evolvingstuff.agent.FlatJumpNetwork;
import com.evolvingstuff.agent.Neuron;
import com.evolvingstuff.agent.RectifiedLinearNeuron;
import com.evolvingstuff.task.MNIST;

public class App {
	public static void main(String[] args) throws Exception {
		
		Random r = new Random(54324);
		
		boolean doLoad = false;
		boolean doSave = true;
		
		MNIST task = new MNIST();

		System.out.println("MNIST");

		int epoches = 500;
		int hidden_per_layer = 6000;
		Neuron neuron = new RectifiedLinearNeuron(0.01);
		double init_weight_range = 0.2;
		double learning_rate = 0.3;
		
		String path = "data/";
		
		FlatJumpNetwork agent = new FlatJumpNetwork(r, task.GetObservationDimension(), task.GetActionDimension(), hidden_per_layer, neuron, init_weight_range, learning_rate);
		
		if (doLoad) {
			System.out.println("Loading previously saved agent.");
			agent.Load(path);
		}
		
		int low_at = 1;
		double low = Double.POSITIVE_INFINITY;
		for (int t = 0; t < epoches; t++)
		{
			System.out.println("-----------------------------------------------------------------");
			System.out.println("[epoch "+(t+1)+"]:");
			
			task.SetValidationMode(false);
			double fit = task.EvaluateFitnessSupervised(agent);
			task.SetValidationMode(true);
			double validation = task.EvaluateFitnessSupervised(agent);

			if (1 - validation < low) {
				low = 1 - validation;
				low_at = t;
				
				if (doSave) {			
					System.out.println("Lowest test error so far. Saving.");
					agent.Save(path);
				}
				
			}
			System.out.println("lowest test error @ epoch " + (low_at+1));
		}

		System.out.println("\n\ndone.");

	}
}
