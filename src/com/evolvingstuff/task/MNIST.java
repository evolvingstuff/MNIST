package com.evolvingstuff.task;

import java.io.*;

import com.evolvingstuff.agent.IAgentSupervised;
import com.evolvingstuff.evaluator.*;

public class MNIST implements IInteractiveEvaluatorSupervised {
	/*
	The data is stored in a very simple file format designed for storing vectors and multidimensional matrices. General info on this format is given at the end of this page, but you don't need to read that to use the data files.

	All the integers in the files are stored in the MSB first (high endian) format used by most non-Intel processors. Users of Intel processors and other low-endian machines must flip the bytes of the header.

	There are 4 files:
	
	train-images-idx3-ubyte: training set images
	train-labels-idx1-ubyte: training set labels
	t10k-images-idx3-ubyte:  test set images
	t10k-labels-idx1-ubyte:  test set labels
	
	The training set contains 60000 examples, and the test set 10000 examples.  
	
	TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
	[offset] [type]          [value]          [description]
	0000     32 bit integer  0x00000801(2049) magic number (MSB first)
	0004     32 bit integer  60000            number of items
	0008     unsigned byte   ??               label
	0009     unsigned byte   ??               label
	........
	xxxx     unsigned byte   ??               label
	
	The labels values are 0 to 9.
	TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
	[offset] [type]          [value]          [description]
	0000     32 bit integer  0x00000803(2051) magic number
	0004     32 bit integer  60000            number of images
	0008     32 bit integer  28               number of rows
	0012     32 bit integer  28               number of columns
	0016     unsigned byte   ??               pixel
	0017     unsigned byte   ??               pixel
	........
	xxxx     unsigned byte   ??               pixel
	
	Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
	TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
	[offset] [type]          [value]          [description]
	0000     32 bit integer  0x00000801(2049) magic number (MSB first)
	0004     32 bit integer  10000            number of items
	0008     unsigned byte   ??               label
	0009     unsigned byte   ??               label
	........
	xxxx     unsigned byte   ??               label
	
	The labels values are 0 to 9.
	TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
	[offset] [type]          [value]          [description]
	0000     32 bit integer  0x00000803(2051) magic number
	0004     32 bit integer  10000            number of images
	0008     32 bit integer  28               number of rows
	0012     32 bit integer  28               number of columns
	0016     unsigned byte   ??               pixel
	0017     unsigned byte   ??               pixel
	........
	xxxx     unsigned byte   ??               pixel
	
	Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black). 
	*/
	
	private final int height = 28;
	private final int width = 28;
	private final int task_action_dimension = 10; //0-9
	private final int task_observation_dimension = height * width; //28x28=784
	private final int total_train = 60000;
	private final int total_test = 10000;
	private final String train_images = "train-images-idx3-ubyte";
	private final String train_labels = "train-labels-idx1-ubyte";
	private final String test_images = "t10k-images-idx3-ubyte";
	private final String test_labels = "t10k-labels-idx1-ubyte";
	private boolean validation_mode = false;
	
	private double InnerEval(double[] agent_output, double[] input_to_agent, int target_loc) throws Exception {
		double high = Double.NEGATIVE_INFINITY;
		int high_loc = -1;
		for (int i = 0; i < agent_output.length; i++) {
			if (agent_output[i] > high) {
				high = agent_output[i];
				high_loc = i;   
			}
		}
		if (high_loc == target_loc) {
			return 1.0;
		}
		else {
			return 0.0;
		}
	}
	
	private double EvaluateSampleSupervised(byte[] bimg, byte[] blbl, IAgentSupervised agent, boolean give_target) throws Exception {
		double[] input_to_agent = new double[task_observation_dimension];
		int loc = 0;
		int k = 0;
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				input_to_agent[loc] = (double)((int) bimg[k] & 0xFF)/255.0;
				loc++;
				k++;
			}
		}
		int target_loc = (int) blbl[0] & 0xFF;
		double[] target_vec;
		
		target_vec = new double[task_action_dimension];
		target_vec[target_loc] = 1;

		double[] agent_output;
		if (give_target) {
			agent_output = agent.Next(input_to_agent, target_vec);
		}
		else {
			agent_output = agent.Next(input_to_agent);
		}
		return InnerEval(agent_output, input_to_agent, target_loc);
	}

	public int GetActionDimension() {
		return task_action_dimension;
	}

	public int GetObservationDimension() {
		return task_observation_dimension;
	}

	public void SetValidationMode(boolean validation) {
		validation_mode = validation;
	}

	public double EvaluateFitnessSupervised(IAgentSupervised agent) throws Exception {
		byte[] bimg = new byte[task_observation_dimension];
		byte[] blbl = new byte[1];
		double tot_fit = 0;
		int total_errors = 0;
		if (validation_mode == false) {
			FileInputStream images = new FileInputStream(train_images);
			FileInputStream labels = new FileInputStream(train_labels);
			images.skip(16);
			labels.skip(8);
			double tot_evaluated = 0;
			for (int n = 0; n < total_train; n++) {
				if (n % 1000 == 999) {
					System.out.print(".");
				}
				images.read(bimg);
				labels.read(blbl);
				double fit = EvaluateSampleSupervised(bimg, blbl, agent, true);
				if (fit < 1) {
					total_errors++;
				}
				tot_fit += fit;
				tot_evaluated += 1;

			}
			images.close();
			labels.close();
			tot_fit /= tot_evaluated;
			System.out.println("\nTRAIN ERRORS: " + total_errors + " (of "+total_train+")");
			return tot_fit;
		}
		else { //validation == true
			
			FileInputStream images = new FileInputStream(test_images);
			FileInputStream labels = new FileInputStream(test_labels);
			images.skip(16);
			labels.skip(8);
			for (int n = 0; n < total_test; n++) {
				if (n % 1000 == 999) {
					System.out.print(".");
				}
				images.read(bimg);
				labels.read(blbl);
				double fit = EvaluateSampleSupervised(bimg, blbl, agent, false);
				if (fit < 1) {
					total_errors++;
				}
				tot_fit += fit;
			}
			images.close();
			labels.close();
			tot_fit /= total_test;
			System.out.println("\nTEST ERRORS: " + total_errors + " (of "+total_test+")");
			return tot_fit;
		}
	}
}

