package com.evolvingstuff.util;

import java.util.*;

import java.io.*;

public class util 
{
	public static double[] Delta(double[] target, double[] actual) {
		double[] result = new double[target.length];
		for (int i = 0; i < target.length; i++) {
			result[i] = target[i] - actual[i];
		}
		return result;
	}
	
	public static double[] ShortenVector(double[] vector, int new_length) {
		double[] result = new double[new_length];
		for (int i = 0; i < new_length; i++) {
			result[i] = vector[i];
		}
		return result;
	}
	
	public static void MatrixToFile(double[][] matrix, String path) throws Exception {
		FileWriter f = new FileWriter(new File(path));
		for (int j = 0; j < matrix.length; j++) {
			if (j > 0) {
				f.write("\n");
			}
			for (int i = 0; i < matrix[0].length; i++) {
				if (i > 0) {
					f.write(",");
				}
				f.write(String.valueOf(matrix[j][i]));
			}
		}
		int rows = matrix.length;
		int cols = matrix[0].length;
		System.out.println("util.MatrixToFile: " + rows + "x" + cols + " -> " + path);

		f.flush();
		f.close();
	}
	
	public static double[][] FileToMatrix(String path) throws Exception
	{
		int rows = 0;
		int cols = 0;
		
		List<List<Double>> vals = new ArrayList<List<Double>>();
		Scanner sc = new Scanner(new File(path));
		while (sc.hasNextLine())
		{
			String line = sc.nextLine();
			String[] parts = line.split(",");
			List<Double> row = new ArrayList<Double>();
			for (String part : parts)
			{
				double val = Double.parseDouble(part);
				row.add(val);
			}
			if (cols != row.size()) {
				if (cols == 0) {
					cols = row.size();
				}
				else {
					throw new Exception("jagged array?");
				}
			}
			vals.add(row);
		}
		rows = vals.size();
		System.out.println("util.FileToMatrix: " + path + " -> " + rows + "x" + cols);
		double[][] result = new double[rows][cols];
		for (int j = 0; j < rows; j++) {
			for (int i = 0; i < cols; i++) {
				result[j][i] = vals.get(j).get(i);
			}
		}
		return result;
	}
	
	public static double[] ConcatVectors(double[] vec1, double[] vec2) {
		double[] result = new double[vec1.length + vec2.length];
		int loc = 0;
		for (int i = 0; i < vec1.length; i++)
			result[loc++] = vec1[i];
		for (int i = 0; i < vec2.length; i++)
			result[loc++] = vec2[i];
		return result;
	}
}
