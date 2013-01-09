package com.evolvingstuff.util;

import processing.core.*;

public class VisualizeFlatNetwork extends PApplet {
	
	private static final long serialVersionUID = 1L;
	
	public double[][] data;
	public double global_max = 0;
	public double max = 0;
	public int num = 0;
	public int dims = 900;//only show subset of hidden neurons so as to fit on screen
	public String path = "hidden";
	int pad = 2;
	public int pixls = 28;
	public int panels = 30;//30
	public boolean NORMALIZE_FEATURES_GLOBALLY = true;

	public void setup() {
		
		System.out.println("loading data...");
		try {
			data = util.FileToMatrix("../flatnn/"+path+".mtrx");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("done loading.");
		
		System.out.println(data[0].length + " -> " + data.length);
		
		for (int i = 0; i < data.length; i++) {
			for (int j = 0; j < data[i].length; j++) {
				if (i < dims && Math.abs(data[i][j]) > global_max) {
					global_max = Math.abs(data[i][j]);
				}
			}
		}
		
		System.out.println("global_max = " + global_max);
		
		size((pixls+pad)*panels, (pixls+pad)*panels); //P2D, OPENGL
		background(128,0,0);
	}
	
	private int DoubleToColor(double val, double max) {
		float v = (float)(127*val/max + 128);
		return this.color(v, v, v);
	}
	
	public void draw() {

		loadPixels();
		
		double max = 0;
		
		if (!NORMALIZE_FEATURES_GLOBALLY) {
			for (int j = 0; j < pixls; j++) {
				for (int i = 0; i < pixls; i++) {
					if (Math.abs(data[num][i+pixls*j]) > max) {
						max = Math.abs(data[num][i+pixls*j]);
					}
				}
			}
		}
		
		int panelcol = num%panels;
		int panelrow = num/panels;
		int x = panelcol*(pixls + pad) + pad/2;
		int y = panelrow*(pixls + pad) + pad/2;
		
		System.out.println("drawing "+num+"...col:" + panelcol + "/row:" + panelrow);
		
		for (int j = 0; j < pixls; j++) {
			for (int i = 0; i < pixls; i++) {
				int c = 0;
				if (NORMALIZE_FEATURES_GLOBALLY) {
					c = DoubleToColor(data[num][i+pixls*j], global_max);
				}
				else {
					c = DoubleToColor(data[num][i+pixls*j], max);
				}
				int loc = (y+j)*panels*(pixls+pad) + (x+i);
				pixels[loc] = c;
			}
		}
		updatePixels();

		num++;
		if (num >= dims) {
			save("../flatnn/AllFeatures.png");
			noLoop();
		}
	}
}
