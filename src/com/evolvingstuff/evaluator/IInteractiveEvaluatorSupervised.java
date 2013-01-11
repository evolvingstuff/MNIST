package com.evolvingstuff.evaluator;

import com.evolvingstuff.neuralnet.*;

public interface IInteractiveEvaluatorSupervised 
{
	double EvaluateFitnessSupervised(ISupervised agent) throws Exception;
	int GetActionDimension();
	int GetObservationDimension();
	void SetValidationMode(boolean validation);
	
}
