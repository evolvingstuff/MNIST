package com.evolvingstuff.evaluator;

import com.evolvingstuff.agent.*;

public interface IInteractiveEvaluatorSupervised 
{
	double EvaluateFitnessSupervised(IAgentSupervised agent) throws Exception;
	int GetActionDimension();
	int GetObservationDimension();
	void SetValidationMode(boolean validation);
	
}
