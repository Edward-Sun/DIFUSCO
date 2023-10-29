
// Initialize the parameters used in MCTS
void MCTS_Init()
{
	for(int i=0;i<Virtual_City_Num;i++)
		for(int j=0;j<Virtual_City_Num;j++)
		{
			//Weight[i][j]=1;
			Weight[i][j]=Edge_Heatmap[i][j]*100;
			Chosen_Times[i][j]=0;
		}

	/*
	for(int i=0;i<Virtual_City_Num;i++)
		for(int j=0;j<Virtual_City_Num;j++)
			Weight[i][j]+=Edge_Heatmap[i][j]*100;
	*/

	Total_Simulation_Times=0;
}

//Get the average weight of all the edge relative to Cur_City
double Get_Avg_Weight(int Cur_City)
{
	double Total_Weight=0;
	for(int i=0;i<Virtual_City_Num;i++)
	{
		if(i==Cur_City)
			continue;

		Total_Weight+=Weight[Cur_City][i];
	}

	return Total_Weight/(Virtual_City_Num-1);
}

//Estimate the potential of each edge by upper bound confidence function
double Get_Potential(int First_City, int Second_City)
{
	double Potential=Weight[First_City][Second_City]/Avg_Weight+Alpha*sqrt( log(Total_Simulation_Times+1) / ( log(2.718)*(Chosen_Times[First_City][Second_City]+1) ) );

	return Potential;
}

// Indentify the promising cities as candidates which are possible to connect to Cur_City
void Identify_Promising_City(int Cur_City, int Begin_City)
{
	Promising_City_Num=0;
	for(int i=0;i<Candidate_Num[Cur_City];i++)
	{
		int Temp_City = Candidate[Cur_City][i];
		if(Temp_City == Begin_City)
			continue;
		if(Temp_City == All_Node[Cur_City].Next_City)
			continue;
		if(Get_Potential(Cur_City, Temp_City) < 1)
			continue;

		Promising_City[Promising_City_Num++]=Temp_City;
	}
}

// Set the probability (stored in Probabilistic[]) of selecting each candidate city (proportion to the potential of the corresponding edge)
bool Get_Probabilistic(int Cur_City)
{
	if(Promising_City_Num==0)
		return false;

	double Total_Potential=0;
	for(int i=0;i<Promising_City_Num;i++)
		Total_Potential+=Get_Potential(Cur_City, Promising_City[i]);

	Probabilistic[0]=(int)(1000*Get_Potential(Cur_City, Promising_City[0])/Total_Potential);
	for(int i=1;i<Promising_City_Num-1;i++)
		Probabilistic[i]=Probabilistic[i-1]+(int)(1000*Get_Potential(Cur_City, Promising_City[i])/Total_Potential);
	Probabilistic[Promising_City_Num-1]=1000;

	return true;
}

// Probabilistically choose a city, controled by the values stored in Probabilistic[]
int Probabilistic_Get_City_To_Connect()
{
	int Random_Num=Get_Random_Int(1000);
	for(int i=0;i<Promising_City_Num;i++)
		if(Random_Num < Probabilistic[i])
			return Promising_City[i];

	return Null;
}

// The whole process of choosing a city (a_{i+1} in the paper) to connect Cur_City (b_i in the paper)
int Choose_City_To_Connect(int Cur_City, int Begin_City)
{
	Avg_Weight=Get_Avg_Weight(Cur_City);
	Identify_Promising_City(Cur_City, Begin_City);
	Get_Probabilistic(Cur_City);

	return Probabilistic_Get_City_To_Connect();
}

// Generate an action starting form Begin_City (corresponding to a_1 in the paper), return the delta value
Distance_Type Get_Simulated_Action_Delta(int Begin_City)
{
	// Store the current solution to Solution[]
	if(Convert_All_Node_To_Solution()==false)
		return -Inf_Cost;

	int Next_City=All_Node[Begin_City].Next_City;   // a_1=Begin city, b_1=Next_City

	// Break edge (a_1,b_1)
	All_Node[Begin_City].Next_City=Null;
	All_Node[Next_City].Pre_City=Null;

	// The elements of an action is stored in City_Sequence[], where a_{i+1}=City_Sequence[2*i], b_{i+1}=City_Sequence[2*i+1]
	City_Sequence[0]=Begin_City;
	City_Sequence[1]=Next_City;

	Gain[0]=Get_Distance(Begin_City,Next_City);                // Gain[i] stores the delta (before connecting to a_1) at the (i+1)th iteration
	Real_Gain[0]=Gain[0]-Get_Distance(Next_City,Begin_City);   // Real_Gain[i] stores the delta (after connecting to a_1) at the (i+1)th iteration
	Pair_City_Num=1;                                            // Pair_City_Num indicates the depth (k in the paper) of the action

	bool If_Changed=false;
	int Cur_City=Next_City;	    // b_i = Cur_City (1 <= i <= k)
	while(true)
	{
		int Next_City_To_Connect=Choose_City_To_Connect(Cur_City,Begin_City);	// 	Probabilistically choose one city as a_{i+1}
		if(Next_City_To_Connect == Null)
			break;

		//Update the chosen times, used in MCTS
		Chosen_Times[Cur_City][Next_City_To_Connect] ++;
		Chosen_Times[Next_City_To_Connect][Cur_City] ++;

		int Next_City_To_Disconnect=All_Node[Next_City_To_Connect].Pre_City;   // Determine b_{i+1}

		// Update City_Sequence[], Gain[], Real_Gain[] and Pair_City_Num
		City_Sequence[2*Pair_City_Num]=Next_City_To_Connect;
		City_Sequence[2*Pair_City_Num+1]=Next_City_To_Disconnect;
		Gain[Pair_City_Num]=Gain[Pair_City_Num-1]-Get_Distance(Cur_City,Next_City_To_Connect)+Get_Distance(Next_City_To_Connect,Next_City_To_Disconnect);
		Real_Gain[Pair_City_Num]=Gain[Pair_City_Num]-Get_Distance(Next_City_To_Disconnect,Begin_City);
		Pair_City_Num++;

		// Reverse the cities between b_i and b_{i+1}
		Reverse_Sub_Path(Cur_City,Next_City_To_Disconnect);
		All_Node[Cur_City].Next_City=Next_City_To_Connect;
		All_Node[Next_City_To_Connect].Pre_City=Cur_City;
		All_Node[Next_City_To_Disconnect].Pre_City=Null;
		If_Changed=true;

		// Turns to the next iteration
		Cur_City=Next_City_To_Disconnect;

		// Close the loop is meeting an improving action, or the depth reaches its upper bound
		if(Real_Gain[Pair_City_Num-1] > 0 || Pair_City_Num > Max_Depth)
			break;
	}

	// Restore the solution before simulation
	if(If_Changed)
		Convert_Solution_To_All_Node();
	else
	{
		All_Node[Begin_City].Next_City=Next_City;
		All_Node[Next_City].Pre_City=Begin_City;
	}

	// Identify the best depth of the simulated action
	int Max_Real_Gain=-Inf_Cost;
	int Best_Index=1;
	for(int i=1;i<Pair_City_Num;i++)
		if(Real_Gain[i] > Max_Real_Gain)
		{
			Max_Real_Gain=Real_Gain[i];
			Best_Index=i;
		}

	Pair_City_Num=Best_Index+1;

	return Max_Real_Gain;
}

// If the delta of an action is greater than zero, use the information of this action (stored in City_Sequence[]) to update the parameters by back propagation
void Back_Propagation(Distance_Type Before_Simulation_Distance, Distance_Type Action_Delta)
{
	for(int i=0;i<Pair_City_Num;i++)
	{
		int First_City=City_Sequence[2*i];
		int Second_City=City_Sequence[2*i+1];
		int Third_City;
		if(i<Pair_City_Num-1)
			Third_City=City_Sequence[2*i+2];
		else
			Third_City=City_Sequence[0];

		if(Action_Delta >0)
		{
			double Increase_Rate=Beta*(pow(2.718, (double) (Action_Delta) / (double)(Before_Simulation_Distance) )-1);
			Weight[Second_City][Third_City] += Increase_Rate;
			Weight[Third_City][Second_City] += Increase_Rate;
		}
	}
}

// Sampling at most Max_Simulation_Times actions
Distance_Type Simulation(int Max_Simulation_Times)
{
	Distance_Type Best_Action_Delta = -Inf_Cost;
	for(int i=0;i<Max_Simulation_Times;i++)
	{
		int Begin_City=Get_Random_Int(Virtual_City_Num);
		Distance_Type Action_Delta=Get_Simulated_Action_Delta(Begin_City);
		Total_Simulation_Times++;

		//Store the action with the best delta, stored in Temp_City_Sequence[] and Temp_Pair_Num
		if(Action_Delta > Best_Action_Delta)
		{
			Best_Action_Delta = Action_Delta;

			Temp_Pair_Num = Pair_City_Num;
			for(int j=0;j<2*Pair_City_Num;j++)
				Temp_City_Sequence[j]=City_Sequence[j];
		}

		if(Best_Action_Delta >0)
			break;
	}

	// Restore the action with the best delta
	Pair_City_Num=Temp_Pair_Num;
	for(int i=0;i<2*Pair_City_Num;i++)
		City_Sequence[i]=Temp_City_Sequence[i];

	return Best_Action_Delta;
}

//Execute the best action stored in City_Sequence[] with depth Pair_City_Num
bool Execute_Best_Action()
{
	int Begin_City=City_Sequence[0];
	int Cur_City=City_Sequence[1];
	All_Node[Begin_City].Next_City=Null;
	All_Node[Cur_City].Pre_City=Null;
	for(int i=1;i<Pair_City_Num;i++)
	{
		int Next_City_To_Connect=City_Sequence[2*i];
		int Next_City_To_Disconnect=City_Sequence[2*i+1];

		Reverse_Sub_Path(Cur_City,Next_City_To_Disconnect);

		All_Node[Cur_City].Next_City=Next_City_To_Connect;
		All_Node[Next_City_To_Connect].Pre_City=Cur_City;
		All_Node[Next_City_To_Disconnect].Pre_City=Null;

		Cur_City=Next_City_To_Disconnect;
	}

	All_Node[Begin_City].Next_City=Cur_City;
	All_Node[Cur_City].Pre_City=Begin_City;

	if(Check_Solution_Feasible()==false)
	{
		printf("\nError! The solution after applying action from %d is unfeasible\n",Begin_City+1);
		Print_TSP_Tour(Begin_City);
		getchar();
		return false;
	}

	return true;
}

// Process of the MCTS
void MCTS()
{
	//while(true)
	while(((double)clock()-Current_Instance_Begin_Time) /CLOCKS_PER_SEC<Param_T*Virtual_City_Num)
	{
		Distance_Type Before_Simulation_Distance = Get_Solution_Total_Distance();

		//Simulate a number of (controled by Param_H) actions
		Distance_Type Best_Delta=Simulation(Param_H*Virtual_City_Num);

		// Use the information of the best action to update the parameters of MCTS by back propagation
		Back_Propagation(Before_Simulation_Distance,Best_Delta);

		if(Best_Delta > 0)
		{
			// Select the best action to execute
			Execute_Best_Action();

			// Store the best found solution to Struct_Node *Best_All_Node
			Distance_Type Cur_Solution_Total_Distance=Get_Solution_Total_Distance();
			if(Cur_Solution_Total_Distance < Current_Instance_Best_Distance)
			{
				Current_Instance_Best_Distance = Cur_Solution_Total_Distance;
				Store_Best_Solution();
			}
		}
		else
			break;	  // The MCTS terminates if no improving action is found among the sampling pool
	}
}
