// Estimate the potential of each edge by upper bound confidence function
double Temp_Get_Potential(int First_City, int Second_City)
{
	// double Potential=Weight[First_City][Second_City]/Avg_Weight+Alpha*sqrt( log(Total_Simulation_Times+1) / ( log(2.718)*(Chosen_Times[First_City][Second_City]+1) ) );

	return pow(2.718, 1 * Weight[First_City][Second_City]);
}

// Indentify the promising cities as candidates which are possible to connect to Cur_City
void Temp_Identify_Promising_City()
{
	Promising_City_Num = 0;
	for (int i = 0; i < Virtual_City_Num; i++)
	{
		if (If_City_Selected[i] == true)
			continue;

		Promising_City[Promising_City_Num++] = i;
	}
}

// Set the probability (stored in Probabilistic[]) of selecting each candidate city (proportion to the potential of the corresponding edge)
bool Temp_Get_Probabilistic(int Cur_City)
{
	if (Promising_City_Num == 0)
		return false;

	double Total_Potential = 0;
	for (int i = 0; i < Promising_City_Num; i++)
		Total_Potential += Temp_Get_Potential(Cur_City, Promising_City[i]);

	Probabilistic[0] = (int)(1000 * Temp_Get_Potential(Cur_City, Promising_City[0]) / Total_Potential);
	for (int i = 1; i < Promising_City_Num - 1; i++)
		Probabilistic[i] = Probabilistic[i - 1] + (int)(1000 * Temp_Get_Potential(Cur_City, Promising_City[i]) / Total_Potential);
	Probabilistic[Promising_City_Num - 1] = 1000;

	return true;
}

// Probabilistically choose a city, controled by the values stored in Probabilistic[]
int Temp_Probabilistic_Get_City_To_Connect()
{
	int Random_Num = Get_Random_Int(1000);
	for (int i = 0; i < Promising_City_Num; i++)
		if (Random_Num < Probabilistic[i])
			return Promising_City[i];

	return Null;
}

// The whole process of choosing a city (a_{i+1} in the paper) to connect Cur_City (b_i in the paper)
int Temp_Choose_City_To_Connect(int Cur_City)
{
	// Avg_Weight=Get_Avg_Weight(Cur_City);
	Temp_Identify_Promising_City();
	Temp_Get_Probabilistic(Cur_City);

	return Temp_Probabilistic_Get_City_To_Connect();
}

bool Generate_Initial_Solution()
{
	for (int i = 0; i < Virtual_City_Num; i++)
	{
		Solution[i] = Null;
		If_City_Selected[i] = false;
	}

	int Selected_City_Num = 0;
	int Cur_City = Start_City;
	int Next_City;

	Solution[Selected_City_Num++] = Cur_City;
	If_City_Selected[Cur_City] = true;
	do
	{
		// Next_City=Select_Random_City(Cur_City);
		Next_City = Temp_Choose_City_To_Connect(Cur_City);
		if (Next_City != Null)
		{
			Solution[Selected_City_Num++] = Next_City;
			If_City_Selected[Next_City] = true;
			Cur_City = Next_City;
		}
	} while (Next_City != Null);

	Convert_Solution_To_All_Node();

	if (Check_Solution_Feasible() == false)
	{
		cout << "\nError! The constructed solution is unfeasible" << endl;
		getchar();
		return false;
	}

	return true;
}
