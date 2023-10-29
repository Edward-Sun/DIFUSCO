
// Return an integer between [0,Divide_Num)
int Get_Random_Int(int Divide_Num)
{
  return rand()%Divide_Num;
}

//Calculate the distance between two cities, rounded up to the nearest integer
int Calculate_Int_Distance(int First_City,int Second_City)
{
  	return (int)(0.5 + sqrt( (Coordinate_X[First_City]-Coordinate_X[Second_City])*(Coordinate_X[First_City]-Coordinate_X[Second_City]) +
                (Coordinate_Y[First_City]-Coordinate_Y[Second_City])*(Coordinate_Y[First_City]-Coordinate_Y[Second_City]) ) );
}

//Calculate the distance between two cities
double Calculate_Double_Distance(int First_City,int Second_City)
{
  	return sqrt( (Coordinate_X[First_City]-Coordinate_X[Second_City])*(Coordinate_X[First_City]-Coordinate_X[Second_City]) +
                   (Coordinate_Y[First_City]-Coordinate_Y[Second_City])*(Coordinate_Y[First_City]-Coordinate_Y[Second_City]) );
}

// Calculate the distance (integer) between any two cities, stored in Distance[][]
void Calculate_All_Pair_Distance()
{
  	for(int i=0;i<Virtual_City_Num;i++)
  		for(int j=0;j<Virtual_City_Num;j++)
  		{
	  		if(i!=j)
	    		Distance[i][j]=Calculate_Int_Distance(i,j);
	  		else
	    		Distance[i][j]=Inf_Cost;
		}
}

// Fetch the distance (already stored in Distance[][]) between two cities
Distance_Type Get_Distance(int First_City,int Second_City)
{
	return Distance[First_City][Second_City];
}

// Using the information stored in Solution[] to update the information stored in Struct_Node *All_Node
void Convert_Solution_To_All_Node()
{
  	int Temp_Cur_City;
  	int Temp_Pre_City;
  	int Temp_Next_City;
  	int Cur_Salesman=0;

  	for(int i=0;i<Virtual_City_Num;i++)
  	{
  		Temp_Cur_City = Solution[i];
  		Temp_Pre_City = Solution [(i-1+Virtual_City_Num)%Virtual_City_Num];
  		Temp_Next_City = Solution [(i+1+Virtual_City_Num)%Virtual_City_Num];

  		if(Temp_Cur_City >= City_Num)
  	  		Cur_Salesman++;

  		All_Node[Temp_Cur_City].Pre_City=Temp_Pre_City;
  		All_Node[Temp_Cur_City].Next_City=Temp_Next_City;
		All_Node[Temp_Cur_City].Salesman=Cur_Salesman;
  	}
}

// Using the information stored in Struct_Node *All_Node to update the information stored in Solution[]
bool Convert_All_Node_To_Solution()
{
	for(int i=0;i<Virtual_City_Num;i++)
		Solution[i]=Null;

	int Cur_Index=0;
	Solution[Cur_Index]=Start_City;

	int Cur_City=Start_City;
	do
	{
		Cur_Index++;

		Cur_City=All_Node[Cur_City].Next_City;
		if(Cur_City == Null || Cur_Index >= Virtual_City_Num)
			return false;

		Solution[Cur_Index]=Cur_City;
	}while(All_Node[Cur_City].Next_City != Start_City);

	return true;
}

// Check the current solution stored in Struct_Node *All_Node is a feasible TSP tour
bool Check_Solution_Feasible()
{
	int Cur_City=Start_City;
	int Visited_City_Num=0;
	while(true)
	{
		Cur_City = All_Node[Cur_City].Next_City;
		if(Cur_City == Null)
		{
			printf("\nThe current solution is unvalid. Current city is Null\n");
			return false;
		}

		Visited_City_Num++;
		if(Visited_City_Num > Virtual_City_Num)
		{
			printf("\nThe current solution is unvalid. Loop may exist\n");
			getchar();
			return false;
		}

		if(Cur_City == Start_City && Visited_City_Num == Virtual_City_Num)
			return true;
	}
}

// Return the total distance (integer) of the solution stored in Struct_Node *All_Node
Distance_Type Get_Solution_Total_Distance()
{
  	Distance_Type Solution_Total_Distance=0;
  	for(int i=0;i<Virtual_City_Num;i++)
  	{
  		int Temp_Next_City=All_Node[i].Next_City;
  		if(Temp_Next_City != Null)
  	  		Solution_Total_Distance += Get_Distance(i,Temp_Next_City);
  		else
  		{
  			printf("\nGet_Solution_Total_Distance() fail!\n");
  			getchar();
  			return Inf_Cost;
		}
  	}

  	return Solution_Total_Distance;
}

//For TSP20-50-100 instances
// Return the total distance (double) of the solution stored in Stored_Opt_Solution[Inst_Index]
double Get_Stored_Solution_Double_Distance(int Inst_Index)
{
	double Stored_Solution_Double_Distance=0;
	for(int i=0;i<Virtual_City_Num-1;i++)
		Stored_Solution_Double_Distance += Calculate_Double_Distance(Stored_Opt_Solution[Inst_Index][i],Stored_Opt_Solution[Inst_Index][i+1]);

	Stored_Solution_Double_Distance += Calculate_Double_Distance(Stored_Opt_Solution[Inst_Index][Virtual_City_Num-1],Stored_Opt_Solution[Inst_Index][0]);
	return Stored_Solution_Double_Distance;
}

//For TSP20-50-100 instances
// Return the total distance (double) of the solution stored in Struct_Node *All_Node
double Get_Current_Solution_Double_Distance()
{
  	double Current_Solution_Double_Distance=0;
  	for(int i=0;i<Virtual_City_Num;i++)
  	{
  		int Temp_Next_City=All_Node[i].Next_City;
  		if(Temp_Next_City != Null)
  	  		Current_Solution_Double_Distance += Calculate_Double_Distance(i,Temp_Next_City);
  		else
  		{
  			printf("\nGet_Current_Solution_Double_Distance() fail!\n");
  			getchar();
  			return Inf_Cost;
		}
  	}

  	return Current_Solution_Double_Distance;
}


// Modified for ICML
// Return the unselected city neareast to Cur_City
int Get_Best_Unselected_City(int Cur_City)
{
	int Best_Unselected_City=Null;
	for(int i=0;i<Virtual_City_Num;i++)
	{
		if(i==Cur_City || If_City_Selected[i] || Get_Distance(Cur_City,i) >= Inf_Cost )
			continue;

		if(Best_Unselected_City == Null || Edge_Heatmap[Cur_City][i] > Edge_Heatmap[Cur_City][Best_Unselected_City])
			Best_Unselected_City=i;
	}

	if(Edge_Heatmap[Cur_City][Best_Unselected_City] >= 0.0001)
		return Best_Unselected_City;
	else
		return Null;
}

// Modified for ICML
// Identify a set of candidate neighbors for each city, stored in Candidate_Num[] and Candidate[][]
void Identify_Candidate_Set()
{
	for(int i=0;i<Virtual_City_Num;i++)
	{
		Candidate_Num[i]=0;

		for(int j=0;j<Virtual_City_Num;j++)
			If_City_Selected[j]=false;

		while(true)
		{
			int Best_Unselected_City=Get_Best_Unselected_City(i);
			if(Best_Unselected_City != Null)
			{
				Candidate[i][Candidate_Num[i]++]=Best_Unselected_City;
				If_City_Selected[Best_Unselected_City]=true;
			}
			else
				break;
		}
	}
}


bool Check_If_Two_City_Same_Or_Adjacent(int First_City, int Second_City)
{
	if(First_City==Second_City || All_Node[First_City].Next_City == Second_City || All_Node[Second_City].Next_City == First_City)
		return true;
	else
		return false;
}

// For each city between First_City and Second City, reverse its Pre_City and Next_City
void Reverse_Sub_Path(int First_City,int Second_City)
{
	int Cur_City=First_City;
	int Temp_Next_City=All_Node[Cur_City].Next_City;

	while(true)
	{
		int Temp_City = All_Node[Cur_City].Pre_City;
		All_Node[Cur_City].Pre_City=All_Node[Cur_City].Next_City;
		All_Node[Cur_City].Next_City=Temp_City;

		if(Cur_City==Second_City)
			break;

		Cur_City=Temp_Next_City;
		Temp_Next_City=All_Node[Cur_City].Next_City;
	}
}

// Copy information from Struct_Node *All_Node to Struct_Node *Best_All_Node
void Store_Best_Solution()
{
	for(int i=0;i<Virtual_City_Num;i++)
	{
		Best_All_Node[i].Salesman=All_Node[i].Salesman;
		Best_All_Node[i].Next_City=All_Node[i].Next_City;
		Best_All_Node[i].Pre_City=All_Node[i].Pre_City;
	}
}

// Copy information from Struct_Node *Best_All_Node to Struct_Node *All_Node
void Restore_Best_Solution()
{
	for(int i=0;i<Virtual_City_Num;i++)
	{
		All_Node[i].Salesman=Best_All_Node[i].Salesman;
		All_Node[i].Next_City=Best_All_Node[i].Next_City;
		All_Node[i].Pre_City=Best_All_Node[i].Pre_City;
	}
}
