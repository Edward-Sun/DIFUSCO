#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <ctime>
#include <vector>
#include <string.h>
#include <math.h>
using namespace std;

#define Null -1
#define Inf_Cost 1000000000
#define Magnify_Rate 10000
#define Max_Inst_Num 0 // to be modified
#define Max_City_Num 10000
#define Max_Candidate_Num 1000
#define Max_Depth 10

// Hyper parameters
double Alpha = 1;      // used in estimating the potential of each edge
double Beta = 10;      // used in back propagation
double Param_H = 10;   // used to control the number of sampling actions
double Param_T = 0.10; // used to control the termination condition

// #define Default_Random_Seed  (unsigned)time(NULL);
#define Default_Random_Seed 489663920;
unsigned Random_Seed = Default_Random_Seed;

typedef int Distance_Type;

/* 2020-02-11 */
char *Input_File_Name;
// char *Input_File_Name="tsp10000_test_concorde.txt";
char Heatmap_File_Name[100];

/* 2020-02-11 */
int Temp_City_Num;
// int Temp_City_Num=10000;
double Stored_Coordinates_X[Max_Inst_Num][Max_City_Num];
double Stored_Coordinates_Y[Max_Inst_Num][Max_City_Num];
int Stored_Opt_Solution[Max_Inst_Num][Max_City_Num];

/* 2020-02-11 */
int Inst_Num_Per_Batch;
// int Inst_Num_Per_Batch=1+Total_Instance_Num/32;
// int Inst_Num_Per_Batch=32;
// int Inst_Num_Per_Batch=16;
int Index_In_Batch = 0;

/*
//Information of the 38 TSPLIB instances
char *Input_Inst_File_Name="Instances-To-Run.txt";
char Instance_Name[Max_Inst_Num][100];
Distance_Type  Best_Known[Max_Inst_Num];
Distance_Type Best_Known_Result;
char Input_Inst_Name[100];
*/

// Used to store the statistic results
char *Statistics_File_Name = "Statistic_Results.txt";
int Test_Inst_Num = 0;
int Beat_Best_Known_Times = 0;
int Match_Best_Known_Times = 0;
int Miss_Best_Known_Times = 0;
double Sum_Opt_Distance = 0;
double Sum_My_Distance = 0;
double Sum_Gap = 0;

// Used to store the input information of a given instance
int City_Num;
int Start_City;
int Salesman_Num; // This program was proposed for the multiple TSP. If Salesman_Num=1, it reduces to the TSP
int Virtual_City_Num;
double *Coordinate_X;
double *Coordinate_Y;
Distance_Type **Distance;

double Current_Instance_Begin_Time;
Distance_Type Current_Instance_Best_Distance;

// Used to store a solution in double link
struct Struct_Node
{
    int Pre_City;
    int Next_City;
    int Salesman;
};

Struct_Node *All_Node;      // Store the incumbent tour
Struct_Node *Best_All_Node; // Store the best found tour

// Used to store a solution in an array
int *Solution;

// Used to store a set of candidate neighbors of each city
int *Candidate_Num;
int **Candidate;
bool *If_City_Selected;

// Used to store the information of an action
int Pair_City_Num;
int Temp_Pair_Num;
int *City_Sequence;
int *Temp_City_Sequence;
Distance_Type *Gain;
Distance_Type *Real_Gain;

// Used in MCTS
double **Edge_Heatmap;
double **Weight;
double Avg_Weight;
int **Chosen_Times;
int *Promising_City;
int *Probabilistic;
int Promising_City_Num;
int Total_Simulation_Times;

Distance_Type Get_Solution_Total_Distance();
void Convert_Solution_To_All_Node();

void Allocate_Memory(int City_Num)
{
    Coordinate_X = new double[City_Num];
    Coordinate_Y = new double[City_Num];

    Distance = new Distance_Type *[City_Num];
    for (int i = 0; i < City_Num; i++)
        Distance[i] = new Distance_Type[City_Num];

    All_Node = new Struct_Node[City_Num];
    Best_All_Node = new Struct_Node[City_Num];
    Solution = new int[City_Num];

    Candidate_Num = new int[City_Num];
    Candidate = new int *[City_Num];
    for (int i = 0; i < City_Num; i++)
        Candidate[i] = new int[Max_Candidate_Num];
    If_City_Selected = new bool[City_Num];

    City_Sequence = new int[City_Num];
    Temp_City_Sequence = new int[City_Num];
    Gain = new Distance_Type[2 * City_Num];
    Real_Gain = new Distance_Type[2 * City_Num];

    Edge_Heatmap = new double *[City_Num];
    for (int i = 0; i < City_Num; i++)
        Edge_Heatmap[i] = new double[City_Num];

    Weight = new double *[City_Num];
    for (int i = 0; i < City_Num; i++)
        Weight[i] = new double[City_Num];

    Chosen_Times = new int *[City_Num];
    for (int i = 0; i < City_Num; i++)
        Chosen_Times[i] = new int[City_Num];

    Promising_City = new int[City_Num];
    Probabilistic = new int[City_Num];
}

void Release_Memory(int City_Num)
{
    delete[] Coordinate_X;
    delete[] Coordinate_Y;

    for (int i = 0; i < City_Num; i++)
        delete[] Distance[i];
    delete[] Distance;

    delete[] All_Node;
    delete[] Best_All_Node;
    delete[] Solution;

    delete[] Candidate_Num;
    for (int i = 0; i < City_Num; i++)
        delete[] Candidate[i];
    delete[] Candidate;
    delete[] If_City_Selected;

    delete[] City_Sequence;
    delete[] Temp_City_Sequence;
    delete[] Gain;
    delete[] Real_Gain;

    for (int i = 0; i < City_Num; i++)
        delete[] Edge_Heatmap[i];
    delete[] Edge_Heatmap;

    for (int i = 0; i < City_Num; i++)
        delete[] Weight[i];
    delete[] Weight;

    for (int i = 0; i < City_Num; i++)
        delete[] Chosen_Times[i];
    delete[] Chosen_Times;

    delete[] Promising_City;
    delete[] Probabilistic;
}

// For TSP20-50-100 instances
bool Fetch_Stored_Instance_Info(int Inst_Index)
{
    City_Num = Temp_City_Num;
    Start_City = 0;
    Salesman_Num = 1;
    Virtual_City_Num = City_Num + Salesman_Num - 1;

    Allocate_Memory(Virtual_City_Num);

    for (int i = 0; i < City_Num; i++)
    {
        Coordinate_X[i] = Stored_Coordinates_X[Inst_Index][i] * Magnify_Rate;
        Coordinate_Y[i] = Stored_Coordinates_Y[Inst_Index][i] * Magnify_Rate;
    }

    return true;
}

bool Save_Solution(char *Output_File_Name)
{
    FILE *fp;
    fp = fopen(Output_File_Name, "w+");

    fprintf(fp, "%d\n", City_Num);
    fprintf(fp, "%d\n", Start_City + 1);
    fprintf(fp, "%d\n", Salesman_Num);

    int Cur_City = Start_City;
    do
    {
        fprintf(fp, "%d\n ", Cur_City + 1);
        Cur_City = All_Node[Cur_City].Next_City;
    } while (Cur_City != Null && Cur_City != Start_City);

    fprintf(fp, "%d\n ", Get_Solution_Total_Distance());

    fclose(fp);
    return true;
}

bool Read_Solution(char *Input_File_Name)
{
    ifstream FIC;
    FIC.open(Input_File_Name);

    if (FIC.fail())
    {
        cout << "Read_Solution() fail! Fail to open the input file " << Input_File_Name << endl;
        return false;
    }

    FIC >> City_Num;
    FIC >> Start_City;
    Start_City--;
    FIC >> Salesman_Num;

    Virtual_City_Num = City_Num + Salesman_Num - 1;

    int Cur_City;
    for (int i = 0; i < Virtual_City_Num; i++)
    {
        FIC >> Cur_City;
        Solution[i] = Cur_City - 1;
    }
    FIC.close();

    Convert_Solution_To_All_Node();
    // cout<<"Read_Solution() finished. Begin to search. "<<endl<<endl;
    return true;
}

// Print the cities of a solution one by one
void Print_TSP_Tour(int Begin_City)
{
    cout << "\nThe current tour is:" << endl;
    int Cur_City = Begin_City;
    do
    {
        printf("%d ", Cur_City + 1);
        Cur_City = All_Node[Cur_City].Next_City;
    } while (Cur_City != Null && Cur_City != Begin_City);
}

// Modified for ICML
void Set_Heapmap_Fine_Name(int Inst_Index)
{
    int len = 0;
    string prefix_str = "/path/to/your/heatmap/tsp"; // to be modified

    for (int i = 0; i < prefix_str.length(); i++)
    {
        Heatmap_File_Name[len++] = prefix_str[i];
    }

    if (Temp_City_Num == 20)
    {
        Heatmap_File_Name[len++] = '2';
        Heatmap_File_Name[len++] = '0';
    }
    else if (Temp_City_Num == 50)
    {
        Heatmap_File_Name[len++] = '5';
        Heatmap_File_Name[len++] = '0';
    }
    else if (Temp_City_Num == 100)
    {
        Heatmap_File_Name[len++] = '1';
        Heatmap_File_Name[len++] = '0';
        Heatmap_File_Name[len++] = '0';
    }
    else if (Temp_City_Num == 200)
    {
        Heatmap_File_Name[len++] = '2';
        Heatmap_File_Name[len++] = '0';
        Heatmap_File_Name[len++] = '0';
    }
    else if (Temp_City_Num == 500)
    {
        Heatmap_File_Name[len++] = '5';
        Heatmap_File_Name[len++] = '0';
        Heatmap_File_Name[len++] = '0';
    }
    else if (Temp_City_Num == 1000)
    {
        Heatmap_File_Name[len++] = '1';
        Heatmap_File_Name[len++] = '0';
        Heatmap_File_Name[len++] = '0';
        Heatmap_File_Name[len++] = '0';
    }
    else if (Temp_City_Num == 10000)
    {
        Heatmap_File_Name[len++] = '1';
        Heatmap_File_Name[len++] = '0';
        Heatmap_File_Name[len++] = '0';
        Heatmap_File_Name[len++] = '0';
        Heatmap_File_Name[len++] = '0';
    }

    Heatmap_File_Name[len++] = '/';
    Heatmap_File_Name[len++] = 'h';
    Heatmap_File_Name[len++] = 'e';
    Heatmap_File_Name[len++] = 'a';
    Heatmap_File_Name[len++] = 't';
    Heatmap_File_Name[len++] = 'm';
    Heatmap_File_Name[len++] = 'a';
    Heatmap_File_Name[len++] = 'p';
    Heatmap_File_Name[len++] = 't';
    Heatmap_File_Name[len++] = 's';
    Heatmap_File_Name[len++] = 'p';

    if (Temp_City_Num == 20)
    {
        Heatmap_File_Name[len++] = '2';
        Heatmap_File_Name[len++] = '0';
    }
    else if (Temp_City_Num == 50)
    {
        Heatmap_File_Name[len++] = '5';
        Heatmap_File_Name[len++] = '0';
    }
    else if (Temp_City_Num == 100)
    {
        Heatmap_File_Name[len++] = '1';
        Heatmap_File_Name[len++] = '0';
        Heatmap_File_Name[len++] = '0';
    }
    else if (Temp_City_Num == 200)
    {
        Heatmap_File_Name[len++] = '2';
        Heatmap_File_Name[len++] = '0';
        Heatmap_File_Name[len++] = '0';
    }
    else if (Temp_City_Num == 500)
    {
        Heatmap_File_Name[len++] = '5';
        Heatmap_File_Name[len++] = '0';
        Heatmap_File_Name[len++] = '0';
    }
    else if (Temp_City_Num == 1000)
    {
        Heatmap_File_Name[len++] = '1';
        Heatmap_File_Name[len++] = '0';
        Heatmap_File_Name[len++] = '0';
        Heatmap_File_Name[len++] = '0';
    }
    else if (Temp_City_Num == 10000)
    {
        Heatmap_File_Name[len++] = '1';
        Heatmap_File_Name[len++] = '0';
        Heatmap_File_Name[len++] = '0';
        Heatmap_File_Name[len++] = '0';
        Heatmap_File_Name[len++] = '0';
    }

    Heatmap_File_Name[len++] = '_';

    bool if_Met_Not_Zero = false;
    int temp_Index = Inst_Index;

    int temp_Data = temp_Index / 10000;
    if (temp_Data > 0)
    {
        Heatmap_File_Name[len++] = '0' + temp_Data;
        if_Met_Not_Zero = true;
    }
    else if (temp_Data == 0 && if_Met_Not_Zero == true)
        Heatmap_File_Name[len++] = '0' + temp_Data;

    temp_Index = temp_Index % 10000;

    temp_Data = temp_Index / 1000;
    if (temp_Data > 0)
    {
        Heatmap_File_Name[len++] = '0' + temp_Data;
        if_Met_Not_Zero = true;
    }
    else if (temp_Data == 0 && if_Met_Not_Zero == true)
        Heatmap_File_Name[len++] = '0' + temp_Data;

    temp_Index = temp_Index % 1000;

    temp_Data = temp_Index / 100;
    if (temp_Data > 0)
    {
        Heatmap_File_Name[len++] = '0' + temp_Data;
        if_Met_Not_Zero = true;
    }
    else if (temp_Data == 0 && if_Met_Not_Zero == true)
        Heatmap_File_Name[len++] = '0' + temp_Data;

    temp_Index = temp_Index % 100;

    temp_Data = temp_Index / 10;
    if (temp_Data > 0)
    {
        Heatmap_File_Name[len++] = '0' + temp_Data;
        if_Met_Not_Zero = true;
    }
    else if (temp_Data == 0 && if_Met_Not_Zero == true)
        Heatmap_File_Name[len++] = '0' + temp_Data;

    temp_Index = temp_Index % 10;

    temp_Data = temp_Index;
    Heatmap_File_Name[len++] = '0' + temp_Data;

    Heatmap_File_Name[len++] = '.';
    Heatmap_File_Name[len++] = 't';
    Heatmap_File_Name[len++] = 'x';
    Heatmap_File_Name[len++] = 't';
    Heatmap_File_Name[len++] = '\0';

    // printf("%s\n", Heatmap_File_Name);
}

// Modified for ICML
bool Read_Heatmap()
{
    ifstream FIC;
    FIC.open(Heatmap_File_Name);

    if (FIC.fail())
    {
        cout << "\n\nError! Fail to open heatmap file" << Heatmap_File_Name << endl;
        getchar();
        return false;
    }

    int temp_Num;
    FIC >> temp_Num;

    double temp_Weight;
    for (int i = 0; i < Temp_City_Num; i++)
    {
        for (int j = 0; j < Temp_City_Num; j++)
        {
            FIC >> temp_Weight;
            Edge_Heatmap[i][j] = temp_Weight;
        }
    }

    for (int i = 0; i < Temp_City_Num; i++)
        for (int j = i + 1; j < Temp_City_Num; j++)
        {
            Edge_Heatmap[i][j] = (Edge_Heatmap[i][j] + Edge_Heatmap[j][i]) / 2;
            Edge_Heatmap[j][i] = Edge_Heatmap[i][j];
        }
}
