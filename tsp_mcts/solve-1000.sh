#!/bin/bash
# author: Please make the following modifications before running this script

# code/TSP_IO.h (line 17): #define Max_Inst_Num 0 ==> #define Max_Inst_Num 128
# code/TSP_IO.h (line 339): string prefix_str = "/path/to/your/heatmap/tsp"; ==> string prefix_str = "/your/own/output_path/to/heatmap/tsp";

rm -r ./code/TSP.o
rm -r ./test
make

Temp_City_Num=1000
Total_Instance_Num=128
threads=16
Inst_Num_Per_Batch=$((Total_Instance_Num / threads))

python -u convert_numpy_to_txt.py \
    --heatmap_dir "/some/path/like/tsp/models/tsp_diffusion/wandb_id" \
    --output_dir "/your/own/output_path" \
    --num_nodes $Temp_City_Num \
    --num_files $Total_Instance_Num \
    --expected_valid_prob 0.01

for ((i = 0; i < $threads; i++)); do
    {
        touch ./results/${Temp_City_Num}/result_${i}.txt
        ./test "$i" ./results/${Temp_City_Num}/result_${i}.txt ./tsp${Temp_City_Num}_test_concorde.txt ${Temp_City_Num} ${Inst_Num_Per_Batch}
    } &
done
wait

echo "Done."
