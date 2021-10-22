seqname=$1
model_path=$2
vid_id=$3
args=$4

# Set space as the delimiter
IFS=' '

#Read the split words into an array based on space delimiter
read -a strarr <<< "$vid_id"

for vid in "${strarr[@]}"; do
echo $vid

bash scripts/render_result.sh $seqname \
                    $model_path \
                    {$vid} \
                    $args  # for python script
done
