davisdir=../../database/DAVIS/
res=Full-Resolution
seqname=$1
testres=1

#rm -rf ./$seqname 
rm -rf $davisdir/FlowFW*/$res/${seqname}* 
rm -rf $davisdir/FlowBW*/$res/${seqname}* 

array=(1 2 4 8 16 32)
for i in "${array[@]}"
do
python auto_gen.py --datapath $davisdir/JPEGImages/$res/$seqname/ --loadmodel ../../lasr_vcn/vcn_rob.pth  --testres $testres --dframe $i
mkdir -p $davisdir/FlowFW_$i/$res/$seqname
mkdir -p $davisdir/FlowBW_$i/$res/$seqname
cp $seqname/FlowFW_$i/* -rf $davisdir/FlowFW_$i/$res/$seqname
cp $seqname/FlowBW_$i/* -rf $davisdir/FlowBW_$i/$res/$seqname
done
