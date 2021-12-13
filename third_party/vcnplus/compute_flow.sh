davisdir=../../database/DAVIS/
res=Full-Resolution
seqname=$1
testres=1

rm ./$seqname -Irf
rm $davisdir/FlowFW*/$res/${seqname}* -Irf
rm $davisdir/FlowBW*/$res/${seqname}* -Irf

array=(1 2 4 8 16 32)
for i in "${array[@]}"
do
python auto_gen.py --datapath $davisdir/JPEGImages/$res/$seqname/ --loadmodel ../../lasr_vcn/vcn_rob.pth  --testres $testres --dframe $i
mkdir -p $davisdir/FlowFW_$i/$res/$seqname
mkdir -p $davisdir/FlowBW_$i/$res/$seqname
cp $seqname/FlowFW_$i/* -Irf $davisdir/FlowFW_$i/$res/$seqname
cp $seqname/FlowBW_$i/* -Irf $davisdir/FlowBW_$i/$res/$seqname
done
