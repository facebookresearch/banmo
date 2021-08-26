indavisdir=../../database/DAVIS/
outdavisdir=../../database/DAVIS/
res=Full-Resolution
seqname=$1
testres=1

. activate viser
rm ./$seqname -rf
rm $outdavisdir/FlowFW/$res/${seqname}* -rf
rm $outdavisdir/FlowBW/$res/${seqname}* -rf
CUDA_VISIBLE_DEVICES=1 python auto_gen.py --datapath $indavisdir/JPEGImages/$res/$seqname/ --loadmodel ../../lasr_vcn/vcn_rob.pth  --testres $testres --medflow 0

mkdir $outdavisdir/FlowFW/$res/$seqname
mkdir $outdavisdir/FlowBW/$res/$seqname
cp $seqname/FlowFW/*           -rf $outdavisdir/FlowFW/$res/$seqname
cp $seqname/FlowBW/*           -rf $outdavisdir/FlowBW/$res/$seqname
rm ./$seqname -rf
