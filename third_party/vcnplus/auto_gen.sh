indavisdir=../database/DAVIS/
outdavisdir=../database/DAVIS/
#res=480p
res=Full-Resolution
seqname=$1
newname=r${seqname}
#newname=r${seqname}
refname=no
testres=1
array=(2 4 8)
outdir=/scratch/gengshany/Dropbox/

rm ./$seqname -rf
rm $outdavisdir/*/$res/${newname}* -rf
CUDA_VISIBLE_DEVICES=1 python auto_gen.py --datapath $indavisdir/JPEGImages/$res/$seqname/ --loadmodel ../tmp/vcn_rob.pth  --testres $testres --medflow 0

mkdir $outdavisdir/JPEGImages/$res/$newname
mkdir $outdavisdir/Annotations/$res/$newname
mkdir $outdavisdir/FlowFW/$res/$newname
mkdir $outdavisdir/FlowBW/$res/$newname
mkdir $outdavisdir/Depth/$res/$newname
cp $seqname/JPEGImages/*   -rf     $outdavisdir/JPEGImages/$res/$newname
cp $seqname/Annotations/* -rf      $outdavisdir/Annotations/$res/$newname
cp $seqname/FlowFW/*           -rf $outdavisdir/FlowFW/$res/$newname
cp $seqname/FlowBW/*           -rf $outdavisdir/FlowBW/$res/$newname
cp $seqname/Depth/*           -rf $outdavisdir/Depth/$res/$newname
rm ./$seqname -rf



for i in "${array[@]}"
do
CUDA_VISIBLE_DEVICES=1 python skip_gen.py --datapath $outdavisdir/JPEGImages/$res/$newname/ --loadmodel ../tmp/vcn_rob.pth  --testres $testres --dframe $i
done

#CUDA_VISIBLE_DEVICES=1 python semantic_flow.py --datapath $outdavisdir/JPEGImages/$res/$newname/ --refname $refname

#cd $outdavisdir
#rm  $outdir/$newname.zip
#zip $outdir/$newname.zip -r ./*/$res/${newname}*
#cd -
