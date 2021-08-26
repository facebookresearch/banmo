indavisdir=/data/gengshay/ytvos/processed/
outdavisdir=../database/DAVIS/
res=Full-Resolution
#seqname=elephant0005
array=(2 4)

for seqname in /data/gengshay/ytvos/processed/JPEGImages/Full-Resolution/*; do
  seqname="$(cut -d'/' -f8 <<<$seqname)"
  echo $seqname
  newname=r${seqname}
  
  rm ./$seqname -rf
  rm $outdavisdir/*/$res/${newname}* -rf
  CUDA_VISIBLE_DEVICES=1 python auto_gen.py --datapath $indavisdir/JPEGImages/$res/$seqname/ --loadmodel /data/gengshay/lasr_vcn/flow-rob-4th.pth  --testres -1 --medflow 0.

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

  for i in "${array[@]}"
  do
    CUDA_VISIBLE_DEVICES=1 python skip_gen.py --datapath $outdavisdir/JPEGImages/$res/$newname/ --loadmodel /data/gengshay/lasr_vcn/flow-rob-4th.pth  --testres -1 --dframe $i
  done

#  CUDA_VISIBLE_DEVICES=1 python semantic_flow.py --datapath $outdavisdir/JPEGImages/$res/$newname/ --refname no

  cd $outdavisdir
  rm /home/gengshay/$newname.zip
  zip /home/gengshay/$newname.zip -r ./*/$res/${newname}*
  cd -
  rm ./$seqname -rf
done
