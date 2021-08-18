import os
import submitit
import sys
import pdb

partition='devlab'
#partition='learnlab'
executor = submitit.AutoExecutor(folder="/private/home/gengshany/newStart")  # submission interface (logs are dumped in the folder)

cmd_str = "bash"
for arg in sys.argv[1:]:
    cmd_str = "%s %s"%(cmd_str, arg)
#cmd_str = f'screen -dmS "viser" bash -c ". activate viser; %s"'%(cmd_str)
with open(sys.argv[1]) as f:
    lines = f.readlines()
ngpu = [int(l.split('ngpu=')[-1]) for l in lines if 'ngpu=' in l][0]

# divide gpus
numgpu=8
nodes=ngpu//numgpu
if ngpu%numgpu!=0:nodes+=1
if nodes==1:
    numgpu=ngpu

pdb.set_trace()

def run_cmd(run_cmd):
    os.system(run_cmd)

    
executor.update_parameters(timeout_min=600, nodes=nodes, gpus_per_node=numgpu, cpus_per_task=8*numgpu, slurm_partition=partition, slurm_comment= '', name="text")  # timeout in min
#run_cmd(cmd_str )
job = executor.submit(run_cmd,cmd_str)
