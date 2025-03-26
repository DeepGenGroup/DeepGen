#! /bin/bash
temp=$(dirname "$0")
taskfile=$1
cd ${temp}/..
project_dir=`pwd`
echo "ProjectDir="$project_dir ; cd ${project_dir} 
mkdir ${project_dir}/_cluster_run
python ${project_dir}/Runtime/kcg/startup_cluster_tasks.py ${taskfile}
