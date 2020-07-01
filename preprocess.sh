#!/usr/bin/env bash
#
#SBATCH --job-name=mni
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH -o ./test1bisbisbis.out
#SBATCH --exclusive 

FILES=/om4/group/gablab/data/datalad/openneuro/ds000224/derivatives/surface_pipeline/sub-MSC*/processed_restingstate_timecourses/ses-func*/cifti/sub-MSC*_ses-func*_task-rest_bold_32k_fsLR.dtseries.nii

for file in $FILES
do
 
  new_file=$(echo "$file" | cut -f 1 -d '.')
  new_extension="_2.dtseries.nii"
  new_file2="$new_file$new_extension"
  echo "$new_file2"

  if [ -e "$new_path" ]
  then 
	echo "$new_path"
  else

  	wb_command -file-convert -cifti-version-convert $file 2 $new_file2
  fi
done

echo End

