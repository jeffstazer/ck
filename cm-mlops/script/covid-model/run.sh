#!/bin/bash -l
#!/bin/bash

CM_TMP_CURRENT_SCRIPT_PATH=${CM_TMP_CURRENT_SCRIPT_PATH:-$PWD}
#export VAR=abc

#echo "Activate conda environment"
#conda activate ${CM_ENV_CONDA}

#sudo chgrp -R cm /home/covid-model-results
if [ ! -z ${CM_DOCKER_GROUP} ] && [ ! -z ${CM_PREFIX} ]; then
	sudo chgrp -R ${CM_DOCKER_GROUP} ${CM_PREFIX}
fi

cd ${CM_TMP_CURRENT_SCRIPT_PATH}
echo "Install Bayesian Model..."
pip install --upgrade pip && pip install -e .
cd scripts
echo "Run Model..."
echo "state: ${CM_ENV_STATE}"
echo "start: ${CM_ENV_START}"
echo "end: ${CM_ENV_END}"
if [ ! -z ${CM_DOCKER_GROUP} ] && [ ! -z ${CM_PREFIX} ]; then
	echo "prefix: ${CM_PREFIX}"
	python run_sir.py ${CM_ENV_STATE} --start ${CM_ENV_START} --end ${CM_ENV_END} --prefix ${CM_PREFIX}
else
	python run_sir.py ${CM_ENV_STATE} --start ${CM_ENV_START} --end ${CM_ENV_END}
fi
#conda deactivate

test $? -eq 0 || exit $?

echo "CM_NEW_VAR_FROM_RUN=$MLPERF_XYZ" > tmp-run-env.out


