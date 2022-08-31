#!/bin/bash
# Run an embarrassingly parallel job, where each command is totally independant
# Uses gnu parallel as a task scheduler, then executes each task on the available cpus with pbsdsh
# express

#PBS -q normal
#PBS -l ncpus=4
#PBS -l walltime=0:10:00
#PBS -l mem=2gb
#PBS -l wd

module load parallel

SCRIPT=./open_xarray_and_do_something.sh  # Script to run.
INPUTS=inputs.txt   # Each line in this file is used as arguments to ${SCRIPT}
                    # It's fine to have more input lines than you have requested cpus,
                    # extra jobs will be executed as cpus become available

# Here '{%}' gets replaced with the job slot ({1..$PBS_NCPUS})
# and '{}' gets replaced with a line from ${INPUTS}
parallel -j ${PBS_NCPUS} pbsdsh -n {%} -- /bin/bash -l -c "${SCRIPT} {}" :::: ${INPUTS}