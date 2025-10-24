source /etc/profile.d/modules.sh
module --force purge
module use $OTHERSTAGES

module load Stages/2024
module load GCC OpenMPI
module load Python/3.11.3

# Some base modules commonly used in AI
module load mpi4py numba tqdm matplotlib IPython SciPy-Stack bokeh git
module load Flask Seaborn

# ML Frameworks
module load PyTorch scikit-learn torchvision PyTorch-Lightning cuDNN
module load tensorboard
module load h5py