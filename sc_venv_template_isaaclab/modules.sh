module purge
module load Stages/2025
module load Python/3.11.3
module load GCC OpenMPI
# Some base modules commonly used in AI
module load mpi4py numba tqdm matplotlib IPython SciPy-Stack bokeh git
module load Flask Seaborn

# ML Frameworks
module load PyTorch scikit-learn torchvision PyTorch-Lightning cuDNN/9
module load tensorboard
module load h5py