module purge
module load Stages/2024
module load GCCcore/.12.3.0 OpenMPI
module load Python/3.11.3

# Some base modules commonly used in AI
module load mpi4py numba tqdm matplotlib IPython SciPy-Stack bokeh git
module load Flask Seaborn

# ML Frameworks
module load PyTorch scikit-learn torchvision PyTorch-Lightning cuDNN/8.9.5.29-CUDA-12
module load tensorboard
module load h5py