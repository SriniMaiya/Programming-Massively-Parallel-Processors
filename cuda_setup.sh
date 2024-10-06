wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run
sudo sh cuda_12.6.2_560.35.03_linux.run
# Setup and install the nvcc compiler in the installer.
echo "export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}" >> ~/.zshrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.zshrc