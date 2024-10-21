#include <iostream>
#include <iomanip>

using std::cout, std::endl;

int main()
{
    int deviceCount;
    // Get number of available devices.
    cudaGetDeviceCount(&deviceCount);

    cudaDeviceProp deviceProperties[deviceCount];
    // Get device properties for the available devices.
    for (int i = 0; i < deviceCount; i++)
    {
        cudaGetDeviceProperties(&deviceProperties[i], i);
    }
    // Get the first device.
    cudaDeviceProp devProp_0 = deviceProperties[0];

    cout << std::setw(40) << std::left << "Maximum threads per block " << " : " << devProp_0.maxThreadsPerBlock << endl;
    cout << std::setw(40) << std::left << "Number of multiprocessors " << " : " << devProp_0.multiProcessorCount << endl;
    cout << std::setw(40) << std::left << "Max num blocks / multiprocessor" << " : " << devProp_0.maxBlocksPerMultiProcessor << endl;
    cout << std::setw(40) << std::left << "Clock frequency " << " : " << devProp_0.clockRate / 1e6 << " GHz" << endl;
    cout << "::::::::::::" << endl;
    cout << std::setw(40) << std::left << "Maximum threads/block @ x " << " : " << devProp_0.maxThreadsDim[0] << endl;
    cout << std::setw(40) << std::left << "Maximum threads/block @ y " << " : " << devProp_0.maxThreadsDim[1] << endl;
    cout << std::setw(40) << std::left << "Maximum threads/block @ z " << " : " << devProp_0.maxThreadsDim[2] << endl;
    cout << "::::::::::::" << endl;
    cout << std::setw(40) << std::left << "Maximum num block @ x " << " : " << devProp_0.maxGridSize[0] << endl;
    cout << std::setw(40) << std::left << "Maximum num block @ y " << " : " << devProp_0.maxGridSize[1] << endl;
    cout << std::setw(40) << std::left << "Maximum num block @ z " << " : " << devProp_0.maxGridSize[2] << endl;
    cout << "::::::::::::" << endl;
    cout << std::setw(40) << std::left << "Number of registers / block " << " : " << devProp_0.regsPerBlock << endl;
    cout << std::setw(40) << std::left << "Number of registers / multiprocessor " << " : " << devProp_0.regsPerMultiprocessor << endl;
    cout << std::setw(40) << std::left << "Number of threads / warp " << " : " << devProp_0.warpSize << endl;
}