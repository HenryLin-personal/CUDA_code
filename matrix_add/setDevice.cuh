#ifndef __SETDEVICE_CUH__
#define __SETDEVICE_CUH__
#include <cstdio>

void setDevice()
{
    int iDeviceCount = 0;
    cudaError_t ret = cudaGetDeviceCount(&iDeviceCount);
    if(ret != cudaSuccess || iDeviceCount == 0)
    {
        printf("There is no GPU!\n");
        exit(-1);
    }
    printf("The count of GPUs is %d\n", iDeviceCount);

    int iDev = 0;
    ret = cudaSetDevice(iDev);
    if(ret != cudaSuccess)
    {
        printf("cudaSetDevice fail!\n");
        exit(-1);
    }
    printf("Set GPU 0 as the device!\n");
}
#endif