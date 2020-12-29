
extern "C"

//must be same as threads!!!
//Block_Size = blockDim.x
#define Block_Size 64
#define m 0.001/2000

#define PI 3.14159265359f

__global__ void ker_rho(float *out, const float *x, const int *ind, const float h)
{
    //int IND = gridDim.z * gridDim.y * blockIdx.x + gridDim.z * blockIdx.y + blockIdx.z

    int istart = ind[2 * gridDim.z * gridDim.y * blockIdx.x + 2 * gridDim.z * blockIdx.y + 2 * blockIdx.z + 0];
    int iend = ind[2 * gridDim.z * gridDim.y * blockIdx.x + 2 * gridDim.z * blockIdx.y + 2 * blockIdx.z + 1];

    for (int i = istart; i < iend; i += Block_Size)
    {
        int id = i + threadIdx.x;

        float xi[3];

        if (id < iend)
        {
            xi[0] = x[3 * id + 0];
            xi[1] = x[3 * id + 1];
            xi[2] = x[3 * id + 2];
        }

        float dx[3];
        float r2;
        float r;
        float W;

        float rho = 0;

        __shared__ float xj[Block_Size * 3];

        for (int a = -1; a < 2; a++)
        {
            for (int b = -1; b < 2; b++)
            {
                if ((int)blockIdx.x + a < 0 || (int)blockIdx.x + a >= (int)gridDim.x || (int)blockIdx.y + b < 0 || (int)blockIdx.y + b >= (int)gridDim.y)
                {
                    continue;
                }

                int Zstart = max((int)blockIdx.z - 1, 0);
                int Zend = min((int)blockIdx.z + 1, (int)gridDim.z - 1);

                int jstart = ind[2 * gridDim.z * gridDim.y * (blockIdx.x+a) + 2 * gridDim.z * (blockIdx.y+b) + 2 * Zstart + 0];
                int jend = ind[2 * gridDim.z * gridDim.y * (blockIdx.x+a) + 2 * gridDim.z * (blockIdx.y+b) + 2 * Zend + 1];

                for (int j = jstart; j < jend; j += Block_Size)
                {
                    int jd = j + threadIdx.x;
                    if (jd < jend)
                    {
                        xj[3 * threadIdx.x + 0] = x[3 * jd + 0];
                        xj[3 * threadIdx.x + 1] = x[3 * jd + 1];
                        xj[3 * threadIdx.x + 2] = x[3 * jd + 2];
                    }
                    __syncthreads();
                    if (id < iend)
                    {
                        for (int k = 0; k < Block_Size; k++)
                        {
                            if (j + k < jend)
                            {
                                dx[0] = xj[3 * k + 0] - xi[0];
                                dx[1] = xj[3 * k + 1] - xi[1];
                                dx[2] = xj[3 * k + 2] - xi[2];

                                r2 = (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]) / (h * h);
                                if (r2 < 1.0)
                                {
                                    r = sqrtf(r2+0.001*h*h);

                                    W = (1.0-r);
                                    W*=W;
                                    W*=W;//(1-r)^4
                                    W*=(1+4.0*r)*21.0/(2.0*PI*h*h*h);

                                    //Wendland
                                    //dW = (1.0 - r);
                                    //dW *= dW*dW; //(1-r)^3
                                    //dW *= -5*r; 
                                    //dW *= 21.0 / (16.0 * PI * h * h * h * h);
                                    
                                    rho += m*W;
                                }
                            }
                        }
                    }
                    __syncthreads();
                }
                //ivol = 2 * gridDim.z * gridDim.y * blockIdx.x + 2 * gridDim.z * blockIdx.y + 2 * Zend;
            }
        }

        if (id < iend)
        {
            out[id] = rho;
        }
    }
}
