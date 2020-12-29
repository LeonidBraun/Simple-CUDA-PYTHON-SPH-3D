
extern "C"

//must be same as threads!!!
//Block_Size = blockDim.x
#define Block_Size 64

#define m 0.001/2000
#define c 10
#define rho0 1
#define p0 1
#define gamma 7

#define PI 3.14159265359f

//__device__ float calc_p(float rho){
//    return c*c*rho0*(powf(rho/rho0,gamma) -1)/gamma+p0;
//}

__global__ void ker_dv(float *out, const float *x, const float *v, const float *rho, const int *ind, const float h)
{
    //int IND = gridDim.z * gridDim.y * blockIdx.x + gridDim.z * blockIdx.y + blockIdx.z

    int istart = ind[2 * gridDim.z * gridDim.y * blockIdx.x + 2 * gridDim.z * blockIdx.y + 2 * blockIdx.z + 0];
    int iend = ind[2 * gridDim.z * gridDim.y * blockIdx.x + 2 * gridDim.z * blockIdx.y + 2 * blockIdx.z + 1];

    for (int i = istart; i < iend; i += Block_Size)
    {
        int id = i + threadIdx.x;

        float xi[3];
        float vi[3];
        float rhoi;

        if (id < iend)
        {
            xi[0] = x[3 * id + 0];
            xi[1] = x[3 * id + 1];
            xi[2] = x[3 * id + 2];
            vi[0] = v[3 * id + 0];
            vi[1] = v[3 * id + 1];
            vi[2] = v[3 * id + 2];
            rhoi = rho[id];
        }

        float dx[3];
        float dv[3];
        float r2;
        float r;
        float dW;

        //float pi = calc_p(rhoi);
        float pi = c*c*rho0*(powf(rhoi/rho0,gamma) -1)/gamma+p0;
        float pj;

        float dV[3] = {0,0,0};

        __shared__ float xj[Block_Size * 3];
        __shared__ float vj[Block_Size * 3];
        __shared__ float rhoj[Block_Size];


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
                        vj[3 * threadIdx.x + 0] = v[3 * jd + 0];
                        vj[3 * threadIdx.x + 1] = v[3 * jd + 1];
                        vj[3 * threadIdx.x + 2] = v[3 * jd + 2];
                        rhoj[threadIdx.x] = rho[jd];
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

                                dv[0] = vj[3 * k + 0] - vi[0];
                                dv[1] = vj[3 * k + 1] - vi[1];
                                dv[2] = vj[3 * k + 2] - vi[2];

                                r2 = (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]) / (h * h);
                                if (r2 < 1.0)
                                {
                                    r = sqrtf(r2+0.001*h*h);

                                    dW = (1.0 - r);
                                    dW *= dW*dW; //(1-r)^3
                                    dW = -dW*r*20; 
                                    dW *= 21.0 / (2.0 * PI * h * h * h * h);
                                    
                                    //pj = calc_p(rhoj[k]);
                                    pj = c*c*rho0*(powf(rhoj[k]/rho0,gamma) -1)/gamma+p0;
                                    
                                    float d = ( pi/(rhoi*rhoi) + pj/(rhoj[k]*rhoj[k]) );
                                    
                                    d -= 2*(dv[0]*dx[0]+dv[1]*dx[1]+dv[2]*dx[2])/((r2+0.001*h*h)*rhoj[k]*rhoi);
                                    
                                    d *= m*dW/r;

                                    dV[0] += d*dx[0];
                                    dV[1] += d*dx[1];
                                    dV[2] += d*dx[2];
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
            out[3*id+0] = dV[0] + 10*c*c*fmaxf(0.0,-xi[0])/h - 10*c*c*fmaxf(0.0,xi[0]-2)/h;
            out[3*id+1] = dV[1] + 10*c*c*fmaxf(0.0,-xi[1])/h - 10*c*c*fmaxf(0.0,xi[1]-1)/h;
            out[3*id+2] = dV[2] + 10*c*c*fmaxf(0.0,-xi[2])/h - 1;
        }
    }
}
