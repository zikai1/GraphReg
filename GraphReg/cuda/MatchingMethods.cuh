#include "CudaMath.cuh"

__device__ void Gravity(const float* p1, const float* p2, float* force) 
{
    float r[3];
    r[0] = p1[0] - p2[0];
    r[1] = p1[1] - p2[1];
    r[2] = p1[2] - p2[2];

    float dist = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    
	dist = dist*dist*dist;

	if(dist < 0.0000001)
		dist = 0.0000001;
	    

	force[0] = r[0] / dist;
	force[1] = r[1] / dist;
	force[2] = r[2] / dist;
	    
}

__device__ void Coulomb(const float* p1, const float* p2, const int numFeat, const float* f1, int idx1, const float* f2, int idx2, float* force) 
{
	float r[3];
    r[0] = p1[0] - p2[0];
    r[1] = p1[1] - p2[1];
    r[2] = p1[2] - p2[2];

    float dist = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);

    float cDist = 0.f;

    for(int i = 0; i < numFeat; i++)
    {
        float fd = abs(f1[idx1*numFeat + i] - f2[idx2*numFeat + i]);
        cDist += abs(fd*fd);
    }

    cDist = cDist / sqrt((float)numFeat);

	float c = 1-abs(cDist);
    
	dist = dist*dist*dist;

	//c = c*c*c;

	if(dist < 0.0000001)
		dist = 0.0000001;
	    

	force[0] = c * r[0] / dist;
	force[1] = c * r[1] / dist;
	force[2] = c * r[2] / dist;
}

__device__ void CoulombPM(const float* p1, const float* p2, const int numFeat, const float* f1, int idx1, const float* f2, int idx2, float* force) 
{
	float r[3];
    r[0] = p1[0] - p2[0];
    r[1] = p1[1] - p2[1];
    r[2] = p1[2] - p2[2];

    float dist = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);

    float cDist = 0.f;
    for(int i = 0; i < numFeat; i++)
    {
        float fd = abs(f1[idx1*numFeat + i] - f2[idx2*numFeat + i]);
        cDist += abs(fd*fd);
    }
    cDist = cDist / sqrt((float)numFeat);
	float c = 2 * (1 - cDist) - 1;
    
	dist = dist*dist*dist;

	//c = c*c*c;

	if(dist < 0.0000001)
		dist = 0.0000001;

	force[0] = c * r[0] / dist;
	force[1] = c * r[1] / dist;
	force[2] = c * r[2] / dist;
}