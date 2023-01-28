#include "MatchingMethods.cuh"

__device__ void GetForce(int matchingMethod, const float* p1, const float* p2, const int numFeat, const float* f1, int id1, const float* f2, int id2, float* force)
{
	switch (matchingMethod)
	{
		// Gravitation
		case 0:
			{
				Gravity(p1, p2, force);
				break;
			}
		case 1:
			{
				Coulomb(p1, p2, numFeat, f1, id1, f2, id2, force);
				break;
			}
		case 2:
			{
				CoulombPM(p1, p2, numFeat, f1, id1, f2, id2, force);
				break;
			}
		default:
			{
				Gravity(p1, p2, force);
				break;
			}
	}
	
}

__global__ void MatchingStep(   const float* cloud1, const float* cloud2,
                                const int* index1, const int* index2,
                                const float* feat1, const float* feat2,
                                const int numFeat, const float* centroid2,
								const float* M,
								float* force, float* torque, float* momentInt,
								const int matchingMethod,
								const int numPoints, const int numSteps)
{
    int tID = threadIdx.x;

	if(tID >= numPoints)
		return;

	int idx2 = index2[tID];

	float p2[3];
	p2[0] = cloud2[idx2 * 3 + 0];
	p2[1] = cloud2[idx2 * 3 + 1];
	p2[2] = cloud2[idx2 * 3 + 2]; 

	// apply the current transformation to the points of cloud2 and the centroid 2
	// transform the cloud2
	float tp2[3];
	tp2[0] = M[0]*p2[0] + M[1]*p2[1] + M[2]*p2[2]  + M[3];
	tp2[1] = M[4]*p2[0] + M[5]*p2[1] + M[6]*p2[2]  + M[7];
	tp2[2] = M[8]*p2[0] + M[9]*p2[1] + M[10]*p2[2] + M[11];

	float f[3];
	f[0] = 0;
	f[1] = 0;
	f[2] = 0;

	for(int i = 0; i < numSteps; i++)
	{
		//int idx1 = index1[tID];

		int idx1 = index1[i];

		float p1[3];
		p1[0] = cloud1[idx1 * 3 + 0];
		p1[1] = cloud1[idx1 * 3 + 1];
		p1[2] = cloud1[idx1 * 3 + 2];

		float fDir[3];
		
		GetForce(matchingMethod, p1, tp2, numFeat, feat1, idx1, feat2, idx2, fDir);

		f[0] += fDir[0];
		f[1] += fDir[1];
		f[2] += fDir[2];

	}

    force[tID * 3 + 0] = f[0] ;
    force[tID * 3 + 1] = f[1] ;
    force[tID * 3 + 2] = f[2] ;

    float p2c[3];
    p2c[0] = tp2[0] - centroid2[0];
    p2c[1] = tp2[1] - centroid2[1];
    p2c[2] = tp2[2] - centroid2[2];

    float tDir[3];
    crossVec3<float>(tDir, p2c, f);
    
    torque[tID * 3 + 0] = tDir[0];
    torque[tID * 3 + 1] = tDir[1];
    torque[tID * 3 + 2] = tDir[2];

	// the moment of inertia with the mass = 1
	momentInt[tID] = sqrt( p2c[0]*p2c[0] + p2c[1]*p2c[1] + p2c[2]*p2c[2] );
	
}