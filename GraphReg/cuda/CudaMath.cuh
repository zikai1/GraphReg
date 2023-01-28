////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	CudaMath.cuh
//
// summary:	cuda mathematic operations
////////////////////////////////////////////////////////////////////////////////////////////////////

//#ifndef __CUDA_MATH_CUH__
//#define __CUDA_MATH_CUH__
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cmath>

/// <summary>	Defines an epsilon for computational accuracy. </summary>
#define CM_EPS 1e-6

/// <summary>	Defines PI. </summary>
#define CM_PI 3.1415926535897932384626433832795028841971693993751f

/// <summary>	Size of the float 4x4 matrix. </summary>
#define SIZE_OF_FLOAT_MAT4 sizeof(float) * 16


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Creates a 4x4 identity. </summary>
///
/// <remarks>	I. Kuhlemann, P. Jauer, 2014-10-27. </remarks>
///
/// <param name="M">	[in,out] If non-null, the float* to process. </param>
///
/// <returns>	true if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ __forceinline__ bool mat4CreateIdentity(T* M) {

	
	/*if (sizeof(M) != SIZE_OF_FLOAT_MAT4)
		return false;*/

	M[0]  = 1.0; M[1]  = 0.0; M[2]  = 0.0; M[3]  = 0.0;
	M[4]  = 0.0; M[5]  = 1.0; M[6]  = 0.0; M[7]  = 0.0;
	M[8]  = 0.0; M[9]  = 0.0; M[10] = 1.0; M[11] = 0.0;
	M[12] = 0.0; M[13] = 0.0; M[14] = 0.0; M[15] = 1.0;

	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Makes a exact copy of a 4x4 matrix. </summary>
///
/// <remarks>	Philipp Jauer, 2014-12-15. </remarks>
///
/// <param name="Dst">	[in,out] The copy destination. </param>
/// <param name="Src">	[in,out] The copy source. </param>
///
/// <returns>	true if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ __forceinline__ bool mat4Copy(T* Dst, T* Src) {

	for (int i = 0; i < 16; i++)
		Dst[i] = Src[i];

	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Creates a 4x4 matrix rotation about the x-axis. </summary>
///
/// <remarks>	I. Kuhlemann, P. Jauer, 2014-10-27. </remarks>
///
/// <param name="M">		[in,out] If non-null, the float* to process. </param>
/// <param name="angle">	The angle in radians. </param>
///
/// <returns>	true if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ bool mat4CreateRotX(float* M, float angle) {

	/*if (sizeof(M) != SIZE_OF_FLOAT_MAT4)
		return false;*/

	float ct = cos(angle);
	float st = sin(angle);

	M[0]  = 1.0; M[1]  = 0.0; M[2]  = 0.0; M[3]  = 0.0;
	M[4]  = 0.0; M[5]  =  ct; M[6]  = -st; M[7]  = 0.0;
	M[8]  = 0.0; M[9]  =  st; M[10] =  ct; M[11] = 0.0;
	M[12] = 0.0; M[13] = 0.0; M[14] = 0.0; M[15] = 1.0;

	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Creates a 4x4 matrix rotation about the y-axis. </summary>
///
/// <remarks>	I. Kuhlemann, P. Jauer, 2014-10-27. </remarks>
///
/// <param name="M">		[in,out] If non-null, the float* to process. </param>
/// <param name="angle">	The angle. </param>
///
/// <returns>	true if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ bool mat4CreateRotY(float* M, float angle) {
	
	/*if (sizeof(M) != SIZE_OF_FLOAT_MAT4)
		return false;*/

	float ct = cos(angle);
	float st = sin(angle);

	M[0]  =  ct; M[1]  = 0.0; M[2]  =  st; M[3]  = 0.0;
	M[4]  = 0.0; M[5]  = 1.0; M[6]  = 0.0; M[7]  = 0.0;
	M[8]  = -st; M[9]  = 0.0; M[10] =  ct; M[11] = 0.0;
	M[12] = 0.0; M[13] = 0.0; M[14] = 0.0; M[15] = 1.0;

	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Creates a 4x4 matrix rotation about the z-axis. </summary>
///
/// <remarks>	I. Kuhlemann, P. Jauer, 2014-10-27. </remarks>
///
/// <param name="M">		[in,out] If non-null, the float* to process. </param>
/// <param name="angle">	The angle in radians. </param>
///
/// <returns>	true if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ bool mat4CreateRotZ(float* M, float angle){

	/*if (sizeof(M) != SIZE_OF_FLOAT_MAT4)
		return false;*/

	float ct = cos(angle);
	float st = sin(angle);
		
	M[0]  = ct;  M[1]  = -st; M[2]  = 0.0; M[3]  = 0.0;
	M[4]  = st;  M[5]  =  ct; M[6]  = 0.0; M[7]  = 0.0;
	M[8]  = 0.0; M[9]  = 0.0; M[10] = 1.0; M[11] = 0.0;
	M[12] = 0.0; M[13] = 0.0; M[14] = 0.0; M[15] = 1.0;

	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Multiplies two 4x4 matrices A and B, A*B = C. </summary>
///
/// <remarks>	Philipp Jauer, 2014-10-23. </remarks>
///
/// <param name="C">	[out] Resulting matrix C. </param>
/// <param name="A">	[in] Left Matrix A. </param>
/// <param name="B">	[in] Right Matrix B. </param>
///
/// <returns>	true if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ bool mat4Mult(float* C, float* A, float* B) {

	/*if (sizeof(C) != SIZE_OF_FLOAT_MAT4 || sizeof(A) != SIZE_OF_FLOAT_MAT4 || sizeof(B) != SIZE_OF_FLOAT_MAT4)
		return false;*/

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			float sum = 0.0;
			for (int k = 0; k < 4; k++)
			{
				sum += A[i * 4 + k] * B[k * 4 + j];
			}
			C[i * 4 + j] = sum;
		}
	}

	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	4x4 matrix inverse. </summary>
///
/// <remarks>	I. Kuhlemann, P. Jauer, 2014-10-24. </remarks>
///
/// <param name="M">	[out] Result of the inversion. </param>
/// <param name="A">	[in]  Matrix to invert. </param>
///
/// <returns>	true if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ bool mat4Inv( float* A)
{
	//if (sizeof(M) != SIZE_OF_FLOAT_MAT4 || sizeof(A) != SIZE_OF_FLOAT_MAT4)
	//	return false;


	double inv[16], det;
	int i;

	inv[0] = A[5] * A[10] * A[15] - 
		A[5] * A[11] * A[14] -
		A[9] * A[6] * A[15] +
		A[9] * A[7] * A[14] +
		A[13] * A[6] * A[11] -
		A[13] * A[7] * A[10];

	inv[4] = -A[4] * A[10] * A[15] +
		A[4] * A[11] * A[14] +
		A[8] * A[6] * A[15] -
		A[8] * A[7] * A[14] -
		A[12] * A[6] * A[11] +
		A[12] * A[7] * A[10];

	inv[8] = A[4] * A[9] * A[15] -
		A[4] * A[11] * A[13] -
		A[8] * A[5] * A[15] +
		A[8] * A[7] * A[13] +
		A[12] * A[5] * A[11] -
		A[12] * A[7] * A[9];

	inv[12] = -A[4] * A[9] * A[14] +
		A[4] * A[10] * A[13] +
		A[8] * A[5] * A[14] -
		A[8] * A[6] * A[13] -
		A[12] * A[5] * A[10] +
		A[12] * A[6] * A[9];

	inv[1] = -A[1] * A[10] * A[15] +
		A[1] * A[11] * A[14] +
		A[9] * A[2] * A[15] -
		A[9] * A[3] * A[14] -
		A[13] * A[2] * A[11] +
		A[13] * A[3] * A[10];

	inv[5] = A[0] * A[10] * A[15] -
		A[0] * A[11] * A[14] -
		A[8] * A[2] * A[15] +
		A[8] * A[3] * A[14] +
		A[12] * A[2] * A[11] -
		A[12] * A[3] * A[10];

	inv[9] = -A[0] * A[9] * A[15] +
		A[0] * A[11] * A[13] +
		A[8] * A[1] * A[15] -
		A[8] * A[3] * A[13] -
		A[12] * A[1] * A[11] +
		A[12] * A[3] * A[9];

	inv[13] = A[0] * A[9] * A[14] -
		A[0] * A[10] * A[13] -
		A[8] * A[1] * A[14] +
		A[8] * A[2] * A[13] +
		A[12] * A[1] * A[10] -
		A[12] * A[2] * A[9];

	inv[2] = A[1] * A[6] * A[15] -
		A[1] * A[7] * A[14] -
		A[5] * A[2] * A[15] +
		A[5] * A[3] * A[14] +
		A[13] * A[2] * A[7] -
		A[13] * A[3] * A[6];

	inv[6] = -A[0] * A[6] * A[15] +
		A[0] * A[7] * A[14] +
		A[4] * A[2] * A[15] -
		A[4] * A[3] * A[14] -
		A[12] * A[2] * A[7] +
		A[12] * A[3] * A[6];

	inv[10] = A[0] * A[5] * A[15] -
		A[0] * A[7] * A[13] -
		A[4] * A[1] * A[15] +
		A[4] * A[3] * A[13] +
		A[12] * A[1] * A[7] -
		A[12] * A[3] * A[5];

	inv[14] = -A[0] * A[5] * A[14] +
		A[0] * A[6] * A[13] +
		A[4] * A[1] * A[14] -
		A[4] * A[2] * A[13] -
		A[12] * A[1] * A[6] +
		A[12] * A[2] * A[5];

	inv[3] = -A[1] * A[6] * A[11] +
		A[1] * A[7] * A[10] +
		A[5] * A[2] * A[11] -
		A[5] * A[3] * A[10] -
		A[9] * A[2] * A[7] +
		A[9] * A[3] * A[6];

	inv[7] = A[0] * A[6] * A[11] -
		A[0] * A[7] * A[10] -
		A[4] * A[2] * A[11] +
		A[4] * A[3] * A[10] +
		A[8] * A[2] * A[7] -
		A[8] * A[3] * A[6];

	inv[11] = -A[0] * A[5] * A[11] +
		A[0] * A[7] * A[9] +
		A[4] * A[1] * A[11] -
		A[4] * A[3] * A[9] -
		A[8] * A[1] * A[7] +
		A[8] * A[3] * A[5];

	inv[15] = A[0] * A[5] * A[10] -
		A[0] * A[6] * A[9] -
		A[4] * A[1] * A[10] +
		A[4] * A[2] * A[9] +
		A[8] * A[1] * A[6] -
		A[8] * A[2] * A[5];

	det = A[0] * inv[0] + A[1] * inv[4] + A[2] * inv[8] + A[3] * inv[12];

	// if nearly zero
	if (det < CM_EPS)
		return false;

	det = 1.0 / det;

	for (i = 0; i < 16; i++)
		A[i] = inv[i] * det;

	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Normalizes the given vector of size 2. </summary>
///
/// <remarks>	I. Kuhlemann, P. Jauer, 2014-10-27. </remarks>
///
/// <param name="vector">	[in,out] If non-null, the vector. </param>
///
/// <returns>	true if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ bool normalizeVec2(float* vector) {

	//int size = sizeof(vector) / sizeof(float);

	float n = sqrtf(vector[0] * vector[0]
					+ vector[1] * vector[1]);
	vector[0] /= n;
	vector[1] /= n;
	
	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Normalize vector of size 3. </summary>
///
/// <remarks>	I. Kuhlemann, P. Jauer, 2014-10-27. </remarks>
///
/// <param name="vector">	[in,out] If non-null, the vector. </param>
///
/// <returns>	true if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ bool normalizeVec3(float* vector) {

	float n = sqrtf(vector[0] * vector[0]
		+ vector[1] * vector[1]
		+ vector[2] * vector[2]);
	vector[0] /= n;
	vector[1] /= n;
	vector[2] /= n;

	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Normalize vector of size 4. </summary>
///
/// <remarks>	I. Kuhlemann, P. Jauer, 2014-10-27. </remarks>
///
/// <param name="vector">	[in,out] If non-null, the vector. </param>
///
/// <returns>	true if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ bool normalizeVec4(float* vector) {
	float n = sqrtf(vector[0] * vector[0]
		+ vector[1] * vector[1]
		+ vector[2] * vector[2]
		+ vector[3] * vector[3]);

	vector[0] /= n;
	vector[1] /= n;
	vector[2] /= n;
	vector[3] /= n;

	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Cross product of a vector of size 3. </summary>
///
/// <remarks>	I. Kuhlemann, P. Jauer, 2014-10-27. </remarks>
///
/// <param name="c">	[in,out] If non-null, the float* to process. </param>
/// <param name="a">	[in,out] If non-null, the float* to process. </param>
/// <param name="b">	[in,out] If non-null, the float* to process. </param>
///
/// <returns>	true if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__device__ __forceinline__ bool crossVec3(T* c, T* a, T* b) {

	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];

	return true;
}

// TODO:: Hier unbedingt nochmal nachdenken und optimieren
__device__ __forceinline__ bool mat4ExtractRotation(float* M, float* A, float* B) {
	
	float x[3];
	x[0] = A[3] - B[3];
	x[1] = A[7] - B[7];
	x[2] = A[11] - B[11];

	normalizeVec3(x);
	
	int cnt = 0;
	for (int i = 0; i < 3; i++)
	{
		if (fabsf(x[i]) > CM_EPS)
			cnt++;
	}

	switch (cnt)
	{
	case 1:
		if (fabsf(x[0]) > CM_EPS){

			float beta = (x[0] < 0) ? -CM_PI/2.0 : CM_PI/2.0;
			
			float ct1 = cos(beta);
			float st1 = sin(beta);
			float ct2 = cos(CM_PI);
			float st2 = sin(CM_PI);

			float T1[16] = {ct2,  -st2, 0.0, 0.0,
							st2,   ct2, 0.0, 0.0,
							0.0, 0.0, 1.0, 0.0,
							0.0, 0.0, 0.0, 1.0 };
			
			float T2[16] = {  ct1, 0.0, st1,  0.0,
							 0.0, 1.0, 0.0, 0.0,
							 -st1, 0.0, ct1,  0.0,
							 0.0, 0.0, 0.0, 1.0 };

			mat4Mult(M, T1, T2);

		} else if (fabsf(x[0]) > CM_EPS) {
				
			float gamma = (x[1] < 0) ? CM_PI / 2.0 : -CM_PI / 2.0;

			float ct1 = cos(CM_PI/2);
			float st1 = sin(CM_PI/2);
			float ct2 = cos(gamma);
			float st2 = sin(gamma);

			float T1[16] = { ct2, -st2, 0.0, 0.0,
				st2, ct2, 0.0, 0.0,
				0.0, 0.0, 1.0, 0.0,
				0.0, 0.0, 0.0, 1.0 };

			float T2[16] = { ct1, 0.0, st1, 0.0,
				0.0, 1.0, 0.0, 0.0,
				-st1, 0.0, ct1, 0.0,
				0.0, 0.0, 0.0, 1.0 };

			mat4Mult(M, T1, T2);

		} else {
			
			float ct2 = cos(CM_PI);
			float st2 = sin(CM_PI);

			M[0] = ct2; M[1] = -st2; M[2] = 0.0; M[3] = 0.0;
			M[4] = st2; M[5] = ct2; M[6] = 0.0; M[7] = 0.0;
			M[8] = 0.0; M[9] = 0.0; M[10] = 1.0; M[11] = 0.0;
			M[12] = 0.0; M[13] = 0.0; M[14] = 0.0; M[15] = 1.0;
		}
		break;
	case 2:
		if (fabsf(x[0]) <= CM_EPS){
			// vector u
			float u[3];
			u[0] = x[2];
			u[2] = 0.0;
			u[1] = (-(x[0] * u[0]) - (x[2] * u[2])) / x[1];

			normalizeVec3(u);

			// vector v
			float v[3];
			crossVec3(v, x, u);

			float ct1 = cos(-CM_PI / 2);
			float st1 = sin(-CM_PI / 2);
			float ct2 = cos(CM_PI);
			float st2 = sin(CM_PI);

			//TODO:: hier 3x3 wegen schnelligkeit
			float T1[16] = { x[0], u[0], v[0], 0.0,
				x[1], u[1], v[1], 0.0,
				x[2], u[2], v[2], 0.0,
				0.0, 0.0, 0.0, 1.0 };

			float T2[16] = { ct1, 0.0, st1, 0.0,
				0.0, 1.0, 0.0, 0.0,
				-st1, 0.0, ct1, 0.0,
				0.0, 0.0, 0.0, 1.0 };

			float T3[16];
			mat4Mult(T3, T1, T2);

			T1[0] = ct2; T1[1] = -st2; T1[2] = 0.0; T1[3] = 0.0;
			T1[4] = st2; T1[5] = ct2; T1[6] = 0.0; T1[7] = 0.0;
			T1[8] = 0.0; T1[9] = 0.0; T1[10] = 1.0; T1[11] = 0.0;
			T1[12] = 0.0; T1[13] = 0.0; T1[14] = 0.0; T1[15] = 1.0;

			mat4Mult(M, T3, T1);
		}
		else if (fabsf(x[1]) <= CM_EPS) {
			// vector u
			float u[3];
			u[1] = fabsf(x[2]);
			u[2] = 0.0;
			u[0] = (-(x[1] * u[1]) - (x[2] * u[2])) / x[0];

			normalizeVec3(u);

			// vector v
			float v[3];
			crossVec3(v, x, u);

			float ct1 = cos(-CM_PI / 2);
			float st1 = sin(-CM_PI / 2);
			float ct2 = cos(CM_PI);
			float st2 = sin(CM_PI);

			//TODO:: hier 3x3 wegen schnelligkeit
			float T1[16] = { x[0], u[0], v[0], 0.0,
				x[1], u[1], v[1], 0.0,
				x[2], u[2], v[2], 0.0,
				0.0, 0.0, 0.0, 1.0 };

			float T2[16] = { ct1, 0.0, st1, 0.0,
				0.0, 1.0, 0.0, 0.0,
				-st1, 0.0, ct1, 0.0,
				0.0, 0.0, 0.0, 1.0 };

			float T3[16];
			mat4Mult(T3, T1, T2);

			T1[0] = ct2; T1[1] = -st2; T1[2] = 0.0; T1[3] = 0.0;
			T1[4] = st2; T1[5] = ct2; T1[6] = 0.0; T1[7] = 0.0;
			T1[8] = 0.0; T1[9] = 0.0; T1[10] = 1.0; T1[11] = 0.0;
			T1[12] = 0.0; T1[13] = 0.0; T1[14] = 0.0; T1[15] = 1.0;

			mat4Mult(M, T3, T1);
		}
		else {
			// vector u
			float u[3];
			u[0] = 0.0;
			u[2] = x[1];
			//u[1] = (-(x[2] * u[2]) - (x[1] * u[2])) / x[1];
			u[1] = 0.0;

			normalizeVec3(u);

			// vector v
			float v[3];
			crossVec3(v, x, u);

			float ct1 = cos(-CM_PI / 2);
			float st1 = sin(-CM_PI / 2);

			//TODO:: hier 3x3 wegen schnelligkeit
			float T1[16] = { x[0], u[0], v[0], 0.0,
				x[1], u[1], v[1], 0.0,
				x[2], u[2], v[2], 0.0,
				0.0, 0.0, 0.0, 1.0 };

			float T2[16] = { ct1, 0.0, st1, 0.0,
				0.0, 1.0, 0.0, 0.0,
				-st1, 0.0, ct1, 0.0,
				0.0, 0.0, 0.0, 1.0 };

			float T3[16];
			mat4Mult(T3, T1, T2);

			T1[0] = ct1; T1[1] = -st1; T1[2] = 0.0; T1[3] = 0.0;
			T1[4] = st1; T1[5] = ct1; T1[6] = 0.0; T1[7] = 0.0;
			T1[8] = 0.0; T1[9] = 0.0; T1[10] = 1.0; T1[11] = 0.0;
			T1[12] = 0.0; T1[13] = 0.0; T1[14] = 0.0; T1[15] = 1.0;

			mat4Mult(M, T3, T1);
		}
		break;
	case 3:
		{
			// vector u
			float u[3];
			u[0] = x[2];
			u[2] = 0.0;
			u[1] = (-(x[0] * u[0]) - (x[2] * u[2])) / x[1];
		
			normalizeVec3(u);

			// vector v
			float v[3];
			crossVec3(v, x, u);

			float ct1 = cos(-CM_PI / 2);
			float st1 = sin(-CM_PI / 2);
			float ct2 = cos(CM_PI);
			float st2 = sin(CM_PI);

			//TODO:: hier 3x3 wegen schnelligkeit
			float T1[16] = {x[0], u[0], v[0], 0.0,
							x[1], u[1], v[1], 0.0,
							x[2], u[2], v[2], 0.0,
							0.0,  0.0,  0.0,  1.0};

			float T2[16] = {ct1, 0.0, st1, 0.0,
							0.0, 1.0, 0.0, 0.0,
							-st1, 0.0, ct1, 0.0,
							0.0, 0.0, 0.0, 1.0 };

			float T3[16];
			mat4Mult(T3, T1, T2);

			T1[0] = ct2; T1[1] = -st2; T1[2] = 0.0; T1[3] = 0.0;
			T1[4] = st2; T1[5] = ct2; T1[6] = 0.0; T1[7] = 0.0;
			T1[8] = 0.0; T1[9] = 0.0; T1[10] = 1.0; T1[11] = 0.0;
			T1[12] = 0.0; T1[13] = 0.0; T1[14] = 0.0; T1[15] = 1.0;

			mat4Mult(M, T3, T1);
		
			break;
		}
	default:
		// fuck off state
		break;
	}

	return true;
}

//#endif // __CUDA_MATH_CUH