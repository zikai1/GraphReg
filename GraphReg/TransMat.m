function [ M ] = TransMat( vector )
%TRANSMAT Generates a homogenuous 4x4-matrix representing a translation
%      about the vector (x,y,z)
%   
%input:
%   vector  the translation vector (x,y,z)
%
%return:
%   M   the matrix

M = [1 0 0 vector(1); ...
     0 1 0 vector(2); ...
     0 0 1 vector(3); ...
     0 0 0 1];
end

