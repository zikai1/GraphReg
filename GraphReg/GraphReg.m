%===========================
% GraphReg: Dynamical point cloud registration with geometry-aware graph signal
% processing
% GraphReg is a local registration framework aiming to produce robust and
% accurate results through the graph signal processing theory and the adaptive simulated annealing mechanism. 
% M. Zhao, L. Ma, X. Jia, D. -M. Yan and T. Huang, 
%"GraphReg: Dynamical Point Cloud Registration With Geometry-Aware Graph Signal Processing," 
% IEEE Transactions on Image Processing, vol. 31, pp. 7449-7464, 2022.
%==========================

clear;
clc;
close all;

%------read the input point clouds
file1='.\data\bun000.ply';
file2='.\data\bun045.ply';

% file1='.\data\dragonStandRight_0.ply';
% file2='.\data\dragonStandRight_24.ply';

% add outliers
file1='.\data\bun000_outlier_0.1.ply';
file2='.\data\bun045_outlier_0.1.ply';

% file1='.\data\bun000_outlier_0.5.ply';
% file2='.\data\bun045_outlier_0.5.ply';



tgt=pcread(file1);
src=pcread(file2);

%------downsample and show the point clouds 
tgt=pcdownsample(tgt,'gridAverage',0.001); 
src=pcdownsample(src,'gridAverage',0.001);

figure;
pcshowpair(src,tgt);
title("Input point clouds");


%------compute the point intensity (scoreP,scoreQ) and the local geometric feature (featP, featQ)
[tgt2,scoreP,featP]=SalientFeature(tgt,10,true);% 10 in general
[src2,scoreQ,featQ]=SalientFeature(src,10,true);


tgtPt=tgt2.Location;
srcPt=src2.Location;


tgtPt=tgtPt';
srcPt=srcPt';

fp=featP';
fq=featQ';


%-----------start registration----------------
% add features and score to a struct 
feat = struct('p', fp, 'q', fq);
score=struct('pScore',scoreP','qScore',scoreQ');


addpath('cuda');


cool_down=0.9;%0.9 in default, adjust cool_down for better results or faster convergence process such as 0.8:0.02:0.98;


[T] = AdaptiveSimulatedAnnealingOptimization(tgtPt, srcPt, feat,score,cool_down);


tform=affine3d(T');

src2tgt=pctransform(src,tform);


figure;
pcshowpair(src2tgt,tgt);
title("Registration results");
