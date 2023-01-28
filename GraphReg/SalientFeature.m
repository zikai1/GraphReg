function [pt,feature_score,struct_tensor]=SalientFeature(pc,num,removeOut)
%========================================
% Compute the point response intensity and the local geometry variation
% Input:
%       pc   the point cloud 
%       num  the downsampling rate of the salient points
%       removeOut  if removing outliers according to the x84 statistics
% Output:
%       pt   the downsampled point cloud with salient feature points
%       feature_score  the point response intensity
%       struct_tensor  the local geometry variation
%       
%========================================

coordinate=pc.Location;

%------------compute normal and curvature for each data point----------
[Nor,Cur]=findPointNormals(coordinate,10,[0,0,10],true); 

geoFeat=[Nor Cur];

%-----------point response intensity (pointScore) and local feature variation (featScore)-----------------
[pointScore,featScore] = computeVariationAndFeaturePoint(coordinate, geoFeat,10);

nanInx=isnan(pointScore);
featScore(nanInx,:)=[];
pointScore(nanInx,:)=[];
coordinate(nanInx,:)=[];

N = size(pointScore,1);


%----------------sample round(N/num) points------------------
s = RandStream('mlfg6331_64'); % set random number 


if removeOut
    % use robust statistics x84 rule to remove scattered outliers
    x84_rule=median(abs(pointScore-repmat(median(pointScore),N,1)));
    M=find(pointScore<=5.2*x84_rule);%5.2 in default or smaller values to remove more outliers 
    
else
    % random sampling
    % use graph filter h(A)=I-A
    M = datasample(s,1:N, round(N/num), 'Replace', false,'Weights',  pointScore(:,1) );
end


feature_data=[coordinate(M,1), coordinate(M,2), coordinate(M, 3)];
feature_score=pointScore(M,:);
struct_tensor=featScore(M,:);


%----------sampled points are stored as point cloud---------------
pt=pointCloud(feature_data);


if ~removeOut
%---------no sampling
pt=pc;
feature_score=pointScore;
struct_tensor=featScore;
end




