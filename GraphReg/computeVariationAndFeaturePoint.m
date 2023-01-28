function [pointScore,featScore] = computeVariationAndFeaturePoint(coords, geoFeat,K)
%==========================
% Compute the point response intensity and the graph gradient w.r.t. normals and curvatures. 
%Input:
% coords: Nx3 array point coordinate
% geoFeat: Nx4  point normal and curvature
% K: K nearest neighbors
%
% Output:
% pointScore: Nx1  the response of each data point with respect to local...
% variation
% featScore: Nx2  the difference between adjacent normals and curvature
%==========================

z = single( coords );
N = size(z, 1);


%----------determine the radius and build the adjacency matrix


[~, dist] = knnsearch( coords, coords( round( linspace(1,N, 20))  ,:), 'k', K,  'distance', 'euclidean');
radius = max( dist(:, K) );
   
% find the K nearest points in z of each point in z
[idx, dist] = knnsearch( z, z, 'k', K,  'distance', 'euclidean');
dist(:,1)=[];
idx(:,1)=[];

%build the adjacency matrix
tmp=exp( -(dist./(radius*0.5) ).^2);
weight=tmp./sum(tmp,2);



weightPointSum=zeros(N,3);
normalDiff=zeros(N,3);


%--------------------compute the featScore
for i=1:N
    % near index of current point
    nearInx=idx(i,:);
    %extract current normal and near normal
    currentNormal=geoFeat(i,1:3);   
    nearPoint=z(nearInx,:);
    nearNormal=geoFeat(nearInx,1:3);
    
    % reshape near weight by K
    nearWeight=repmat(weight(i,:)',1,3);
    
    % difference of current point and normal with its nieghborhood
    weightPointSum(i,:)=sum(nearPoint.*nearWeight,1);
    
    % the second calculation manner  
    normalDiffSquare=sum((repmat(currentNormal,K-1,1)-nearNormal).^2,2);
    normalDiff(i,:)=sqrt(nearWeight(:,1)')*normalDiffSquare;
end

%record scores of point response intensity and local geometry variation
pointScore=sum((z-weightPointSum).^2,2);
featScore=[normalDiff(:,1) geoFeat(:,4)];