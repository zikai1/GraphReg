function [ T ] = AxisAngle( axis, angle )
%AXISANGLE Convertes axis angle representation to a 4x4 homogenuous
%rotation matrix
%   

c = cos(angle);
s = sin(angle);
t = 1.0 - c;

axis = axis / norm(axis);
x = axis(1);
y = axis(2);
z = axis(3);

T = [(x*x*t + c),   (x*y*t - z*s), (x*z*t + y*s), 0;...
     (y*x*t + z*s), (y*y*t + c),   (y*z*t - x*s), 0;...
     (z*x*t - y*s), (z*y*t + x*s), (z*z*t + c),   0;...
     0,              0,             0,            1];
end

