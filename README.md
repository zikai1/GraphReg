# GraphReg

This work aims to improve the robustness and convergence process with respect to currently dynamical point cloud registration methods. 

The designed method is a local registration framework and is an effective alternative to ICP.

We use the graph signal processing theory to describe the local geometry features, i.e., the point response intensity and the local geometry variations, through which we can remove outliers and attain invariants to rigid transformations.

Run "GraphReg.m" to see demo examples.

There are several parameters that can be adjusted for better registration results or faster convergence process, such as "cool_down".


If you find any question, please do not hesitate to contact me 
myzhao@baai.ac.cn

If you find our work useful, please cite 
M. Zhao, L. Ma, X. Jia, D. -M. Yan and T. Huang, "GraphReg: Dynamical Point Cloud Registration With Geometry-Aware Graph Signal Processing," in IEEE Transactions on Image Processing, vol. 31, pp. 7449-7464, 2022. 
