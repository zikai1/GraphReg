# GraphReg
<table>
    <tr>
        <td ><center><img src="https://github.com/zikai1/GraphReg/blob/main/GraphReg/input_clean.png" width = "200" height = "200"> </center></td>
        <td ><center><img src="https://github.com/zikai1/GraphReg/blob/main/GraphReg/result_clean.png" width = "200" height = "200"> </center></td>
        <td ><center><img src="https://github.com/zikai1/GraphReg/blob/main/GraphReg/input.png" width = "200" height = "200"> </center></td>
        <td ><center><img src="https://github.com/zikai1/GraphReg/blob/main/GraphReg/result.png" width = "200" height = "200"> </center></td>
    </tr>
</table>

## 1. Motivation
This work aims to improve the robustness and convergence process with respect to currently dynamical point cloud registration methods. 

The designed method is a local registration framework and is an effective alternative to ICP.

We use the graph signal processing theory to describe the local geometry features, i.e., the point response intensity and the local geometry variations, through which we can remove outliers and attain invariants to rigid transformations.

## 2. Useage

> Run "GraphReg.m" to see demo examples.

There are several parameters that can be adjusted for better registration results or faster convergence process:
+ **cool_down**: Smaller values typically result in faster convergence but with fluctuations, whereas larger ones ensure more accurate registrations. The suggested interval scope empirically attaining good results is [0.8, 0.98]. 
+ **$\alpha$**: Although we can fix $\alpha=5.2$ to remove outliers for most test settings, smaller $\alpha$ will further improve the robusteness when there are a large percentage of outliers. 

## 3. Contact
If you find any question, please do not hesitate to contact me 
myzhao@baai.ac.cn

## 4. Citation
If you find this implementation useful for your research, please cite:
+ M. Zhao, L. Ma, X. Jia, D. -M. Yan and T. Huang, "GraphReg: Dynamical Point Cloud Registration With Geometry-Aware Graph Signal Processing," in IEEE Transactions on Image Processing, vol. 31, pp. 7449-7464, 2022, doi: 10.1109/TIP.2022.3223793.
+ P. Jauer, I. Kuhlemann, R. Bruder, A. Schweikard and F. Ernst, "Efficient Registration of High-Resolution Feature Enhanced Point Clouds," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 5, pp. 1102-1115, 1 May 2019, doi: 10.1109/TPAMI.2018.2831670.
