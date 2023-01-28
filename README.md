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
+ **$\alpha$**: Although we can fix $\alpha=5.2$ for most test settings, smaller $\alpha$ will further improve the robusteness when there are a large percentage of outliers. 

## 3. Contact
If you find any question, please do not hesitate to contact me 
myzhao@baai.ac.cn

## 4. Citation
If you find our work useful, please cite 
@ARTICLE{9966512,
  author={Zhao, Mingyang and Ma, Lei and Jia, Xiaohong and Yan, Dong-Ming and Huang, Tiejun},
  journal={IEEE Transactions on Image Processing}, 
  title={GraphReg: Dynamical Point Cloud Registration With Geometry-Aware Graph Signal Processing}, 
  year={2022},
  volume={31},
  number={},
  pages={7449-7464},
  doi={10.1109/TIP.2022.3223793}}
