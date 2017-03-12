clear all;
close all;

gt_pred_pan = dlmread('/Users/jimmy/Desktop/CVPR2017_UserStudy/ICCV_Camera_planning/code/srf_ball_soccer_44_fn_gd_pred.txt', '\t', 1, 0);
pred_pan=gt_pred_pan(:,3)

origin_data = dlmread('/Users/jimmy/Desktop/CVPR2017_UserStudy/ICCV_Camera_planning/code/soccer_44_ptz_ground_truth.txt', '\t', 1, 0);

origin_data(:,3)=pred_pan

[N,d]= size(origin_data)

fid = fopen('/Users/jimmy/Desktop/CVPR2017_UserStudy/ICCV_Camera_planning/code/replaced_soccer_44_ptz_ground_truth.txt', 'w');
for i=1:N
     fprintf(fid, '%d %f %f %f %f\n',origin_data(i,1),origin_data(i,2),origin_data(i,3), origin_data(i,4), origin_data(i,5));
end
