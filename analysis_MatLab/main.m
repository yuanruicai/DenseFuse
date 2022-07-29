    %% Li H, Wu X J. DenseFuse: A Fusion Approach to Infrared and Visible Images[J]. arXiv preprint arXiv:1804.08361, 2018. 
%% https://arxiv.org/abs/1804.08361

EN_t = [];
MI_t = [];
Qabf_t = [];
FMI_pixel_t = [];
FMI_dct_t = [];
FMI_w_t = [];
Nabf_t = [];
SCD_t = [];
SSIM_t = [];
MS_SSIM_t = [];


disp("Start");
disp('---------------------------Analysis---------------------------');
for i = 1:20
fileName_source_ir  = strcat("IV_images/IR",num2str(i),".png");
fileName_source_vis = strcat("IV_images/VIS",num2str(i),".png");
fileName_fused       = strcat("fusion_coco_l1norm_softmax /F",num2str(i),".png");
source_image1 = imread(fileName_source_ir);
source_image2 = imread(fileName_source_vis);
fused_image   = imread(fileName_fused);
[EN,MI,Qabf,FMI_pixel,FMI_dct,FMI_w,Nabf,SCD,SSIM, MS_SSIM] = analysis_Reference(fused_image,source_image1,source_image2);
EN_t = [EN_t;EN];
MI_t = [MI_t;MI];
Qabf_t = [Qabf_t;Qabf];
FMI_pixel_t = [FMI_pixel_t;FMI_pixel];
FMI_dct_t = [FMI_dct_t;FMI_dct];
FMI_w_t = [FMI_w_t;FMI_w];
Nabf_t = [Nabf_t;Nabf];
SCD_t = [SCD_t;SCD];
SSIM_t = [SSIM_t;SSIM];
MS_SSIM_t = [MS_SSIM_t;MS_SSIM];
end
disp('Done');
disp(mean(EN_t));
disp(mean(Qabf_t));
disp(mean(SCD_t));
disp(mean(FMI_w_t));
disp(mean(FMI_dct_t));
disp(mean(SSIM_t));
disp(mean(MS_SSIM_t));

