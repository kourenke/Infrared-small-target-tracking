
%   This script runs the original implementation of Background Aware Correlation Filters (BACF) for visual tracking.
%   the code is tested for Mac, Windows and Linux- you may need to compile
%   some of the mex files.
%   Paper is published in ICCV 2017- Italy
%   Some functions are borrowed from other papers (SRDCF, CCOT, KCF, etc)- and
%   their copyright belongs to the paper's authors.
%   copyright- Hamed Kiani (CMU, RI, 2017)

%   contact me: hamedkg@gmail.com

clear; clc; close all;
%   Load video information
base_path  = './seq';
video      = 'Small_target9';
video_path = [base_path '/' video];
[seq, ground_truth] = load_video_info(video_path);
seq.VidName = video;
seq.st_frame = 1;
seq.en_frame = seq.len;

% 将标签格式由[xmin,ymin,w,h]转为[xmin,ymin,xmax,ymax,]
gt_boxes = [ground_truth(:,1:2), ground_truth(:,1:2) + ground_truth(:,3:4) - ones(size(ground_truth,1), 2)];
gt_target_center = [(ground_truth(:,1) + ground_truth(:,3)/2), (ground_truth(:,2) + ground_truth(:,4)/2)];

% gt_boxes = ground_truth;
% gt_target_center = [ground_truth(:,1) + (ground_truth(:,3)-ground_truth(:,1))/2, ground_truth(:,2) + (ground_truth(:,4) - ground_truth(:,2))/2];



%   Run BACF- main function
learning_rate = 0.013;  %you can use different learning rate for different benchmarks.
results       = run_BACF(seq, video_path, learning_rate);


pd_boxes0 = results.res;
% 将预测结果格式由[xmin,ymin,w,h]转为[xmin,ymin,xmax,ymax,]
pd_boxes = [pd_boxes0(:,1:2), pd_boxes0(:,1:2) + pd_boxes0(:,3:4) - ones(size(pd_boxes0,1), 2)  ];
pd_target_center = [pd_boxes(:,1) + (pd_boxes(:,3)-pd_boxes(:,1))/2, pd_boxes(:,2) + (pd_boxes(:,4)-pd_boxes(:,2))/2];

%计算预测box与标签box的交并比，阈值大于0.5，视为该帧跟踪成功
OP = zeros(size(gt_boxes,1),1);
for i=1:size(gt_boxes,1)
    b_gt = gt_boxes(i,:);
    b_pd = pd_boxes(i,:);
    OP(i) = computePascalScore(b_gt,b_pd);
end
OP_vid = sum(OP >= 0.5) / numel(OP);

%绘制Overlap threshold' Vs Success rate的曲线
OP_vid1 = [];
for j=0:0.05:1
    Success_rate = sum(OP >= j) / numel(OP);
    OP_vid1=[OP_vid1,Success_rate];
end
figure;
x = [0:0.05:1.0];
y = OP_vid1;
plot(x, y)
xlim([0 1])
ylim([0 1])
xlabel('Overlap threshold');
ylabel('Success rate(%)');
legend('BACF');
grid on;
title('Overlap threshold - Success rate')

%计算accuracy（目标的标签中心与预测中心的欧式距离≤5piexl）
ac = zeros(size(pd_target_center,1),1);
for i=1:size(pd_target_center,1)
    d(i) = sqrt((pd_target_center(i,1)-gt_target_center(i,1))^2 + (pd_target_center(i,2)-gt_target_center(i,2))^2);
end
d_vid = sum(d <= 5) / numel(ac);

%绘制Location error threshold' Vs accuracy的曲线
d_vid1 = [];
for k=0:0.5:10
    accuracy = sum(d <= k) / numel(ac);
    d_vid1=[d_vid1,accuracy];
end
figure;
x = [0:0.5:10];
y = d_vid1;
plot(x, y)
xlim([0 10])
ylim([0 1])
xlabel('Location error threshold');
ylabel('Precision(%)');
legend('BACF');
grid on;
title('Location error threshold - Precision')


%计算FPS
FPS_vid = results.fps;

%计算AUC面积
thresholdSetOverlap = 0: 0.05 : 1;   
success_num_overlap = zeros(1, numel(thresholdSetOverlap));
res = calcRectInt(ground_truth, pd_boxes0);
for t = 1: length(thresholdSetOverlap)
    success_num_overlap(1, t) = sum(res > thresholdSetOverlap(t));
end
cur_AUC = mean(success_num_overlap) / size(ground_truth, 1);


display([video  '---->' '   FPS:' num2str(FPS_vid)   '    Success rate(T=0.5):'   num2str(OP_vid)  '    Precision:'   num2str(d_vid) '    AUC:'   num2str(cur_AUC)]);



