
%   This script runs the original implementation of Background Aware Correlation Filters (BACF) for visual tracking.
%   the code is tested for Mac, Windows and Linux- you may need to compile
%   some of the mex files.
%   Paper is published in ICCV 2017- 
%   Some functions are borrowed from other papers (SRDCF, CCOT, KCF, etc)- and
%   their copyright belongs to the paper's authors.
%   copyright- Hamed Kiani
%   contact me: hamedkg@gmail.com


%   This demo runs on OTB50, you can use any benchmark by setting the seq
%   path, and using the standard annotation txt files.
clear;clc;
close all;
% Load video information
base_path  = '/home/ubuntu/visual_tracker_benchmark/sequences/OTB100/';

videos = {'DragonBaby'};
% videos = {'Basketball', 'Bolt', 'Boy', 'Car4', 'CarDark', 'CarScale', ...
%     'Coke', 'Couple', 'Crossing', 'David2', 'David3', 'David', 'Deer', ...
%     'Dog1', 'Doll', 'Dudek', 'Faceocc1', 'Faceocc2', 'Fish', 'Fleetface', ...
%     'Football', 'Football1', 'Freeman1', 'Freeman3', 'Freeman4', 'Girl', ...
%     'Ironman', 'Jogging_1', 'Jumping', 'Lemming', 'Liquor', 'Matrix', ...
%     'Mhyang', 'MotorRolling', 'MountainBike', 'Shaking', 'Singer1', ...
%     'Singer2', 'Skating1', 'Skiing', 'Soccer', 'Subway', 'Suv', 'Sylvester', ...
%     'Tiger1', 'Tiger2', 'Trellis', 'Walking', 'Walking2', 'Woman'};

OPs = zeros(numel(videos),1);
FPSs = zeros(numel(videos),1);
OPs_OTB50 = [];
FPSs_OTB50 = [];

for vid = 1:numel(videos)
    close all;
    video_path = [base_path '/' videos{vid}];
    [seq, ground_truth] = load_video_info(video_path);
    seq.VidName = videos{vid};
    st_frame = 1;
    en_frame = seq.len;
    if (strcmp(videos{vid}, 'David'))
        st_frame = 300;
        en_frame = 770;
    elseif (strcmp(videos{vid}, 'Football1'))
        st_frame = 1;
        en_frame = 74;
    elseif (strcmp(videos{vid}, 'Freeman3'))
        st_frame = 1;
        en_frame = 460;
    elseif (strcmp(videos{vid}, 'Freeman4'))
        st_frame = 1;
        en_frame = 283;
    end
    seq.st_frame = st_frame;
    seq.en_frame = en_frame;
    gt_boxes = [ground_truth(:,1:2), ground_truth(:,1:2) + ground_truth(:,3:4) - ones(size(ground_truth,1), 2)];
    
    % Run BACF- main function
    learning_rate = 0.013;  %   you can use different learning rate for different benchmarks.
    results = run_BACF(seq, video_path, learning_rate);
    results.gt = gt_boxes;
    %   compute the OP
    pd_boxes = results.res;
    pd_boxes = [pd_boxes(:,1:2), pd_boxes(:,1:2) + pd_boxes(:,3:4) - ones(size(pd_boxes,1), 2)  ];
    OP = zeros(size(gt_boxes,1),1);
    for i=1:size(gt_boxes,1)
        b_gt = gt_boxes(i,:);
        b_pd = pd_boxes(i,:);
        OP(i) = computePascalScore(b_gt,b_pd);
    end
    OPs(vid) = sum(OP >= 0.5) / numel(OP);
    FPSs(vid) = results.fps;
    display([videos{vid}  '---->' '   FPS:   ' num2str(results.fps)   '    op:   '   num2str(OPs(vid))]);
   
    FPSs_OTB50 = [FPSs_OTB50; results.fps];
    OPs_OTB50 =  [OPs_OTB50; OPs(vid)];
end
