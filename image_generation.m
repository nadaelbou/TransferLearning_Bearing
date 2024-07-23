%% 


clear all;

if ~exist('cwt_generated_images','dir')
    mkdir('cwt_generated_images');
end


ClassLabels={'Healthy\', 'Cage\', 'OR\', 'IR\'};
LoadLevels={'0','25','50','75','100'};
n_classes=4; % The definitions of class labels are written in the above comments.
N_Frequencies=1; 
n_loadings=5; % There are a total of 5 loading levels ranging from light loads to full loads. (from l=1:5 the loading level increases.)
n_sensors=3; % Three phase current sensors 

% The main location of the data:
folder_path='Dataset_Bearing\';

for c=1:n_classes
    for l=1:n_loadings 
        Data{c,l}=load(join([folder_path,ClassLabels{c},LoadLevels{l},'.mat']));
    end
end


for c=1:n_classes
    for l=1:n_loadings 
        Data2{c,l,1}=Data{c,l}.ch_AI_6_2;  % Phase A current
        Data2{c,l,2}=Data{c,l}.ch_AI_6_3;  % Phase B current
        Data2{c,l,3}=Data{c,l}.ch_AI_6_4;  % Phase C current
    end
end
       
%Sampling frequency
Fs=20000; 

clear Data;

Start_point=1; % For filtering the signals if needed
WindowLength=20000*1; % The length of the segmentation window
step_length=20000;

 

%% Finding the number of samples for signal segmentation


for c=1:n_classes
    for l=1:n_loadings
        NSample(c,l)=floor((length(Data2{c,l,1}))/WindowLength); % Number of samples for each window.
    end 
end
N_samples=min(NSample(:));


%%  Making all the signals have equal length:
for c=1:n_classes
    for l=1:n_loadings
        for s=1:n_sensors
            Data2{c,l,s}=Data2{c,l,s}(Start_point:Start_point+N_samples*WindowLength-1);
        end
    end
end



%% Cleaning raw data 

for s=1:n_sensors
    for c=1:n_classes
        for l=1:n_loadings
            for z=1:size(Data2{c,l,s},2)
                Data2{c,l,s}(any(isinf(Data2{c,l,s}(:,z)),2),z)=0;
                Data2{c,l,s}(any(isnan(Data2{c,l,s}(:,z)),2),z)=0;
            end
        end
    end
end

%% Band-stop filter

bsFilt = designfilt('bandstopiir','FilterOrder',6, ...
         'HalfPowerFrequency1',49,'HalfPowerFrequency2',51, ...
         'SampleRate',20000);
%fvtool(bsFilt)

%%  Applying band-stop filter on data and removing the first 25000 data points

for i=1:n_classes
    for j=1:n_loadings
        for k=1:n_sensors
            Data2{i,j,k}=filtfilt(bsFilt,Data2{i,j,k});
            Data2{i,j,k}=Data2{i,j,k}(25000:end,1);
        end
    end
end

%% Reshaping and segmentation

max_iter=floor((size(Data2{1,1,1},1)-WindowLength)/step_length)+1;

for s=1:n_sensors
    for c=1:n_classes
        for l=1:n_loadings
            for z=1:max_iter
                index=(z-1)*step_length+1;
                Data3{c,l,s}(:,z)=Data2{c,l,s}(index:index+WindowLength-1,1);
            end
        end
    end
end



%%  Applying CWT for generating the time-frequency images:
% We save the images firstly in loading folders and then at class label
% folders. The reason is the "datasets.ImageFolder" in pytorch.
tic
for k=1:n_sensors
    for j=1:n_loadings
        for i=1:n_classes
            for z=1:size(Data3{i,j,k},2)
                x=Data3{i,j,k}(:,z);
                x=x./max((x));
                figure1=figure(1);
                [cfs,frq]=cwt(x,'amor',Fs);
                t = 0:1/Fs:(length(x)-1)/Fs;
                surf(t,frq,abs(cfs));
                colormap jet;
                caxis([0 0.6]);
                shading interp
                set(gca,'xticklabel',[],'xlabel',[],'yticklabel',[],'ylabel',[],'title',[]);
                ax=gca;
                ylim(ax, [0,1000]);
                view(2)
                xlim('tight')
                if i<10
                    dataFolder = sprintf('cwt_generated_images/Sensor_%d/Loading_%d/Classlabel_0%d',k,j,i);
                else
                    dataFolder = sprintf('cwt_generated_images/Sensor_%d/Loading_%d/Classlabel_%d',k,j,i);
                end
                
                if ~exist(dataFolder,'dir')
                    mkdir(dataFolder);
                end
                saveas(gcf,join([dataFolder,sprintf('/Sample_%d.jpg',z)]))
                close all
            end
        end
    end
end
toc


%%
%%  Reading the images

for k=1:1  % Only for sensor 1. Modify it if you want
    for j=1:n_loadings
        for i=1:n_classes
            for z=1:size(Data3{i,j,k},2)
                images{k,j,i,z}=imread(sprintf('cwt_generated_images/Sensor_%d/Loading_%d/Classlabel_0%d/Sample_%d.jpg',k,j,i,z));
            end
        end
    end
end

%%  Creating the scenario folder
ScenarioAlpha_Path='DL_input_data/ScenarioAlpha/Sensor1';
if ~exist(ScenarioAlpha_Path,'dir')
    mkdir(ScenarioAlpha_Path);
end

ScenarioBeta_Path_train_val='DL_input_data/ScenarioBeta/Sensor1/train_val';
ScenarioBeta_Path_test='DL_input_data/ScenarioBeta/Sensor1/test';
if ~exist(ScenarioBeta_Path_train_val,'dir')
    mkdir(ScenarioBeta_Path_train_val);
end

if ~exist(ScenarioBeta_Path_test,'dir')
    mkdir(ScenarioBeta_Path_test);
end

%%  Saving Images for Scenario Alpha Only for sensor 1, modify it if you want
mmm=0;
for k=1:1
    for j=1:n_loadings
        for i=1:n_classes
            for z=1:size(Data3{i,j,k},2)
                if i<10
                    DataFolder2=join([ScenarioAlpha_Path,sprintf('/Classlabel_0%d',i)]);
                else
                    DataFolder2=join([ScenarioAlpha_Path,sprintf('/Classlabel_%d',i)]);
                end
                if ~exist(DataFolder2,'dir')
                    mkdir(DataFolder2);
                end 
                imwrite(images{k,j,i,z},join([DataFolder2,sprintf('/Sample_%d.jpg',mmm)]))
                mmm=mmm+1;
            end
        end
    end
end

%%  Saving Images for Scenario Beta Only for sensor 1, modify it if you want
tic
list=[1 2 4 5];
mmm=0;
for j=1:4
    for i=1:n_classes
        for z=1:size(Data3{i,j,k},2)
            if i<10
                DataFolder2=join([ScenarioBeta_Path_train_val,sprintf('/Classlabel_0%d',i)]);
            else
                DataFolder2=join([ScenarioBeta_Path_train_val,sprintf('/Classlabel_%d',i)]);
            end
            if ~exist(DataFolder2,'dir')
                mkdir(DataFolder2);
            end 
            imwrite(images{1,list(j),i,z},join([DataFolder2,sprintf('/Sample_%d.jpg',mmm)]))
            mmm=mmm+1;
        end
    end
end


mmm=0;
for j=3:3  % Only 50% load for test data
    for i=1:n_classes
        for z=1:size(Data3{i,j,k},2)
            if i<10
                DataFolder2=join([ScenarioBeta_Path_test,sprintf('/Classlabel_0%d',i)]);
            else
                DataFolder2=join([ScenarioBeta_Path_test,sprintf('/Classlabel_%d',i)]);
            end
            if ~exist(DataFolder2,'dir')
                mkdir(DataFolder2);
            end 
            imwrite(images{1,j,i,z},join([DataFolder2,sprintf('/Sample_%d.jpg',mmm)]))
            mmm=mmm+1;
        end
    end
end
toc
