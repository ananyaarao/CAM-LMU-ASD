%At the very beginning, before running any part of this file, you will need to do/keep in mind the following:
%Download and align the correct sleep eeg files (in edf format) and corresponding hypnograms(in .txt format) and TET app data from camsleepmuc@gmail.com. 
%Copy and paste the TET daily data from the email onto a text file. If the data has been sent from the latest version of the app, you may directly import the data onto an excel file as a .csv and work with it (excel is able to read it and format it). If it is in the old format where the delimiters are ; and not , and they do not have a new line for every instance of the date-time (example: 2022-12-01 01:17:49.479 am) then you will need to include a specific code block below (it will be indicated)
%In the code blocks below, most can be run one after the other; but to generate sleep statistics off of hypnogram files, you will need to refer to and run the python file (<insert file path>) when indicated below (as the required package - YASA - is python based). After obtaining the sleep statistics and saving it to file, you will come back to this file and execute the remaining code blocks. 
%Ensure correct filenames and paths are entered at every step

%Code Block 1

%YASA (https://raphaelvallat.com/yasa/build/html/quickstart.html, https://github.com/raphaelvallat/yasa/blob/9d9fb741887c96c3259e2fd0b66f525b76dbd623/yasa/hypno.py#L740) has a specific requirement for hypnogram (text file) format and sleep stage encoding that is different from what dreem provides. therefore: need an additional step to preprocess:

%enter the list of hypnograms of all days with the entire filepath as stored on your system. You may also do this one hypnogram by one if you wish
hypn_list = ["filepath\filename1.txt", "filepath\filename2.txt"];

%enter filepath/filename of the destination that you want to store the hypnogram in. Ensure that they are in .csv format
hypn_modified_output_list = ["filepath\filename1.csv", "filepath\filename2.csv"];

for k= 1:width(hypn_modified_output_list)
hypn=readtable(hypn_list(1,k));

%picking out only the column "SleepStage" as it has the required data
hypn_data = hypn(:, "SleepStage"); 

%creating a new variable to store the modified data
hypn_data_mod = zeros(length(hypn_data.SleepStage),1);

% Wake=0, REM=4, N1=1, N2=2, N3=3, movement or artifact=0 (as YASA does not
% yet have a specific encoding to count for artifacts - https://raphaelvallat.com/yasa/build/html/quickstart.html#hypnogram - so encoding it as
% wake. Update this if suitable updates to YASA is found 
    for i=1:height(hypn_data)        
        if isequal(hypn_data.SleepStage{i,1}, 'SLEEP-S0')
            hypn_data_mod(i,1) = 0;
        
        elseif isequal(hypn_data.SleepStage{i,1},'SLEEP-S1')
            hypn_data_mod(i,1) = 1;
        
        elseif isequal(hypn_data.SleepStage{i,1},'SLEEP-S2')
            hypn_data_mod(i,1) = 2;
        elseif isequal(hypn_data.SleepStage{i,1} ,'SLEEP-S3')
            hypn_data_mod(i,1) = 3;
        elseif isequal(hypn_data.SleepStage{i,1} ,'SLEEP-REM')
            hypn_data_mod(i,1) = 4;
        elseif isequal(hypn_data.SleepStage{i,1} ,'SLEEP-MT')
            hypn_data_mod(i,1) = 0; %revisit the numberings!
        end
    end

%Now to output this as a textfile suitable for YASA
% Give the text to be written as the first line of the file (YASA format requires this)
firstLineOfText = 'Stage';

% Open file for writing
fid = fopen(hypn_modified_output_list(1,k), 'w');

% Write the text to file
fprintf(fid, '%s\n', firstLineOfText);

% Enter the list of doubles, in this case hypn_data_mod, each item of which will be on a new line
fprintf(fid, '%d\n', hypn_data_mod);

% Close file
fclose(fid);

end

%after this, go to python

%Code Block 2

%after obtaining sleep stats from python and saving it in required folder as a csv,
%bring it back as a struct and use it for further downstream analysis. This
%will also be done one by one to allow proper visualisation of individual
%TET and to store the sleep stats as well as the individual TET parameters
%into structs that can all be called back together at once later for
%combined visualisation
T = readtable("filepath\filename1.csv");

%In order to convert the table to a struct where the first column of the table turns into fieldnames of the struct and the other column into
%corresponding field values:
SleepSt = struct(T.Var1{1}, T.Var2(1,1));
for i=2:length(T.Var1)
    if '%' == T.Var1{i}(1) %some of the variables have "%" as part of their names and this is not accepted by matlab as a valid fieldname. So we need to change this
        SleepSt.(strcat(string(T.Var1{i}(2:end)), 'perc')) = T.Var2(i,1);
    else
    SleepSt.(T.Var1{i}) = T.Var2(i,1);
    end
end

%Code Block 3 (OPTIONAL)

%next, get tet data - if the data was obtained from the latest version of
%the app, you may not have to run this code block. But if it was from the
%old version with ";" as delimitters, you would need to run this code
%block. This code block could even be modified to process any other kind
%delimitters by simply changing the search expressions. \
%This code block replaces the ";" delimitters with "," and by introducing a
%new line everytime a date-time expression is found. This changes the text
%document to a format suitable for excel to read correctly. 

% Open the input file in read mode
input_file = fopen("filepath\filename1.txt", 'r');

% Read the content of the file
content = fread(input_file, '*char')';

% Add a new line before each date time in the format yyyy-mm-dd hh:mm:ss.sss am/pm
content = regexprep(content, '(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} [ap]m)', '\n$1');

% Replace semicolons with commas
content = strrep(content, ';', ',');

% Open the output file in write mode with csv extension
output_file = fopen('filepath\filename1.csv', 'w');

% Write the modified content to the output file
fwrite(output_file, content, 'char');

% Close the input and output files
fclose(input_file);
fclose(output_file);

%Code Block 4

%extracting TET values from excel file and plotting
dat=readtable("filepath\filename1.csv");

% Define the search expressions - to extract and partition all the x and y values by question
expr = 'Question';
%to avoid duplicates in question indices - that could happen if the subject erased graph data instead of submitting the graph
expr2 = 'Submit rating'; 

%define the required question number (the required TET dimension). In this
%case, we want 4 & 5
reqQuestionNum1 = 4; %4 for sensory avoidance
reqQuestionNum2 = 5; %5 for social avoidance

%sometimes, it gets imported with headings such as var1,2,3,4,,,,etc. or else as time, xval, yval, etc. 
% To account for both:
if "Question" == dat.Properties.VariableNames{4} %i.e if it is imported as time, xval, yval, question, etc - if question variable exists it is in the 4th position. That is why we search for it in this way
% Find the indices where the expression exists in the cell array
indx = find(contains(dat(:,4).Question, expr) & contains(dat(:,3).Action, expr2));
% Display the indices and the question at these indices (optional but good to ensure there are no duplicates)
disp(indx);
for i = 1:height(indx)
    disp(dat(indx(i,1),4))
end

%collecting the required x and y values for the TET dimensions
%for Q4
x1 = rmmissing(dat(indx(reqQuestionNum1,1):indx(reqQuestionNum1+1,1)-1,7)).xValue; %rmissing is used to drop NaN values
y1 = rmmissing(dat(indx(reqQuestionNum1,1):indx(reqQuestionNum1+1,1)-1,8)).yValue;
titleGraphs1 = dat(indx(reqQuestionNum1,1),4).Question{1};

%for Q5
x2 = rmmissing(dat(indx(reqQuestionNum2,1):indx(reqQuestionNum2+1,1)-1,7)).xValue;
y2 = rmmissing(dat(indx(reqQuestionNum2,1):indx(reqQuestionNum2+1,1)-1,8)).yValue;
titleGraphs2 = dat(indx(reqQuestionNum2,1),4).Question{1};

%to visualise TET
%plotting Q4
subplot(2,1,1)
plot(x1,y1)
title(titleGraphs1)
xlabel('time (scaled down to 0-4 by app)')
ylabel('Not Avoiding - Avoiding') %similar to the way it is on the TET graph on the app

% Optional - once a correct scale to transform to 12am-12am (24 hours) is decided, this may be included. If included, update the subplot grid numbers
% subplot(2,2,2)
% plot(x1.*6, y1)
% title(titleGraphs1)
% xlabel('time - hours (from 12am)')
% ylabel('Not Avoiding - Avoiding')

%plotting Q5
subplot(2,1,2)
plot(x2,y2)
title(titleGraphs2)
xlabel('time - (scaled down by app)')
ylabel('Not Avoiding - Avoiding')

% Optional - once a correct scale to transform to 12am-12am (24 hours) is decided, this may be included. If included, update the subplot grid numbers
% subplot(2,2,4)
% plot(x2.*6, y2)
% title(titleGraphs2)
% xlabel('time - hours (from 12am)')
% ylabel('Not Avoiding - Avoiding')

else %i.e if they are imported as var1, var2, etc
% Find the indices where the expression exists in the cell array
indx = find(contains(dat(:,4).Var4, expr) & contains(dat(:,3).Var3, expr2));
% Display the indices
disp(indx);
for i = 1:height(indx)
    disp(dat(indx(i,1),4))
end

x1 = rmmissing(dat(indx(reqQuestionNum1,1):indx(reqQuestionNum1+1,1)-1,7)).Var7;
y1 = rmmissing(dat(indx(reqQuestionNum1,1):indx(reqQuestionNum1+1,1)-1,8)).Var8;
titleGraphs1 = dat(indx(reqQuestionNum1,1),4).Var4{1};

x2 = rmmissing(dat(indx(reqQuestionNum2,1):indx(reqQuestionNum2+1,1)-1,7)).Var7;
y2 = rmmissing(dat(indx(reqQuestionNum2,1):indx(reqQuestionNum2+1,1)-1,8)).Var8;
titleGraphs2 = dat(indx(reqQuestionNum2,1),4).Var4{1};

%plotting Q4
subplot(2,1,1)
plot(x1,y1)
title(titleGraphs1)
xlabel('time (scaled down by app)')
ylabel('Not Avoiding - Avoiding')

% Commented out for the same reasons as in the previous conditional block
% subplot(2,2,2)
% plot(x1.*6, y1)
% title(titleGraphs1)
% xlabel('time - hours (from 12am)')
% ylabel('Not Avoiding - Avoiding')

%plotting Q5
subplot(2,1,2)
plot(x2,y2)
title(titleGraphs2)
xlabel('time (scaled down by app')
ylabel('Not Avoiding - Avoiding')

% Commented out for the same reasons as in the previous conditional block
% subplot(2,2,4)
% plot(x2.*6, y2)
% title(titleGraphs2)
% xlabel('time - hours (from 12am)')
% ylabel('Not Avoiding - Avoiding')
end

%Code Block 5

% extracting statistical parameters

%1. range
range_tet4 = range(y1);
range_tet5 = range(y2);

%2. max or peak - optional
max_tet4 = max(y1);
max_tet5 = max(y2);

%3. time at which maximum occurs
maxtime_tet4 = x1(y1==max_tet4); %but this is just x value. to get exact time stamps, need to convert the x values from the app - axis varies over a 24 hour scale on the app.
maxtime_tet5 = x2(y2==max_tet5);

% 4. the amount of time someone exceeds a preset threshold
thresh_tet = 2;
timegthresh4 = range(x1(y1>thresh_tet)); %will need to be scaled to get actual time period value
timegthresh5 = range(x2(y2>thresh_tet));

%5. Mean
mean_TET4 = mean(y1);
mean_TET5 = mean(y2);

%6. Median
median_TET4 = median(y1);
median_TET5 = median(y2);


%Code Block 6

%saving all sleep and avoidance parameters as a struct and saved to file
TET_Params_sensav = struct('xVals', x1, 'yVals', y1, 'rangeTet', range_tet4, 'maxTet', max_tet4,'maxtimeTet', maxtime_tet4, 'threshTet', thresh_tet, 'timeGTthresh', timegthresh4, 'meanTET', mean_TET4, 'medianTET', median_TET4);
TET_Params_socav = struct('xVals', x2, 'yVals', y2, 'rangeTet', range_tet5, 'maxTet', max_tet5, 'maxtimeTet', maxtime_tet5, 'threshTet', thresh_tet, 'timeGTthresh', timegthresh5, 'meanTET', mean_TET5, 'medianTET', median_TET5);

sleepVSTet = struct('SleepParams', SleepSt, 'TETParamsSensAvoid', TET_Params_sensav, 'TETParamsSocAvoid', TET_Params_socav);

% Save "sleepVSTet" to a file called "sleepVSTet_<insert_date>.mat" in the required directory
save('filepath\filename1.mat', 'sleepVSTet');

%keep repreating until this code block (i.e from block 2-6) until all days' data has been processed and stored as structs

%Code Block 7

%to infer a relationship between various parameters
%first, to bring all the saved structs back to workspace

%provide list of the form filepath/filename of all the structs
sleepVSTetList = ["filepath\filename1.mat", "filepath\filename2.mat"];

% Initialize an empty cell array of structs
all_structs = {};

%populate the cell with the structs
for i = 1:width(sleepVSTetList)
    saved_struct = load(sleepVSTetList(1,i));
    % Add the saved struct to the cell array
    all_structs{end+1} = saved_struct;
end

% Convert the cell array of structs to a table - to make iterating easier
% to generate graphs later
all_structs_table = cell2table(all_structs);

%Code Block 8

%Plot variables of choice. Label the points by the date on which they were recorded
X = [];
Y = [];
for i = 1:width(sleepVSTetList)
    X(1,i) = all_structs_table{1,i}.sleepVSTet.SleepParams.TST;
    Y(1,i) = all_structs_table{1,i}.sleepVSTet.TETParamsSensAvoid.meanTET;
end
figure
scatter(X,Y)

% label the points with their corresponding names - i.e; dates
dates = { 'Date1', 'Date2'};
for i = 1:length(dates)
    text(X(i)+0.1, Y(i)+0.1, dates{i});
end

%provide the appropriate labels and titles
xlabel('Total Sleep time')
ylabel('Mean value of sensory avoidance scaled from 0-4')
title('MeanTET for Sensory Avoidance v/s TST')

%Code Block 9

%Overall variable visualisation
%first define y - a matrix of values where each row represents the values
%for that day. Need to do it thrice because sleep, sensory and social
%avoidance values needed

ysleep = [];
ysleepPerc = []; %for the percentage value
ysens = [];
ysoc = [];
%to define fields from sensory and social avoidance structs that need not
%be plotted:
av_flds = {'xVals', 'yVals', 'maxtimeTet', 'threshTet'};
%to split the sleep struct into 2 - one with fields with values in minutes - and the other with those in percentages
sleepMins = {'TIB','SPT','WASO','TST','N1','N2','N3','REM','NREM','SOL','Lat_N1','Lat_N2','Lat_N3','Lat_REM'};
sleepPerc = {'N1perc','N2perc','N3perc','REMperc','NREMperc', 'SE','SME'};

for i=1:width(sleepVSTetList)
    %should only contain fields whose values are in minutes
    sleepMinsStruct = rmfield(all_structs_table{1,i}.sleepVSTet.SleepParams, sleepPerc);
    ysleep = [ysleep; struct2array(sleepMinsStruct)];

    %should only contain fields whose values are in percentages
    sleepPercStruct = rmfield(all_structs_table{1,i}.sleepVSTet.SleepParams, sleepMins);
    ysleepPerc = [ysleepPerc; struct2array(sleepPercStruct)];

    %to remove fields from sensory and social avoidance structs that need not be plotted and redefine struct:
    senstest = all_structs_table{1,i}.sleepVSTet.TETParamsSensAvoid;
    senstest = rmfield(senstest, av_flds);
    if isempty(senstest.timeGTthresh)
        senstest.timeGTthresh = 0;
    end
    ysens = [ysens; struct2array(senstest)];

    socialtest = all_structs_table{1,i}.sleepVSTet.TETParamsSocAvoid;
    socialtest = rmfield(socialtest, av_flds);
    if isempty(socialtest.timeGTthresh)
        socialtest.timeGTthresh = 0;
    end
    ysoc = [ysoc; struct2array(socialtest)];
end

%We want the x axis in the graphs to be defined as the dates of eeg recording
dates = {'Date1', 'Date2'};
xsleep = categorical(dates);
xsleep = reordercats(xsleep,dates); %to retain the order in which we supply the xsleep values. These values will be used for the avoidance graphs as well

%This below commented-out code snippet is optional i.e; if we want to plot the bar graphs separately
% b = bar(xsleep, ysleep); 
% set(b, {'DisplayName'}, {'TIB','SPT','WASO','TST','N1','N2','N3','REM','NREM','SOL','LatN1','LatN2','LatN3','LatREM','N1perc','N2perc','N3perc','REMperc','NREMperc','SE','SME'}')
% legend() 
% 
% bsens = bar(xsleep, ysens); 
% set(bsens, {'DisplayName'}, fieldnames(senstest))
% legend()
% 
% bsoc = bar(xsleep, ysoc);
% set(bsoc, {'DisplayName'}, fieldnames(socialtest))
% legend()

%Code Block 10

%to plot all 3 bar graphs in one figure
tiledlayout(4,1);

% Tile 1
nexttile
b = bar(xsleep, ysleep); 
set(b, {'DisplayName'}, {'TIB','SPT','WASO','TST','N1','N2','N3','REM','NREM','SOL','LatN1','LatN2','LatN3','LatREM'}')
legend()
title('Sleep Parameters (values in minutes) assembled by date')
xlabel('Dates')
ylabel('Value in minutes')

% Tile 2
nexttile
b = bar(xsleep, ysleepPerc); 
set(b, {'DisplayName'}, {'N1perc','N2perc','N3perc','REMperc','NREMperc','SE','SME'}')
legend()
title('Sleep Parameters (values in percentages) assembled by date')
xlabel('Dates')
ylabel('Value in percentage')

% Tile 3
nexttile
bsens = bar(xsleep, ysens); 
set(bsens, {'DisplayName'}, fieldnames(senstest))
legend()
title('Sensory avoidance parameters (the corresponding day time for indicated sleep date on x-axis)')
xlabel('Dates')
ylabel('Values from App (vary from 0-4)')

% Tile 4
nexttile
bsoc = bar(xsleep, ysoc);
set(bsoc, {'DisplayName'}, fieldnames(socialtest))
legend()
title('Social avoidance parameters (the corresponding day time for indicated sleep date on x-axis)')
xlabel('Dates')
ylabel('Values from App (vary from 0-4)')
