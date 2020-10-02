
% clc;% clear all


CONNECTION_MAP = [[1 2];
                  [2 3];
                  [2 4];
                  [2 20];
                  [3 4];
                  [3 20];
                  [4 5];
                  [5 6];
                  [6 7];
                  [4,8];
                  [8 9];
                  [9 10];
                  [10 11];
                  [8 12];
                  [12 13];
                  [13 14];
                  [14 15];
                  [12 16];
                  [16 17];
                  [17 18];
                  [18 19];
                  [16 20];
                  [20 21];
                  [21 22];
                  [22 23];
                  [24 25];
                  [25 26];
                  [25 27];
                  [25 43];
                  [26 27];
                  [26 43];
                  [27 28];
                  [28 29];
                  [29 30];
                  [27 31];
                  [31 32];
                  [32 33];
                  [33 34];
                  [31 35];
                  [35 36];
                  [36 37];
                  [37 38];
                  [35 39];
                  [39 40];
                  [40 41];
                  [41 42];
                  [39 43];
                  [43 44];
                  [44 45];
                  [45 46]];

%Change the location of file appropriately 
load('DataFile1.mat')

% initialize figure window and axis
window = figure('Name', 'Viewer', 'Units', 'Pixels', 'Position', [100 100 600 500]);     
set(gca, 'XColor', [0.5 0.5 0.5], 'YColor', [0.5 0.5 0.5], 'ZColor', [0.5 0.5 0.5]);  
view(-20, 20);
xlim([-400 400]); ylim([-400 400]); zlim([-400 400]);   
axis on; grid on;
ori = zeros(2,50);
hline = line(ori, ori, ori, 'LineWidth', eps); 
zoomFactor=1;

currentAction=1;
classLabel=labels{currentAction,1};

try
    % skeleton data
    %%Change the start as you wish to 
    %i = 4200;
    i = 1;
    while(i < size(skeleton,1))
    %for i = 1:size(skeleton,1) 
 
        % play skeleton data
        x = [skeleton{i}(CONNECTION_MAP(:,1),1) skeleton{i}(CONNECTION_MAP(:,2),1)];
        x =x*zoomFactor;
        y = [skeleton{i}(CONNECTION_MAP(:,1),2) skeleton{i}(CONNECTION_MAP(:,2),2)];
        y=y*zoomFactor;
        z = [skeleton{i}(CONNECTION_MAP(:,1),3) skeleton{i}(CONNECTION_MAP(:,2),3)]; 
        z=z*zoomFactor;
     
   %     axes(h_skeleton) 
       
        if(i>Anotations(currentAction,2))
            
            
            currentAction=currentAction+1;
            classLabel=labels{currentAction,1};
% %             while(strcmp(classLabel,'REPOS'))
% %                 
% %                 i=Anotations(currentAction,2);
% %                 currentAction=currentAction+1;
% %                 classLabel=labels{currentAction,1};
% %             
% %             end
                
            
        end

   
        title(sprintf('[Frame = %d,  Class= %s]', i,classLabel))
        for j = 1:length(hline)
            set(hline(j), 'XData', x(j,:), 'YData', y(j,:), 'ZData', z(j,:), 'LineWidth', 3);
        end
        
        drawnow   
        pause(0.0003)
      %waitforbuttonpress;
      i = i+1;
    end
catch err
    close('all')
    disp('force closed!')
end
