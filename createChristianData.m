function [] = createChristianData(outDir)
    % Loading the data.
    load('dataForAutoEncoder.mat')
    clear('LEIFallLabels','LEIFallMat')
    
    % Removing problematic neurons 
    shouldRemove = ones(length(NORMAllLabels),1);
    
    % Neuron data dimensions
    neuronsTime = size(NORMallMat,2);
    neuronsCount = size(NORMallMat,1);
    
    START_MARGIN = 20;
    END_MARGIN = neuronsTime - 40;
    
    for i=1:length(NORMAllLabels)
        currentLabel = NORMAllLabels(i,:);
        
        % Trying to check if the name is a number or not.
        [val, status] = str2num(currentLabel{1});
        
        if (status == 0)
            if all(~isnan(NORMallMat(i,START_MARGIN:END_MARGIN)))
                shouldRemove(i) = 0;
            end 
        end
    end
       
    % Removing the useless lines.
    NORMAllLabels(logical(shouldRemove),:) = [];
    NORMallMat(logical(shouldRemove),:) = [];
    
    newSize = size(NORMallMat,1);
    
    for i=1:newSize
        currentLabel = NORMAllLabels(i,:);
        newName = sprintf('{%s}_{%s}_{%s}',currentLabel{1}, currentLabel{4}, currentLabel{5});
        
        act = NORMallMat(i,START_MARGIN:END_MARGIN);
        save(fullfile(outDir, newName),'act');
        
    end
    

end

