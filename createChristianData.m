function [] = createChristianData(outDir)
    % Loading the data.
    load('dataForAutoEncoder.mat')
    
    % Removing problematic neurons 
    shouldRemove = ones(length(LEIFallLabels),1);
    
    % Neuron data dimensions
    neuronsTime = size(LEIFallMat,2);
    neuronsCount = size(LEIFallMat,1);
    
    START_MARGIN = 20;
    END_MARGIN = neuronsTime - 40;
    
    for i=1:length(LEIFallLabels)
        currentLabel = LEIFallLabels(i,:);
        
        % Trying to check if the name is a number or not.
        [val, status] = str2num(currentLabel{1});
        
        if (status == 0)
            if all(~isnan(LEIFallMat(i,START_MARGIN:END_MARGIN)))
                shouldRemove(i) = 0;
            end 
        end
    end
       
    % Removing the useless lines.
    LEIFallLabels(logical(shouldRemove),:) = [];
    LEIFallMat(logical(shouldRemove),:) = [];
    
    newSize = size(LEIFallMat,1);
    
    for i=1:newSize
        currentLabel = LEIFallLabels(i,:);
        newName = sprintf('{%s}_{%s}_{%s}',currentLabel{1}, currentLabel{4}, currentLabel{5});
        
        act = LEIFallMat(i,START_MARGIN:END_MARGIN);
        save(fullfile(outDir, newName),'act');
        
    end
    

end

