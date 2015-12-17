function visualizeResults( Te )
%visualizeResults Show some exmple of false results

    nbImgToPlot = 30;
    
    % Plot the errors instead of sucess.
    figure(1000);
    nbFalseDetection = 0;
    i = 1;
    while nbFalseDetection < nbImgToPlot % Will crash if we have less than this number of error

        if Te.y(i) ~= Te.predictions(i) % Plot if different
            clf();

            img = imread( sprintf('train/imgs/train%05d.jpg', Te.idxs(i)) );
            imshow(img);

            % show if it is classified as pos or neg, and true label
            title(sprintf('%05d: Label: %d, Pred: %d', Te.idxs(i), Te.y(i), Te.predictions(i)));

            nbFalseDetection = nbFalseDetection + 1;
            pause;  % wait for keydo that then, 
        end

        i = i+1;
        if (i > length(Te.y))
            nbFalseDetection = nbImgToPlot +1; % Exit
        end
    end

end

