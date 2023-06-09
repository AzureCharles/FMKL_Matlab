function [cm,precision,recall,f1_score]=getF1Score(y_test,y_pred)
    
    cm=confusionmat(y_test,y_pred);
    % tp=0;tn=0;fp=0;fn=0;
    % for i=1:size(y_test,1)
    %     if y_pred(i)==y_test(i)
    %         if y_pred(i)>0
    %             tp=tp+1;
    %         else
    %             tn=tn+1;
    %         end
    %     else
    %         if y_pred(i)>0
    %             fp=fp+1;
    %         else
    %             fn=fn+1;
    %         end
    %     end
    % end
    precision=diag(cm)./sum(cm,2);
    recall=diag(cm)./sum(cm,1)';
    f1_score=2*precision.*recall./(precision+recall);
    % mat=[precision;recall;f1_score];
end
