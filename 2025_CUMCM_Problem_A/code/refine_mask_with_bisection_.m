function [interval, total_len]=refine_mask_with_bisection_(mask,tGrid,pred,tol,maxit)
%需要使用的辅助函数之一
    dt=tGrid(2)-tGrid(1);
    idx=find(mask);  %寻找标志
    interval =zeros(0,2); 
    total_len=0; %总的长度
    if isempty(idx), return; end

    % 找真区间的分组
    jumps = [1, find(diff(idx)>1)+1, numel(idx)+1];
    for s=1:numel(jumps)-1
        seg=idx(jumps(s):jumps(s+1)-1);
        tR=tGrid(seg(1));  tL = max(0, tR - dt);
        bL=pred(tL); bR = pred(tR);
        if ~(bL==false && bR==true)
            t_left = tR;
        else
            for k=1:maxit
                tm=0.5*(tL+tR);
                if pred(tm)==bR, tR=tm; else, tL=tm; end
                if (tR - tL) <= tol, break; end
            end
            t_left=tR;
        end

        tL = tGrid(seg(end));  tR = min(tGrid(end)+dt, tL + dt);
        bL = pred(tL); bR = pred(tR);
        if ~(bL==true && bR==false)
            t_right = tL; % 退化
        else
            for k=1:maxit
                tm = 0.5*(tL+tR);
                if pred(tm)==bL, tL=tm; else, tR=tm; end
                if (tR - tL) <= tol, break; end
            end
            t_right = tL;
        end

        interval(end+1,:)=[t_left, t_right]; 
        total_len=total_len+(t_right - t_left);
    end
end
