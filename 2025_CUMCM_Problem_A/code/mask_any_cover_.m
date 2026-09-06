function covered=mask_any_cover_(S, tGrid, P, M1, vM, uM, blow, vd, R_yun, safety_ep)
% 对每个时间步，若任一云团完全遮蔽所有采样线段，则记为 true
    covered = false(size(tGrid));
    for k=1:numel(tGrid)
        t = tGrid(k);
        rM = M1 + vM*t*uM; % 导弹位置
        anyOn = false;
        for i=1:3
            if t>=S.e(i) && t<=S.e(i)+20
                c = blow(i,:) + [0,0,-vd*(t - S.e(i))]; % 云团中心随时间下沉
                if cloudFullyCovers_vec_(c, R_yun, rM, P, safety_ep)
                    anyOn = true; break;
                end
            end
        end
        covered(k) = anyOn;
    end
end
