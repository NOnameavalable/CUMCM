function P = sampleCylinderPointsHP_(c0, R, H, n_th_side, n_z_side, n_Th_cap, n_rad_cap)
% 需要使用的辅助函数，用于生成随机采样点
    % 侧面
    the = linspace(0,2*pi,n_th_side+1); the(end)=[];   % 表示出角度
    zs  = linspace(c0(3), c0(3)+H, n_z_side);          % 表示出高度
    P_side = zeros(n_th_side*n_z_side,3); idx=0;
    for z = zs
        for th = the
            idx=idx+1; 
            P_side(idx,:) = [c0(1)+R*cos(th), c0(2)+R*sin(th), z];
        end
    end
    %有上下两个地面取
    rs   = linspace(0,R,n_rad_cap);
    the2 = linspace(0,2*pi,n_Th_cap+1); the2(end)=[];
    P_cap = zeros(numel(rs)*numel(the2)*2,3); idx2=0;
    for z = [c0(3), c0(3)+H]
        for r = rs
            for th = the2
                idx2=idx2+1; 
                P_cap(idx2,:) = [c0(1)+r*cos(th), c0(2)+r*sin(th), z];
            end
        end
    end
    P_cap = P_cap(1:idx2,:);

    % 中高度附件的额外极值
    extra = [c0(1)+R, c0(2),     c0(3)+H/2;
             c0(1)-R, c0(2),     c0(3)+H/2;
             c0(1),   c0(2)+R,   c0(3)+H/2;
             c0(1),   c0(2)-R,   c0(3)+H/2];
%去除重合的部分
    P = unique([P_side; P_cap; extra], 'rows');
end
