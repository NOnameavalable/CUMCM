function [drop, blow]=drop_blows_from_S_(S, FY1, g, R_yun) 
% 根据无人机航向与时间，得到每枚弹的投放点和起爆点
    uF=[cos(S.psi), sin(S.psi), 0]; % 无人机水平单位方向
    drop=zeros(3,3);
    blow=zeros(3,3);
    for i=1:3
        % 投放点（同一高度）
        drop(i,:) = FY1 + S.vF*S.d(i)*uF; 
        drop(i,3) = FY1(3);
        % 起爆点：延时 tau 期间，水平随机动、竖直自由落体
        tau = S.D(i);
        blow(i,1:2) = drop(i,1:2) + S.vF*tau*uF(1:2);
        blow(i,3)   = drop(i,3) - 0.5*g*tau^2;
    end
end
