function [interval,T_total]=cover_time_point_target_(T,M1,v_dao,E,vd,te,R_yun)
    uM=-M1/norm(M1);%按照问题一的描述，计算导弹M1指向原点方向的单位向量
    %划分时间网格
    t0=te; t1=te+20;
    dt=1e-3;  %时间步长
    t=(t0:dt:t1).';
    %按时间顺序排序
    if t(end)<t1, t=[t;t1]; end

    %计算导弹M1每个时刻的坐标
    M1_x=M1(1)+v_dao*uM(1).*t;
    M1_y=M1(2)+v_dao*uM(2).*t;
    M1_z=M1(3)+v_dao*uM(3).*t;
    %计算云团每个时刻的坐标
    C_x=E(1)+0.*t;
    C_y=E(2)+0.*t;
    C_z=E(3)-vd.*(t-te);

    %用向量的方法，计算点到直线的距离
    A=T(:).';
    AB=[M1_x-A(1),M1_y-A(2),M1_z-A(3)];   %取线段AB
    CA=[C_x-A(1),C_y-A(2),C_z-A(3)];      %取线段CA
    L=sqrt(sum(AB.^2, 2));   %向量AB的模
    dotC=sum(CA.*AB, 2);
    Lambda=dotC./(L.^2);      %夹角的三角函数值                     
    cr=cross(CA, AB,2);
    ju_li=sqrt(sum(cr.^2,2)) ./ L;  %计算距离

    dA=sqrt(sum(CA.^2,2));  %向量CA的模
    CB=[C_x - M1_x, C_y-M1_y, C_z-M1_z];  %取一线段CB
    dB=sqrt(sum(CB.^2,2));  %向量CB的模

    panduan=(Lambda>= 0) & (Lambda <= 1);%投影是否在线段内
    d=ju_li;
    d(~panduan)=min(dA(~panduan), dB(~panduan));
    %检验标志
    mask=(d<=R_yun+1e-12);
    % 找边界，再进行细化
    edge=diff([0; mask; 0]);       % +1表示进入；-1 离开
    iL=find(edge == +1);
    iR=find(edge == -1) - 1;

    interval=zeros(numel(iL), 2);
    for k=1:numel(iL)
        ia=iL(k); ib=iR(k);
        tL=bisect_(@(x) d_at_(x)-R_yun, t(max(ia-1,1)), t(ia), 1e-10, 80);
        tR=bisect_(@(x) d_at_(x)-R_yun, t(ib), t(min(ib+1,numel(t))), 1e-10, 80);
        interval(k,:)=[tL, tR];
    end

    % 求出总时长
    T_total=sum(interval(:,2) - interval(:,1), 'omitnan');

    function val=d_at_(x)
        %计算导弹M1每个时刻的坐标
        Mx=M1(1)+v_dao*uM(1)*x;
        My=M1(2)+v_dao*uM(2)*x;
        Mz=M1(3)+v_dao*uM(3)*x;
        %计算云团每个时刻的坐标
        Cx=E(1);
        Cy=E(2);
        Cz=E(3)-vd*(x-te);

        AB1=[Mx-A(1),  My-A(2),  Mz-A(3)];    %取线段AB1
        CA1=[Cx-A(1),  Cy-A(2),  Cz-A(3)];     %取线段CA1
        L1 =norm(AB1);
        s1 =dot(CA1, AB1) / L1;              
        d_line1=norm(cross(CA1, AB1)) / L1;  %计算出直线距离

        if s1 >= 0 && s1 <= L1
            val=d_line1;                     % 线段内取点
        else
            val=min( norm(CA1), norm([Cx-Mx, Cy-My, Cz-Mz]) ); % 线段外取端点
        end
    end
end

%使用简单的二分法
function r=bisect_(fun, a, b, tol, itmax)
    fa=fun(a); fb=fun(b);
    if sign(fa)==0, r=a; return; end
    if sign(fb)==0, r=b; return; end
    if fa*fb > 0
        % 假如同号，那么就返回较小值所在的端点
        r=(abs(fa) <= abs(fb))*a +(abs(fa) > abs(fb))*b; 
        return;
    end
    for it=1:itmax
        m=0.5*(a+b); fm=fun(m);
        if fm==0 || (b-a) < tol, r=m; return; end
        if sign(fa)*sign(fm) <=0, b=m; fb=fm; else, a = m; fa = fm; end
    end
    r = 0.5*(a+b);
end
