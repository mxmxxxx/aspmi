%% Q4 – From LMS to Deep Learning
% Cleaned + reformatted version

clear; close all; clc
load('time-series.mat')

%% Global Plot Style
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(groot,'defaultTextInterpreter','latex')
set(groot,'defaultLegendInterpreter','latex')

blue  = [0 0.447 0.741];
red   = [0.850 0.325 0.098];
green = [0.466 0.674 0.188];

lw = 1.8;

%% Data preparation
y = y - mean(y);
N = length(y);
n = 1:N;

%% =========================================================
%% Q4.1 LMS Prediction
%% =========================================================

mu = 1e-5;
gamma = 0;
order = 4;

[yhat,w,error] = LMS(y,mu,gamma,order);

MSE = mean(error.^2);
MSE_dB = 10*log10(MSE);
Rp = 10*log10(var(yhat)/var(error));

%% Figure 1 – LMS prediction
figure('Position',[200 200 900 500])

subplot(2,1,1)
plot(n,y,'Color',blue,'LineWidth',lw); hold on
plot(n,yhat,'Color',red,'LineWidth',lw)
title('LMS One-Step Ahead Prediction')
xlabel('Time index $n$')
ylabel('Amplitude')
legend('True signal','Prediction')
grid on

subplot(2,1,2)
plot(750:1000,y(750:1000),'Color',blue,'LineWidth',lw); hold on
plot(750:1000,yhat(750:1000),'Color',red,'LineWidth',lw)
title('Zoomed Prediction')
xlabel('Time index $n$')
ylabel('Amplitude')
legend('True','Prediction')
grid on

sgtitle('Question 4.1 — LMS Prediction')

%% =========================================================
%% Q4.2 Dynamical Perceptron
%% =========================================================

alpha = 1;

[yhat,w,error] = LMS_dypn(y,mu,order,alpha,0,zeros(order,1),0);

figure('Position',[200 200 900 500])

subplot(2,1,1)
plot(n,y,'Color',blue,'LineWidth',lw); hold on
plot(n,yhat,'Color',red,'LineWidth',lw)
title('Dynamical Perceptron Prediction')
xlabel('Time index $n$')
ylabel('Amplitude')
legend('True','Prediction')
grid on

subplot(2,1,2)
plot(750:1000,y(750:1000),'Color',blue,'LineWidth',lw); hold on
plot(750:1000,yhat(750:1000),'Color',red,'LineWidth',lw)
title('Zoomed Prediction')
xlabel('Time index $n$')
ylabel('Amplitude')
legend('True','Prediction')
grid on

sgtitle('Question 4.2 — Dynamical Perceptron')

%% =========================================================
%% Q4.3 Optimising alpha
%% =========================================================

mu = 1e-7;

alphas = 40:0.1:100;

MSEvals = zeros(length(alphas),1);
Rpvals = zeros(length(alphas),1);

for i = 1:length(alphas)

    alpha = alphas(i);

    [yhat,~,error] = LMS_dypn(y,mu,order,alpha,0,zeros(order,1),0);

    MSEvals(i) = 10*log10(mean(abs(error).^2));
    Rpvals(i) = 10*log10(var(yhat)/var(error));

end

figure('Position',[200 200 900 400])

subplot(1,2,1)
plot(alphas,MSEvals,'Color',blue,'LineWidth',2)
xlabel('$\alpha$')
ylabel('MSE (dB)')
title('MSE vs $\alpha$')
grid on

subplot(1,2,2)
plot(alphas,Rpvals,'Color',green,'LineWidth',2)
xlabel('$\alpha$')
ylabel('Prediction Gain (dB)')
title('Prediction Gain vs $\alpha$')
grid on

sgtitle('Question 4.3 — Optimal $\alpha$')

%% ================================
% Prediction vs True Signal
% ================================

blue  = [0 0.447 0.741];
red   = [0.850 0.325 0.098];
green = [0.466 0.674 0.188];
figure('Position',[200 200 900 500])

subplot(2,1,1)

plot(n,y,'Color',blue,'LineWidth',lw); hold on
plot(n,yhat,'Color',red,'LineWidth',lw)

title('Biased Dynamical Perceptron Prediction')
xlabel('Time index $n$')
ylabel('Amplitude')

legend('True Signal','Prediction','Location','best')

grid on
grid minor


subplot(2,1,2)

idx = 750:1000;

plot(idx,y(idx),'Color',blue,'LineWidth',lw); hold on
plot(idx,yhat(idx),'Color',red,'LineWidth',lw)

title('Zoomed Prediction (Final 250 Samples)')
xlabel('Time index $n$')
ylabel('Amplitude')

legend('True Signal','Prediction','Location','best')

grid on
grid minor

sgtitle('Question 4.4 — Biased Dynamical Perceptron')


%% ================================
% Weight Evolution
% ================================

figure('Color','w')

plot(w','LineWidth',1.5)

title('Weight Evolution for Biased Dynamical Perceptron')
xlabel('Time Index (n)')
ylabel('Weight Value')

legend({'w_0 (bias)','w_1','w_2','w_3','w_4'},'Location','best')

grid on
grid minor

%% =========================================================
%% Q4.5 Pretraining
%% =========================================================

segLength = 20;
epochs = 100;
b = 1;

alphas = 40:0.1:100;

MSEvals = zeros(length(alphas),1);
Rpvals = zeros(length(alphas),1);

for i = 1:length(alphas)

    alpha = alphas(i);
    winit = zeros(order+b,1);

    for e = 1:epochs

        yseg = y(1:segLength);

        for k = order+1:segLength

            x = [1; yseg(k-1); yseg(k-2); yseg(k-3); yseg(k-4)];

            ypred = alpha*tanh(winit'*x);

            err = yseg(k)-ypred;

            grad = alpha*(1-(ypred/alpha)^2);

            winit = winit + mu*grad*err*x;

        end

    end

    [yhat,~,error] = LMS_dypn(y,mu,order,alpha,0,winit,b);

    MSEvals(i) = 10*log10(mean(abs(error).^2));
    Rpvals(i) = 10*log10(var(yhat)/var(error));

end

figure('Position',[200 200 900 400])

subplot(1,2,1)
plot(alphas,MSEvals,'Color',blue,'LineWidth',2)
xlabel('$\alpha$')
ylabel('MSE (dB)')
title('MSE vs $\alpha$')
grid on

subplot(1,2,2)
plot(alphas,Rpvals,'Color',green,'LineWidth',2)
xlabel('$\alpha$')
ylabel('Prediction Gain (dB)')
title('Prediction Gain vs $\alpha$')
grid on

sgtitle('Question 4.5 — Pretrained Perceptron')

%% Final prediction

alpha = 67.9;

[yhat,~,error] = LMS_dypn(y,mu,order,alpha,0,winit,b);

figure('Position',[200 200 900 500])

subplot(2,1,1)
plot(n,y,'Color',blue,'LineWidth',lw); hold on
plot(n,yhat,'Color',red,'LineWidth',lw)
title('Final Prediction')
xlabel('Time index $n$')
ylabel('Amplitude')
legend('True','Prediction')
grid on

subplot(2,1,2)
plot(1:50,y(1:50),'Color',blue,'LineWidth',lw); hold on
plot(1:50,yhat(1:50),'Color',red,'LineWidth',lw)
title('Early Prediction')
xlabel('Time index $n$')
ylabel('Amplitude')
legend('True','Prediction')
grid on

sgtitle('Final Model Performance')

%% Question 4.5

% pre-train the weights by over-fitting to a small number of samples
% starting with w(0) = 0 and using 100 iterations to fir the first 20
% samples t yield w_init

% then use w_init to predict the entire time series

load('time-series.mat')
% Selecting an optimal value of alpha, ranging between 40 and 100 for our guess
% building upon the LMS algorithm to add a dynamical perceptron
sampNo = length(y);
n = 1: sampNo;
mu = 0.0000001;
gamma = 0;
order = 4;
starter =  40;
step = 0.1;
ender = 100;
alphas = [starter:step:ender];
MSEs= [];
R_ps = [];
step = mu; % overwritting for the pretraining part..!
b = 1;
segLength = 20;
epochs = 100;
winit = zeros(order+b,1); % the first initialisation will just be winit = [0 0 0 0]'
all_winits = zeros(order+b,length(alphas));
count = 1;
MSEs = [];
R_s = [];
for alpha = alphas
    
    % training on the first 20 samples with the added bias
    for e = 1: epochs
        y_sample = y(1:segLength); % accounting for the ones added for bias
        
        % to pre-train
            xOut = zeros(length(y_sample),1);
            err = xOut;
            w = zeros(order+b,length(y_sample)+1); % since order can be > 1 (here 2)
            w(:,1) = winit;
            xShift = zeros(order,length(y_sample)); % x(n-k)
            % creating two shifted vectors by i, i.e. the order length
            for i = 1: order
                xShift(i,:) = [ zeros(1,i), y_sample(1: length(y_sample)-i)']; 
            end
            if b
                xShift = [ones(1,length(y_sample)); xShift];
            end

            for k = 1: length(y_sample)

                % calculate the prediction
                xOut(k) = alpha*tanh(w(:,k)'*xShift(:,k));
                err(k) = y_sample(k)-xOut(k);
                act_function = alpha*(1-(xOut(k)/alpha)^2);
                % update
                w(:,k+1)=w(:,k)+(step*act_function*err(k)).*xShift(:,k);
  
            end
            w =  w(:,2:end);
            winit = w(:,end);
    end
    
    all_winits(:,count) = winit;
    
    % now going ahead with the normal algorithm
    [yhat,w,error] = LMS_dypn(y,mu,order,alpha,0,winit,b);
    % calculating the MSE between the true and estimated signals
    MSEs = [MSEs, 10*log10(mean(abs(error(order+1:end)).^2))];
    %MSE_db = 10*log10(MSE);
    R_ps = [R_ps,10*log10(var(yhat(order+1:end))/var(error(order+1:end)))];
    
    count = count+1;
end
%%
figure
subplot(1,2,1)
step = 0.1;
plot(alphas,MSEs,'b','LineWidth',2)
hold on
[val,ind] = min(MSEs);
plot(starter + ind*step,val,'r*','MarkerSize',10)
xlabel('$\alpha$','fontsize',14)
ylabel('Mean Squared Error (dB)','fontsize',14)
grid on 
grid minor
subplot(1,2,2)
plot(alphas,R_ps,'b','LineWidth',2)
hold on
[val,ind] = max(R_ps);
plot(starter + ind*step,val,'r*','MarkerSize',10)
xlabel('$\alpha$','fontsize',14)
ylabel('Prediction Gain(dB)','fontsize',14)
grid on 
grid minor
sgtitle('Finding the Optimal Value for $\alpha$','Interpreter','Latex','fontsize',20)
set(gcf,'color','w')




%%

load('time-series.mat')
% now taking the winits @ alpha = 67.9
sampNo = length(y);
n = 1: sampNo;
mu = 0.0000001;
gamma = 0;
order = 4;
starter =  40;
step = 0.1;
ender = 100;
alpha = 67.9;
MSEs= [];
R_ps = [];


winit = all_winits(:,find(alphas == 67.9));
[yhat,w,error] = LMS_dypn(y,mu,order,alpha,0,winit,b);

% calculating the MSE between the true and estimated signals
MSE = 10*log10(mean(abs(error).^2));
%MSE_db = 10*log10(MSE);
R_ps = [R_ps,10*log10(var(yhat)/var(error))];

MSE_end = mean(error(750:1000).^2);
MSE_db_end = 10*log10(MSE);
R_p_end = 10*log10(var(yhat(750:1000))/var(error(750:1000)));

figure
plot(y,'b','LineWidth',1.5)
hold on
plot(yhat,'r','LineWidth',1.5)
xlabel('Time Index (n)','fontsize',18)
ylabel('(AU)','fontsize',18)
title('One-Step-Ahead Prediction of AR(4) Time Series','Interpreter','latex','fontsize',18)
legend('True Centred Time Series','Dynamical Perceptron Estimated Time Series')
ax = gca;
ax.FontSize = 15; 
grid on
grid minor
set(gcf,'color','w')

% showing that when we approcahing the end of the signal, the prediction
% improves
figure
plot(1:50,y(1:50),'b','LineWidth',1.5)
hold on
plot(1:50,yhat(1:50),'r','LineWidth',1.5)
xlim([1,50])
xlabel('Time Index (n)','fontsize',18)
ylabel('(AU)','fontsize',18)
title('One-Step-Ahead Prediction of AR(4) Time Series','Interpreter','latex','fontsize',18)
legend('Centred Time Series','Estimated Time Series')
ax = gca;
ax.FontSize = 15; 
grid on
grid minor
set(gcf,'color','w')