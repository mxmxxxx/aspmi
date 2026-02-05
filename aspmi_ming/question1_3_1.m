close all
clear all
clc

%% Generation of complex-valued signal with complex noise
fs = 1;
N  = [20, 30, 40, 80, 90, 110];

% Colour palette (consistent & report-friendly)
c1 = [0.00 0.45 0.74];   % blue
c2 = [0.85 0.33 0.10];   % orange
c3 = [0.47 0.67 0.19];   % green
c4 = [0.64 0.08 0.18];   % red
c5 = [0.49 0.18 0.56];   % purple
c6 = [0.30 0.30 0.30];   % dark grey

lw = 1.6;

figure(1);

%% -------------------------------
% Left subplot: N = 20, 30, 40
%% -------------------------------
subplot(1,2,1)
hold on
cols_left = {c1, c2, c3};

for i = 1:3
    n = 0:N(i)-1;
    noise = 0.2/sqrt(2) * (randn(size(n)) + 1j*randn(size(n)));
    x = exp(1j*2*pi*0.3*n) + exp(1j*2*pi*0.32*n) + noise;

    % Periodogram
    dF = fs/N(i);
    dF_new = fs/512;
    K = dF_new/dF;

    [pxx,f] = periodogram(x, rectwin(N(i)), round(N(i)/K), fs);
    plot(f*1000, pow2db(pxx), 'LineWidth', lw, 'Color', cols_left{i});
end

xlabel('Frequency (mHz)','FontSize',11)
ylabel('Power/Frequency (dB/Hz)','FontSize',11)
title('PSD estimates of complex exponentials with noise','FontSize',11)
legend('N = 20','N = 30','N = 40','FontSize',9,'Location','best')
grid on
xlim([0.2 0.4]*1000)

%% -------------------------------
% Right subplot: N = 80, 90, 110
%% -------------------------------
subplot(1,2,2)
hold on
cols_right = {c4, c5, c6};

for i = 4:6
    n = 0:N(i)-1;
    noise = 0.2/sqrt(2) * (randn(size(n)) + 1j*randn(size(n)));
    x = exp(1j*2*pi*0.3*n) + exp(1j*2*pi*0.32*n) + noise;

    % Periodogram
    dF = fs/N(i);
    dF_new = fs/512;
    K = dF_new/dF;

    [pxx,f] = periodogram(x, rectwin(N(i)), round(N(i)/K), fs);
    plot(f*1000, pow2db(pxx), 'LineWidth', lw, 'Color', cols_right{i-3});
end

xlabel('Frequency (mHz)','FontSize',11)
ylabel('Power/Frequency (dB/Hz)','FontSize',11)
title('PSD estimates of complex exponentials with noise','FontSize',11)
legend('N = 80','N = 90','N = 110','FontSize',9,'Location','best')
grid on
xlim([0.2 0.4]*1000)
