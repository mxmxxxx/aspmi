clc
close all
clear all

%% Loading EEG files
load EEG_Data_Assignment1.mat
Tsample = 1/fs;
taxis = (1:length(POz))*Tsample;
N = length(POz);

% Remove mean value
POz = POz - mean(POz);

figure;
plot(taxis,POz)
grid on
title('Plot of EEG signal vs time')
xlabel('Time (s)'); ylabel('Amplitude (V)')

%% Applying the standard periodogram approach
[psd_POz1,fshift1] = periodogram(POz,rectwin(N),N,fs,'onesided');
psd_POz1 = pow2db(psd_POz1); % Convert to dB

figure;
plot(fshift1,psd_POz1)
xlim([0 60])
ylim([-150 -80])
xlabel('Frequency(Hz)')
ylabel('Power/Frequency(dB/Hz)')
title('Standard Periodogram approach')

%% Reducing DFT samples to 10 per Hz.
% For this case, dF is 0.0125 and we want dF=0.1 Hz.

N2 = round(N/8);

[psd_POz2, fshift2] = periodogram(POz,rectwin(N),N2,fs,'onesided');
psd_POz2 = pow2db(psd_POz2);

figure(2);
subplot(2,1,1)
plot(fshift2,psd_POz2,'Linewidth',1)
grid on
xlim([0 60])
ylim([-150 -80])
xlabel('Frequency(Hz)','FontSize',11)
ylabel('Power/Frequency(dB/Hz)','FontSize',11)
title('Standard Periodogram approach','FontSize',11)

%% With and without reducing DFT samples per Hz
figure;
plot(fshift1,psd_POz1,fshift2,psd_POz2)
xlim([0 60])

%% Constructing windows - Use of pwelch() with no overlap.

% Window size 10s.
size1 = round(10/Tsample); % Convert 10 seconds into sample size (integer)
[psd_10s,f_10s] = pwelch(POz,rectwin(size1),0,N2,fs,'onesided');
psd_10s = pow2db(psd_10s);
figure;
plot(f_10s,psd_10s)
xlim([0 60])

% Window size 5s.
size2 = round(5/Tsample);
[psd_5s,f_5s] = pwelch(POz,rectwin(size2),0,N2,fs,'onesided');
psd_5s = pow2db(psd_5s);
figure;
plot(f_5s,psd_5s)
xlim([0 60])

% Window size 1s.
size3 = round(1/Tsample);
[psd_1s,f_1s] = pwelch(POz,rectwin(size3),0,N2,fs,'onesided');
psd_1s = pow2db(psd_1s);
figure;
plot(f_1s,psd_1s)
xlim([0 60])

%% Three windows on the same plot
figure(2);
subplot(2,1,2)
plot(f_10s,psd_10s,'linewidth',1)
grid on
hold on
plot(f_5s,psd_5s,'linewidth',1)
plot(f_1s,psd_1s,'linewidth',1)
xlim([0 60])
xlabel('Frequency(Hz)','FontSize',11)
ylabel('Power/Frequency(dB/Hz)','FontSize',11)
title('Averaged Periodogram approach','FontSize',11)

% ---- FIXED LEGEND (no invalid \Deltat) ----
legend({'\Delta t = 10 s','\Delta t = 5 s','\Delta t = 1 s'}, ...
       'FontSize',11,'Orientation','horizontal','Interpreter','tex');

%% Standard vs 10s | Standard vs 1s
figure(3)
subplot(1,2,1)
plot(fshift2,psd_POz2,'Linewidth',1)
hold on
plot(f_10s,psd_10s,'r','linewidth',1)
xlim([0 60])
ylim([-150 -80])
xlabel('Frequency(Hz)','FontSize',11)
ylabel('Power/Frequency(dB/Hz)','FontSize',11)

% ---- FIXED LEGEND ----
legend({'Standard Method','\Delta t = 10 s'}, 'Interpreter','tex')

figure(3)
subplot(1,2,2)
plot(fshift2,psd_POz2,'Linewidth',1)
hold on
plot(f_1s,psd_1s,'r','linewidth',1)
xlim([0 60])
ylim([-150 -80])
xlabel('Frequency(Hz)','FontSize',11)
ylabel('Power/Frequency(dB/Hz)','FontSize',11)

% ---- FIXED LEGEND ----
legend({'Standard Method','\Delta t = 1 s'}, 'Interpreter','tex')
