%% 4.2.1 Sinusoidal Signal Generation
clear; close all; clc;

%% Generate signals
t  = ((2*pi)/100):((2*pi)/100):10*pi;
y1 = sin(t);
y2 = sin(0.5*t);
y3 = sin(4*t);

%% Part (a): auto-convolution and cross-convolution
c11 = conv(y1, y1, 'full');   % auto-convolution of y1
c12 = conv(y1, y2, 'full');   % convolution of y1 and y2
c13 = conv(y1, y3, 'full');   % convolution of y1 and y3

n_conv = 1:length(c11);

figure('Color','w');
plot(n_conv, c11, 'LineWidth', 1.6); hold on;
plot(n_conv, c12, 'LineWidth', 1.6);
plot(n_conv, c13, 'LineWidth', 1.6);
grid on;
xlabel('Sample Index');
ylabel('Amplitude');
title('Auto- and Cross-Convolution of Sinusoidal Signals');
legend('conv(y_1,y_1)', 'conv(y_1,y_2)', 'conv(y_1,y_3)', 'Location', 'best');

%% Optional: show original signals
figure('Color','w');
plot(t, y1, 'LineWidth', 1.4); hold on;
plot(t, y2, 'LineWidth', 1.4);
plot(t, y3, 'LineWidth', 1.4);
grid on;
xlabel('t');
ylabel('Amplitude');
title('Original Sinusoidal Signals');
legend('y_1 = sin(t)', 'y_2 = sin(0.5t)', 'y_3 = sin(4t)', 'Location', 'best');

%% Part (b): classify y1, y2, y3 using convolution
class1 = classify_sine_conv(y1, y1, y2, y3);
class2 = classify_sine_conv(y2, y1, y2, y3);
class3 = classify_sine_conv(y3, y1, y2, y3);

fprintf('y1 is classified as Class %d\n', class1);
fprintf('y2 is classified as Class %d\n', class2);
fprintf('y3 is classified as Class %d\n', class3);

%% Test with a noisy signal if you want
y_test = y2 + 0.05*randn(size(y2));
class_test = classify_sine_conv(y_test, y1, y2, y3);
fprintf('Noisy y2 is classified as Class %d\n', class_test);