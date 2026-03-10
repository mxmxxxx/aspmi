%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                    Task 1.6                        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
clear all
clc

%% Loading data
load 'PCAPCR.mat';
Nx = (sqrt(4)).*randn(1000,10);
Xnoise = X + Nx;

C = cov(X(:,1),Nx(:,1));

%% Using Singular Value Decomposition (svd) on X and Xnoise
[U_X,S_X,V_X] = svd(X);
s_X = svd(X);

[U_Xnoise,S_Xnoise,V_Xnoise] = svd(Xnoise);
s_Xnoise = svd(Xnoise);

%% =======================
%% Part a)
%% =======================
figure;
stem(s_Xnoise,'k','LineWidth',1,'Marker','o','MarkerSize',5,'MarkerFaceColor','w');
hold on
stem(s_X,'r--','LineWidth',1,'Marker','o','MarkerSize',5,'MarkerFaceColor','w');
grid on
xlabel('Singular value Index','FontSize',11)
ylabel('Singular value magnitude','FontSize',11)
legend('Matrix X_{noise}','Matrix X');
title('Singular spectrum of Matrices X and X_{noise}','FontSize',11)

% Square error of singular values
sq_error = (s_X - s_Xnoise).^2;

figure;
stem(sq_error,'b','LineWidth',1,'Marker','o','MarkerSize',5,'MarkerFaceColor','w');
xlabel('Singular value Index','FontSize',11)
ylabel('Square error','FontSize',11)
grid on
title('Square error between singular values of X and X_{noise}','FontSize',11)

%% =======================
%% Part b)
%% =======================
p = 3; % Rank of Xnoise
s = svds(Xnoise,p);

U_approx = U_Xnoise(:,1:p);
VT_Xnoise = V_Xnoise';
VT_approx = VT_Xnoise(1:p,:);
S_approx = diag(s);

Xdenoised = U_approx * S_approx * VT_approx;

% Errors
err_noise = abs(vecnorm(X - Xnoise)).^2;
err_denoised = abs(vecnorm(X - Xdenoised)).^2;

figure;
stem(err_noise,'k','LineWidth',1,'Marker','o','MarkerSize',5,'MarkerFaceColor','w');
hold on
stem(err_denoised,'r--','LineWidth',1,'Marker','o','MarkerSize',5,'MarkerFaceColor','w');
grid on
xlabel('Index of Input Variable','FontSize',11)
ylabel('Square Error','FontSize',11)
legend('Noise-corrupted','Denoised');
title('Error between noiseless input matrix with noise corrupted and denoised matrices','FontSize',11)

%% =======================
%% Part c)
%% =======================

% OLS Solution
B_ols = inv(Xnoise'*Xnoise)*Xnoise'*Y;
Y_ols = Xnoise*B_ols;

error_ols = (1/size(Y,1)) .* (abs(vecnorm(Y - Y_ols)).^2);

% PCR Solution
B_PCR = VT_approx' * inv(S_approx) * U_approx' * Y;
Y_PCR = Xnoise * B_PCR;

error_PCR = (1/size(Y,1)) .* (abs(vecnorm(Y - Y_PCR)).^2);

figure;
stem(error_ols,'k','LineWidth',1,'Marker','o','MarkerSize',5,'MarkerFaceColor','w');
hold on
stem(error_PCR,'r--','LineWidth',1,'Marker','o','MarkerSize',5,'MarkerFaceColor','w');
grid on
xlabel('Index of Input Variable','FontSize',11)
ylabel('Square Error','FontSize',11)
legend('OLS','PCR');
title('Square error between OLS solution and PCR solution - Training set','FontSize',11)

total_error_ols = sum(error_ols);
total_error_PCR = sum(error_PCR);

%% =======================
%% Test set
%% =======================
Y_ols2 = Xtest * B_ols;
error_ols2 = (1/size(Ytest,1)) .* (abs(vecnorm(Y - Y_ols2)).^2);

Y_PCR2 = Xtest * B_PCR;
error_PCR2 = (1/size(Ytest,1)) .* (abs(vecnorm(Y - Y_PCR2)).^2);

figure;
stem(error_ols2,'k','LineWidth',1,'Marker','o','MarkerSize',5,'MarkerFaceColor','w');
hold on
stem(error_PCR2,'r--','LineWidth',1,'Marker','o','MarkerSize',5,'MarkerFaceColor','w');
grid on
xlabel('Index of Input Variable','FontSize',11)
ylabel('Square Error','FontSize',11)
legend('OLS','PCR');
title('Square error between OLS solution and PCR solution - Test set','FontSize',11)

total_error_ols2 = sum(error_ols2);
total_error_PCR2 = sum(error_PCR2);

%% =======================
%% Part d)
%% =======================
N_iter = 500;

for i = 1:N_iter
    % OLS
    [Yest_ols, Ynew] = regval(B_ols);
    ErrorOLS(i,:) = abs(vecnorm(Ynew - Yest_ols)).^2;

    % PCR
    [Yest_PCR, Ynew] = regval(B_PCR);
    ErrorPCR(i,:) = abs(vecnorm(Ynew - Yest_PCR)).^2;
end

MSE_OLS = mean(ErrorOLS)./size(Ynew,1);
MSE_PCR = mean(ErrorPCR)./size(Ynew,1);

figure;
stem(MSE_OLS,'k','LineWidth',1,'Marker','o','MarkerSize',5,'MarkerFaceColor','w');
hold on
stem(MSE_PCR,'r--','LineWidth',1,'Marker','o','MarkerSize',5,'MarkerFaceColor','w');
grid on
xlabel('Index of Input Variable')
ylabel('MSE value')
legend('OLS','PCR');
title('Mean Squared Error comparison: OLS vs PCR')
