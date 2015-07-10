%-------------------------------------------------------------------
% Simple introductionary example to particle smoothing
% in a linear Gaussian state space model
%
% Implements the Forward-filtering Backward-Smoother from
% the paper by Doucet, Godsill, Andrieu (2000) with the title
% "On sequential Monte Carlo sampling methods for Bayesian filtering"
%
% Written by: Johan Dahlin, Linköping University, Sweden
%             (johan.dahlin (at) isy.liu.se)
%
% Copyright (c) 2013 Johan Dahlin [ johan.dahlin (at) liu.se ]
%               
% Date:       2013-03-20
%
%-------------------------------------------------------------------

clear all;

%-------------------------------------------------------------------
% Parameters
%-------------------------------------------------------------------
sys.a=0.5;              % Scale parameter in the process model
sys.c=1;                % Scale parameter in the observation model
sys.sigmav=1.0;         % Standard deviation of the process noise
sys.sigmae=0.1;         % Standard deviation of the measurement noise
sys.T=100;              % Number of time steps
par.N=1000;             % Number of particles

%-------------------------------------------------------------------
% Generate data
%-------------------------------------------------------------------
% The system is
% x(t+1) = sys.a * x(t) + v(t),   v~N(0,sys.sigmav^2)
% y(t)   = sys.c * x(t) + e(t),   e~N(0,sys.sigmae^2)
%

% Set initial state
x(1)=0;            

for tt=1:sys.T
   x(tt+1) =sys.a*x(tt)+sys.sigmav*randn; 
   y(tt)   =sys.c*x(tt)+sys.sigmae*randn;
end
x=x(1:sys.T);

%-------------------------------------------------------------------
% Particle filter
%-------------------------------------------------------------------
p(:,1)=zeros(par.N,1);  % Set initial particle states

for tt=1:sys.T
   % Selection (resampling) and mutation (propagation)
%   y = randsample(n,k,true,w) or y = randsample(population,k,true,w) returns a weighted sample taken with replacement, using a vector of positive weights w, whose length is n. The probability that the integer i is selected for an entry of y is w(i)/sum(w). Usually, w is a vector of probabilities.
   if ~(tt==1)
      nIdx=randsample(par.N,par.N,'true',W(:,tt-1));
      % forward sampling the next hidden states, the importance distribution is the prior transition probability
      p(:,tt)=sys.a*p(nIdx,tt-1)+sys.sigmav*randn(par.N,1);
   end
   
   % Calculate log-weights (for increased accuracy)
   %w(:,tt)=normpdf(y(tt),sys.c*p(:,tt),sys.sigmae);
   % here is to get log version of normal distribution for p(y|x)
   w(:,tt) = -0.5*log(2*pi*sys.sigmae^2) - 0.5/sys.sigmae^2.*( y(tt)-sys.c*p(:,tt) ).^2;
   
   % this reason for this is to avoid truncation error for tiny numbers
   % Transform weights to usual base
   wmax = max( w(:,tt) );
   w(:,tt) = exp( w(:,tt) - wmax );
   
   % Normalise of weights
   W(:,tt)=w(:,tt)/sum(w(:,tt));
   
   % Calculate state estimate
   xhatPF(tt)=W(:,tt)'*p(:,tt);
end

%-------------------------------------------------------------------
% Particle smoother (Forward-filtering Backward-Smoother) (FFBSm)
%-------------------------------------------------------------------
Ws(:,sys.T)=W(:,sys.T);
xhatPS(sys.T)=xhatPF(sys.T);

for tt=sys.T-1:-1:1
    % Compute the normalisation term
    for jj=1:par.N; v(jj,tt)=sum(W(:,tt).*normpdf(p(jj,tt+1),sys.a*p(:,tt),sys.sigmav)); end
    
    % Compute the smoothing weight
    for ii=1:par.N; Ws(ii,tt)=W(ii,tt)*sum(Ws(:,tt+1).*normpdf(p(:,tt+1),sys.a*p(ii,tt),sys.sigmav)./v(:,tt)); end
    
    % Compute the state estimate
    xhatPS(tt)=Ws(:,tt)'*p(:,tt);
end

%-------------------------------------------------------------------
% Plot the true and estimated states
%-------------------------------------------------------------------
plot(1:sys.T,x,'k',1:sys.T,xhatPS,'r');
xlabel('time'); ylabel('latent state (x)');
legend('true','PF est.');
