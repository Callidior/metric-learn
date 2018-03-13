#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg
import time

def cpGradientDiag( X=None, S=None, batchsize=None):
    #dPsi = diag( X * S* X') / batchsize
    dPsi = np.diag( np.dot(np.dot(X, S), X.conj().transpose()) )/ batchsize
    return dPsi

def cpGradientDiagMKL(X=None, S=None, batchsize=None):

    #[d,n,m] = size(X);
    (d,n,m) = X.shape()

    #dPsi    = zeros(d,m);
    dPsi = np.zeros((d,m),float)
    #for i = 1:m
    #    dPsi(:,i)    = diag(X(:,:,i) * S * X(:,:,i)');
    #end
    for i in range(m):
        dPsi[:,:,i] = np.diag(np.dot(np.dot(X[:,:,i], S), X[:,:,i].conj().transpose()))
    
    #dPsi = dPsi / batchsize;
    dPsi = dPsi / batchsize
    return dPsi

def cpGradientFull(X=None, S=None, batchsize=None):

    #dPsi    = X * S * X' / batchSize;
    dPsi = np.dot( np.dot(X, S), X.conj().transpose()) / batchsize
    return dPsi

def cpGradientFullMKL(X=None, S=None, batchsize=None):

    #[d,n,m] = size(X);s
    (d,n,m) = X.shape()

    #dPsi = zeros(d,d,m);
    dPsi = np.zeros((d,m),float)
    
    #for i = 1:m
    #    dPsi(:,:,i)    = X(:,:,i) * S * X(:,:,i)';
    #end
    for i in range(m):
        dPsi[:,:,i] = np.diag(np.dot(np.dot(X[:,:,i], S), X[:,:,i].conj().transpose()))
    
    #dPsi = dPsi / batchsize;
    dPsi = dPsi / batchsize
    return dPsi

def cuttingPlaneFull(k=None, X=None, W=None, Ypos=None, Yneg=None, batchSize=None, SAMPLES=None, ClassScores=None):
#% [dPsi, M, SO_time] = cuttingPlaneFull(k, X, W, Yp, Yn, batchSize, SAMPLES, ClassScores)
#%
#%   k           = k parameter for the SO
#%   X           = d*n data matrix
#%   W           = d*d PSD metric
#%   Yp          = cell-array of relevant results for each point
#%   Yn          = cell-array of irrelevant results for each point
#%   batchSize   = number of points to use in the constraint batch
#%   SAMPLES     = indices of valid points to include in the batch
#%   ClassScores = structure for synthetic constraints
#%
#%   dPsi        = dPsi vector for this batch
#%   M           = mean loss on this batch
#%   SO_time     = time spent in separation oracle

    #global SO PSI DISTANCE CPGRADIENT; # define these as global                 !!!

    #[d,n,m] = size(X);
    (d,n,m) = X.shape()
    #D       = DISTANCE(W, X);
    D       = DISTANCE(W, X) # global defined function from main file

    #M       = 0;
    #S       = zeros(n);
    #dIndex  = sub2ind([n n], 1:n, 1:n);
    M       = 0
    S       = np.zeros((n,n))
    dIndex  = np.ravel_multi_index((range(n),range(n)),(n,n),order='C')

    SO_time = 0;

    #if isempty(ClassScores)
    #    TS  = zeros(batchSize, n);
    #    parfor i = 1:batchSize
    #        if i <= length(SAMPLES)
    #            j = SAMPLES(i);
    #
    if ClassScores.size == 0:
        TS  = np.zeros((batchSize, n));
        for i in range(batchSize):                                              # parallel better, how?
            if i <= len(SAMPLES):
                j = SAMPLES(i)
    #            if isempty(Ypos{j})
    #                continue;
    #            end
    #            if isempty(Yneg)
    #                % Construct a negative set 
    #                Ynegative = setdiff((1:n)', [j ; Ypos{j}]);
    #            else
    #                Ynegative = Yneg{j};
    #            end
                if Ypos[j].size == 0:                                           #Cell Array Ypos/neg indexing?
                    continue;
                if Yneg.size == 0:
                    # Construct a negative set 
                    Ynegative = np.setdiff1d(np.arange(1,n+1).conj().transpose(), [j,Ypos[j]])     # check indicies; Cell Array Ypos/neg indexing?
                else :
                    Ynegative = Yneg[j];                                        #Cell Array Ypos/neg indexing?
    #            SO_start        = tic();
    #                [yi, li]    =   SO(j, D, Ypos{j}, Ynegative, k);
    #            SO_time         = SO_time + toc(SO_start);
    #
    #            M               = M + li /batchSize;
    #            TS(i,:)         = PSI(j, yi', n, Ypos{j}, Ynegative);
    #        end
    #    end
    #    %Reconstruct the S matrix from TS
    #    S(SAMPLES,:)    = TS;
    #    S(:,SAMPLES)    = S(:,SAMPLES) + TS';
    #    S(dIndex)       = S(dIndex) - sum(TS, 1);
                SO_start        = time.time()
                [yi, li]        =   SO(j, D, Ypos[j], Ynegative, k)             # check indicies, global function SO, Cell Array Ypos/neg indexing?
                SO_time         = SO_time + time.time() - SO_start

                M               = M + li /batchSize
                TS[i,:]         = PSI(j, yi.conj().transpose(), n, Ypos[j], Ynegative) # global function PSI; Cell Array Ypos/neg indexing?
            
        # Reconstruct the S matrix from TS
        S[SAMPLES,:]    = TS
        S[:,SAMPLES]    = S[:,SAMPLES] + TS.conj().transpose()
        S[dIndex]       = S[dIndex] - sum(TS, 1)                                #check correct
# =============================================================================
# 
# =============================================================================
    #else:
    #
    #    % Do it class-wise for efficiency
    #    batchSize = 0;
    #    for j = 1:length(ClassScores.classes)
    #        c       = ClassScores.classes(j);
    #        points  = find(ClassScores.Y == c);
    #
    #        Yneg    = find(ClassScores.Yneg{j});
    #        yp      = ClassScores.Ypos{j};
    #        
    #        if length(points) <= 1
    #            continue;
    #        end
    #
    #        batchSize = batchSize + length(points);
    #        TS      = zeros(length(points), n);
    else :
        # Do it class-wise for efficiency
        batchSize = 0
        for j in range(len(ClassScores.classes)):
            c       = ClassScores.classes(j)
            points  = (ClassScores.Y == c).ravel().nonzero()                    #replaced find

            Yneg    = ClassScores.Yneg[j].ravel().nonzero()                     #replaced find, Cell Array Ypos/neg indexing?
            yp      = ClassScores.Ypos[j]
            
            if length(points) <= 1:
                continue

            batchSize = batchSize + len(points)
            TS        = zeros(len(points), n)                                   #fix here
    #        parfor x = 1:length(points)
    #            i           = points(x);
    #            yl          = yp;
    #            yl(i)       = 0;
    #            Ypos        = find(yl);
    #            SO_start    = tic();
    #                [yi, li]    = SO(i, D, Ypos, Yneg, k);
    #            SO_time     = SO_time + toc(SO_start);
    #
    #            M           = M + li;
    #            TS(x,:)     = PSI(i, yi', n, Ypos, Yneg);
    #        end
            for x in range(len(points)):                                        #better parallel, also went from 1 to length(points)
                i           = points[x]
                yl          = yp
                yl[i]       = 0
                Ypos        = yl.ravel().nonzero()                              #replaced find
                SO_start    = time.time()
                [yi, li]    = SO(i, D, Ypos, Yneg, k)                           #global function SO
                SO_time     = SO_time + time.time() - SO_start

                M           = M + li
                TS[x,:]     = PSI(i, yi.transpose(), n, Ypos, Yneg)             #global function PSI
    #
    #        S(points,:) = S(points,:) + TS;
    #        S(:,points) = S(:,points) + TS';
    #        S(dIndex)   = S(dIndex) - sum(TS, 1);
    #    end
    #    M   = M / batchSize;
    #end
    #dPsi    = CPGRADIENT(X, S, batchSize);
    #
    #return [dPsi, M, SO_time]
            S[points,:] = S[points,:] + TS
            S[:,points] = S[:,points] + TS.conj().transpose()                   #check indicies
            S[dIndex]   = S[dIndex] - sum(TS, 1)
        M   = M / batchSize

    dPsi    = CPGRADIENT(X, S, batchSize);                                      #global defined function CPGRADIENT
    return [dPsi, M, SO_time]


def cuttingPlaneRandom(k=None, X=None, W=None, Ypos=None, Yneg=None, batchSize=None, SAMPLES=None, ClassScores=None):
#function [dPsi, M, SO_time] = cuttingPlaneRandom(k, X, W, Ypos, Yneg, batchSize, SAMPLES, ClassScores)

#% [dPsi, M, SO_time] = cuttingPlaneRandom(k, X, W, Yp, Yn, batchSize, SAMPLES, ClassScores)
#%
#%   k           = k parameter for the SO
#%   X           = d*n data matrix
#%   W           = d*d PSD metric
#%   Yp          = cell-array of relevant results for each point
#%   Yn          = cell-array of irrelevant results for each point
#%   batchSize   = number of points to use in the constraint batch
#%   SAMPLES     = indices of valid points to include in the batch
#%   ClassScores = structure for synthetic constraints
#%
#%   dPsi        = dPsi vector for this batch
#%   M           = mean loss on this batch
#%   SO_time     = time spent in separation oracle
#
#    global SO PSI SETDISTANCE CPGRADIENT;                                      #!!!
    
    #    [d,n]   = size(X);
    #
    #    if length(SAMPLES) == n
    #        % All samples are fair game (full data)
    #        Batch   = randperm(n);
    #        Batch   = Batch(1:batchSize);
    #        D       = SETDISTANCE(X, W, Batch);
    [d,n]   = X.shape;

    if len(SAMPLES) == n:
        #% All samples are fair game (full data)
        Batch   = np.random.permutation(range(n))
        Batch   = Batch[0:batchSize]                                            #check indicies
        D       = SETDISTANCE(X, W, Batch)                                      #global function SETDISTANCE
    #    else
    #        Batch   = randperm(length(SAMPLES));
    #        Batch   = SAMPLES(Batch(1:batchSize));
    #
    #        Ito     = sparse(n,1);
    #        
    #        if isempty(ClassScores)
    #            for i = Batch
    #                Ito(Ypos{i}) = 1;
    #                Ito(Yneg{i}) = 1;
    #            end
    #            D       = SETDISTANCE(X, W, Batch, find(Ito));
    #        else
    #            D       = SETDISTANCE(X, W, Batch, 1:n);
    #        end
    #    end
    else :
        Batch   = np.random.permutation(range(length(SAMPLES)))
        Batch   = SAMPLES[Batch[1:batchSize]];

        Ito     = sparse(n,1);                                                  #TODO: replace sparse with SciPy csr_matrix
        
        if ClassScores.size == 0:
            for i in Batch:
                Ito(Ypos[i]) = 1                                                #Cell Array Ypos/neg indexing?
                Ito(Yneg[i]) = 1                                                #Cell Array Ypos/neg indexing?
            D       = SETDISTANCE(X, W, Batch, find(Ito))                       #global function SETDISTANCE
        else:
            D       = SETDISTANCE(X, W, Batch, range(n))                        #check indicies, probably use arange instead of range, global function SETDISTANCE

    #    M       = 0;
    #    S       = zeros(n);
    #    dIndex  = sub2ind([n n], 1:n, 1:n);
    #
    #    SO_time = 0;
    M       = 0
    S       = np.zeros((n, n))
    dIndex  = np.ravel_multi_index((range(n),range(n)),(n,n),order='C')         #replaced sub2ind

    SO_time = 0;
    
    #    if isempty(ClassScores)
    #        TS = zeros(batchSize, n);
    #        parfor j = 1:batchSize
    #            i = Batch(j);
    #            if isempty(Yneg)
    #                Ynegative   = setdiff((1:n)', [i ; Ypos{i}]);
    #            else
    #                Ynegative   = Yneg{i};
    #            end
    #            SO_start        = tic();
    #                [yi, li]    =   SO(i, D, Ypos{i}, Ynegative, k);
    #            SO_time         = SO_time + toc(SO_start);
    #    
    #            M               = M + li /batchSize;
    #            TS(j,:)         = PSI(i, yi', n, Ypos{i}, Ynegative);
    #        end
    #        S(Batch,:)      = TS;
    #        S(:,Batch)      = S(:,Batch)    + TS';
    #        S(dIndex)       = S(dIndex)     - sum(TS, 1);
    
    if ClassScores.size ==0:
        TS = zeros(batchSize, n)
        for j in range(batchSize):                                              #check indicies, better parallel
            i = Batch[j];
            if Yneg.size==0:
                Ynegative = np.setdiff1d(np.arange(1,n+1).conj().transpose(), [i,Ypos[i]])     # check indicies; Cell Array Ypos/neg indexing?
                                     
            else:
                Ynegative   = Yneg[i]                                           #Cell Array Ypos/neg indexing?
            SO_start        = time.time()
            [yi, li]        =   SO(i, D, Ypos[i], Ynegative, k)                 #Cell Array Ypos/neg indexing?, global function SO
            SO_time         = SO_time + time.time()-SO_start
    
            M               = M + li /batchSize;
            TS[j,:]         = PSI(i, yi.conj().transpose(), n, Ypos[i], Ynegative) # Cell Array Ypos/neg indexing?, global function PSI
        end
        S[Batch,:]      = TS
        S[:,Batch]      = S[:,Batch]    + TS.conj().tranpose()                  #check indicies here
        S[dIndex]       = S[dIndex]     - sum(TS, 1)
    
    #    else
    #        for j = 1:length(ClassScores.classes)
    #            c       = ClassScores.classes(j);
    #            points  = find(ClassScores.Y(Batch) == c);
    #            if ~any(points)
    #                continue;
    #            end
    #
    #            Yneg    = find(ClassScores.Yneg{j});
    #            yp      = ClassScores.Ypos{j};
    #
    #            TS      = zeros(length(points), n);
    
    else :
        for j in range(len(ClassScores.classes)):
            c       = ClassScores.classes[j]
            points  = (ClassScores.Y(Batch) == c).ravel().nonzero()             #find replaced
            if ~any(points):                                                    #python equivalent? - any nonzero elements?
                continue;

            Yneg    = (ClassScores.Yneg[j]).ravel().nonzero()                   #Cell Array Ypos/neg indexing?, find replaced
            yp      = ClassScores.Ypos[j]                                       #Cell Array Ypos/neg indexing?

            TS      = zeros(len(points), n)
    
    #            parfor x = 1:length(points)
    #                i               = Batch(points(x));
    #                yl              = yp;
    #                yl(i)           = 0;
    #                Ypos            = find(yl);
    #                SO_start        = tic();
    #                    [yi, li]    =   SO(i, D, Ypos, Yneg, k);
    #                SO_time         = SO_time + toc(SO_start);
    #    
    #                M               = M + li /batchSize;
    #                TS(x,:)         = PSI(i, yi', n, Ypos, Yneg);
    #            end
    
            for x in range(len(points)):                                        #indicies? -> better parallel
                i               = Batch[points[x]];
                yl              = yp
                yl[i]           = 0
                Ypos            = yl.ravel().nonzero()                          #find replaced
                SO_start        = time.time()
                [yi, li]    =   SO(i, D, Ypos, Yneg, k)                         #global function SO
                SO_time         = SO_time + time.time()-SO_start
    
                M               = M + li /batchSize
                TS[x,:]         = PSI(i, yi.conj().transpose(), n, Ypos, Yneg)  #Cell Array Ypos/neg indexing?
    
    #            S(Batch(points),:)  = S(Batch(points),:) + TS;
    #            S(:,Batch(points))  = S(:,Batch(points)) + TS';
    #            S(dIndex)           = S(dIndex) - sum(TS, 1);
    #        end
    #    end
    #
    #    dPsi    = CPGRADIENT(X, S, batchSize);
#end


            S[Batch[points],:]  = S[Batch[points],:] + TS
            S[:,Batch[points]]  = S[:,Batch[points]] + TS.conj().transpose()
            S[dIndex]           = S[dIndex] - sum(TS, 1)

    dPsi    = CPGRADIENT(X, S, batchSize)                                   #global function CPGRADIENT
    return [dPsi, M, SO_time]