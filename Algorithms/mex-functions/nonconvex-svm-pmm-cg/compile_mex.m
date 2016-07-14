%% Written by Yan Kaganovsky and Ikenna Odinaka 2016 %%

% This script is used to compile the mex functions 
% It sets the options to be able to use OpenMP parallelization
% -g stands for debug mode
% -I for specifying include folders
clc                  

% Release
if isunix
    try
        mex PMM_CG_nonconvex_svm.cpp -largeArrayDims CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"; % Unix
    catch
        mex PMM_CG_nonconvex_svm.cpp -largeArrayDims; % OpenMP not enabled
    end    
else
    try
        mex PMM_CG_nonconvex_svm.cpp -largeArrayDims COMPFLAGS="$COMPFLAGS /openmp"; % Windows
    catch
        mex PMM_CG_nonconvex_svm.cpp -largeArrayDims; % OpenMP not enabled
    end
end

% Debug
% if isunix
%     try
%         mex PMM_CG_nonconvex_svm.cpp -g -largeArrayDims CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"; % Linux
%     catch
%         mex PMM_CG_nonconvex_svm.cpp -g -largeArrayDims; % OpenMP not enabled
%     end    
% else
%     try
%         mex PMM_CG_nonconvex_svm.cpp -g -largeArrayDims COMPFLAGS="$COMPFLAGS /openmp"; % Windows
%     catch
%         mex PMM_CG_nonconvex_svm.cpp -g -largeArrayDims; % OpenMP not enabled
%     end
% end
   
clc        
