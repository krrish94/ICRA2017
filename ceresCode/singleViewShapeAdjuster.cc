/*
 * Shape adjustmer using keypoint likelihoods from a single image
 * Authors: Junaid Ahmed Ansari & Sarthak Sharma
 * Email: junaid.ansari@research.iiit.ac.in, sarthak.sharma@research.iiit.ac.in
 *
 * This code is an adaptation of Krishna Murthy's (krrish94@gmail.com)
 * original 14 keypoint point Pose Adjuster
 */

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>

#include <ceres/loss_function.h>
#include <ceres/iteration_callback.h>
#include <ceres/rotation.h>

#include "problemStructs.hpp"
#include "costFunctions.hpp"


int main(int argc, char** argv){

	google::InitGoogleLogging(argv[0]);
	
	// Input and output file names	
	
	// IMPORTANT: These are relative paths for the files and should be provided based on the directory from where the
	// executable is being called and not based on where the executable is situated.
	const char* shapeBAInput  = "cache/shapeAdjusterInput.txt"; 
	const char* shapeBAOutput = "cache/shapeAdjusterOutput.txt";       
	const char* lambdasOuput  = "cache/lambdasAfterShape.txt";       

	// Read input File  	

	SingleViewShapeAdjustmentProblem singleViewShapeProblem;
	if(!singleViewShapeProblem.loadFile(shapeBAInput)){
		std::cerr << "ERROR: Unable to open file " << shapeBAInput << std::endl;
		return 1;
	}

	// Get the required parameters and initiallization for single view shape adjustment
	
	
	const int numPts           = singleViewShapeProblem.getNumPts();			// Num of observations and number of points	
	double *observations       = singleViewShapeProblem.observations();			// 2D predicted keypoints
	double *observationWeights = singleViewShapeProblem.observationWeights();	// Corresponding weights
	double *K                  = singleViewShapeProblem.getK();					// Camera matrix	
	double *X_bar              = singleViewShapeProblem.getX_bar();				// Mean wireframe	
	double *carCenter          = singleViewShapeProblem.getCarCenter();			// Get the center of the car	
	const double carLength     = singleViewShapeProblem.getCarLength();			// Get the length, width, and height of the car
	const double carWidth      = singleViewShapeProblem.getCarWidth();
	const double carHeight     = singleViewShapeProblem.getCarHeight();	
	const int numVec           = singleViewShapeProblem.getNumVec();			// Get the num of eigen vectors used.	
	double *V                  = singleViewShapeProblem.getV();					// Get the top numvec_ eigenvectors of the wireframe	
	double *lambdas            = singleViewShapeProblem.getLambdas();			// Get the weights of the linear combination			
	double *rot                = singleViewShapeProblem.getRot();				// Get the rotation and translation estimates (after PnP)
	double *trans              = singleViewShapeProblem.getTrans();	
	

	double rotAngleAxis[3] = {0, 0.001, 0};										// Convert the rotation estimate to an axis-angle representation
	ceres::RotationMatrixToAngleAxis(rot, rotAngleAxis);

	// Construct the Optimization Problem
	
	ceres::Solver::Options options;
	//options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.preconditioner_type = ceres::JACOBI;
	options.minimizer_progress_to_stdout = false;
	//options.max_num_iterations = 200;

	// Declare a Ceres problem instance to hold cost functions
	ceres::Problem problem;

	// For each observation, add a standard PnP error (reprojection error) residual block
	for(int i = 0; i < numPts; ++i){
		// Create a vector of eigenvalues for the current keypoint
		double *curEigVec = new double[3*numVec];
		// std::cout << "curEigVec: ";
		for(int j = 0; j < numVec; ++j){
			curEigVec[3*j+0] = V[3*numPts*j + 3*i + 0];
			curEigVec[3*j+1] = V[3*numPts*j + 3*i + 1];
			curEigVec[3*j+2] = V[3*numPts*j + 3*i + 2];
			// std::cout << V[3*numObs*j + 3*i + 0] << " " << V[3*numObs*j + 3*i + 1] << " " << \
			// 	V[3*numObs*j + 3*i + 2] << std::endl;
		}

		// Create a cost function for the lambda reprojection error term
		// i.e., std reprojection error, but instead of 3D points and R,t, we solve for lambdas (shape params)
		
		ceres::CostFunction *lambdaError = new ceres::AutoDiffCostFunction<LambdaReprojectionError, 2, 3, 42 >(
			new LambdaReprojectionError(X_bar+3*i, observations+2*i, numVec, curEigVec, K, observationWeights[i], trans));

		
		// Add a residual block to the problem
		problem.AddResidualBlock(lambdaError, new ceres::HuberLoss(0.5), rotAngleAxis, lambdas);

		// Add a regularizer (to prevent lambdas from growing too large)
		ceres::CostFunction *lambdaRegularizer = new ceres::AutoDiffCostFunction<LambdaRegularizer, 3, 42>(
			new LambdaRegularizer(numVec,curEigVec));
		// Add a residual block to the problem
		problem.AddResidualBlock(lambdaRegularizer, new ceres::HuberLoss(0.001), lambdas);

		// // Create a cost function to regularize 3D keypoint locations (alignment error)
		// ceres::CostFunction *alignmentError = new ceres::AutoDiffCostFunction<LambdaAlignmentError, 3, 5>(
		// 	new LambdaAlignmentError(X_bar+3*i, observations+2*i, curEigVec, K, observationWeights[i], X_bar_initial+3*i));

	}
	// We don't want to optimize over the rotation here. We want it to remain the same as it was after PnP.
	problem.SetParameterBlockConstant(rotAngleAxis);


	ceres::Solver::Summary summary;
  	ceres::Solve(options, &problem, &summary);
  	//std::cout << summary.FullReport()<<std::endl;

  	// Open the output file (to write the result)
	std::ofstream outFile;
	outFile.open(shapeBAOutput);

  	// Compute the resultant 3D wireframe
	for(int i = 0; i < numPts; ++i){
		double temp[3];
		temp[0] = X_bar[3*i];
		temp[1] = X_bar[3*i+1];
		temp[2] = X_bar[3*i+2];

		for(int j = 0; j < numVec; ++j){
			temp[0] += lambdas[j]*V[3*numPts*j + 3*i + 0];
			temp[1] += lambdas[j]*V[3*numPts*j + 3*i + 1];
			temp[2] += lambdas[j]*V[3*numPts*j + 3*i + 2];
			// std::cout << V[3*numObs*j + 3*i + 0] << " " << V[3*numObs*j + 3*i + 1] << " " << \
			// 	V[3*numObs*j + 3*i + 2] << std::endl;
		}

		ceres::AngleAxisRotatePoint(rotAngleAxis, temp, temp);
		temp[0] += trans[0];
		temp[1] += trans[1];
		temp[2] += trans[2];

		// Write the output to file
		outFile << temp[0] << " " << temp[1] << " " << temp[2] << std::endl;
	}

	outFile.close();
	
	// Write the lambdas
	
	outFile.open(lambdasOuput);

	for(int i=0;i<numVec;i++){
		outFile << lambdas[i] << " ";
	}

	outFile.close();
	
   	return 0;

}
