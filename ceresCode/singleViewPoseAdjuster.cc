/*
 * Single view car pose adjuster
 * Authors: Junaid Ahmed Ansari & Sarthak Sharma,
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
	const char *poseBAInput       = "cache/poseAdjusterInput.txt";									
	const char *poseBAOutput      = "cache/poseAdjusterOutput.txt";
	const char *shapeBAInput      = "cache/shapeAfterPose.txt";          
     
	// Read input Files
	
	SingleViewPoseAdjustmentProblem singleViewPoseProblem;
	if(!singleViewPoseProblem.loadFile(poseBAInput)){
				
		std::cerr << "ERROR: Unable to open file ..... " << poseBAInput << std::endl;
		return 1;
	}

	// Get the required parameters and initiallization for single view pose adjustment

	const int numPts           = singleViewPoseProblem.getNumPts();
	double *observations       = singleViewPoseProblem.observations();
	double *observationWeights = singleViewPoseProblem.observationWeights();
	double *K                  = singleViewPoseProblem.getK();
	double *X_bar              = singleViewPoseProblem.getX_bar();				// Mean car shape
	double *carCenter          = singleViewPoseProblem.getCarCenter();	
	const double carLength     = singleViewPoseProblem.getCarLength();              // Average car dimension
	const double carWidth      = singleViewPoseProblem.getCarWidth();
	const double carHeight     = singleViewPoseProblem.getCarHeight();
	const int numVec           = singleViewPoseProblem.getNumVec();				// Num of shape basis vectors
	double *V                  = singleViewPoseProblem.getV();					// Shape basis vectors
	double *lambdas            = singleViewPoseProblem.getLambdas();                // Eigen values

	// Set up the Ceres solver options and optimization prameters

	ceres::Solver::Options options;
	// options.linear_solver_type     = ceres::DENSE_SCHUR;
	// ITERATIVE_SCHUR > DENSE_SCHUR ~= SPARSE_SCHUR
	// options.linear_solver_type     = ceres::ITERATIVE_SCHUR;
	options.linear_solver_type        = ceres::DENSE_SCHUR;
	options.preconditioner_type       = ceres::JACOBI;
	
	// ITERATIVE_SCHUR + explicit schur complement = disaster
	// options.use_explicit_schur_complement = true;
	// options.use_inner_iterations             = true;
	options.max_num_iterations = 100;
	options.minimizer_progress_to_stdout     = false;	
	
	
	// Initialize the delta rotation and actual translation estimates of the mean car
	
	double trans[3] = {0.1, 0.1, 0.1};
	trans[0] = carCenter[0];
	trans[1] = carCenter[1];
	trans[2] = carCenter[2];
	
	double rotAngleAxis[3] = {0.01,0.01, 0.01};

	
	// Instance to hold all the cost functions for optimization	
	ceres::Problem problem;		
		
	// Construct the Optimization Problem for Car -------------------------------------------------------------------------------------		


	// For each observation, add a standard PnP error (reprojection error) residual block		
	for(int i = 0; i < numPts; ++i){
	
		// Create a vector of eigenvalues for the current keypoint		
		double *curEigVec = new double[numVec*3];
		// std::cout << "curEigVec: ";
		for(int j = 0; j < numVec; ++j){
			curEigVec[3*j+0] = V[3*numPts*j + 3*i + 0];
			curEigVec[3*j+1] = V[3*numPts*j + 3*i + 1];
			curEigVec[3*j+2] = V[3*numPts*j + 3*i + 2];			
		}

		// Create a cost function for the PnP error term
		ceres::CostFunction *pnpError = new ceres::AutoDiffCostFunction<PnPError, 2, 3, 3>( 
			new PnPError(X_bar+3*i, observations+2*i, numVec, curEigVec, K, observationWeights[i], lambdas));

		// Add a residual block to the problem		
		problem.AddResidualBlock(pnpError, new ceres::HuberLoss(30.8), rotAngleAxis,trans);// ceres::HuberLoss(0.8) worked for most cases
		
	}

	// Add a regularizer to the translation term (to prevent a huge drift from the initialization)
	ceres::CostFunction *translationRegularizer = new ceres::AutoDiffCostFunction<TranslationRegularizer, 3, 3>(
		new TranslationRegularizer(carCenter));
	problem.AddResidualBlock(translationRegularizer, new ceres::HuberLoss(0.1), trans);

	// Add a rotation regularizer, to ensure that the rotation is about the Y-axis
	ceres::CostFunction *rotationRegularizer = new ceres::AutoDiffCostFunction<RotationRegularizer, 3, 3>(
		new RotationRegularizer());
	problem.AddResidualBlock(rotationRegularizer, NULL, rotAngleAxis);

	
	// Solve the Optimization Problem ---------------------------------------------------------------------------------------------------


	// Solve the problem and print the results
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	// std::cout << summary.FullReport() << std::endl;
	
	// Write the outputs ----------------------------------------------------------------------------------------------------------------
	
	std::ofstream outFile;
	
	// Write the optimized pose of the car
	
	outFile.open(poseBAOutput);
	double rotMat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

	// Write the optimized rotation and translation to the output file
	ceres::AngleAxisToRotationMatrix(rotAngleAxis, rotMat);

	for(int i = 0; i < 9; ++i){
		// Write out an entry of the estimated rotation matrix to file (in column-major order)
		outFile << rotMat[i] << std::endl;
	}
	
	outFile << trans[0] << std::endl;
	outFile << trans[1] << std::endl;
	outFile << trans[2] << std::endl;
	
	outFile.close();
		
	// Write the transformed (after pose adjustment) 3D keypoints of mean car for shape adjustment
	
	outFile.open(shapeBAInput);

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


 	return 0;

}
