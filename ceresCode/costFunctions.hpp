/* CERES cost functions for car pose and shape adjustment
 * 
 * Authors: Junaid Ahmed Ansari & Sarthak Sharma
 * Email: junaid.ansari@research.iiit.ac.in, sarthak.sharma@research.iiit.ac.in
 *
 * This code is an adaptation of Krishna Murthy's (krrish94@gmail.com)
 * original 14 keypoint point Pose Adjuster
 */
#include <ceres/ceres.h>
#include <ceres/rotation.h>
# include <math.h>

// Cost function to optimize over R,t, given the 3D and 2D keypoints, as in PnP
// X: mean shape keypoints in 3D, x: 2D observations.
struct PnPError{

	// Constructor
	PnPError(double *X, double *x, int numVec, double *v, double *K, double w, double *l) \
		: X_(X), x_(x), numVec_(numVec), v_(v), K_(K), w_(w), l_(l) {}

	// Operator method. Evaluates the cost function and computes Jacobians.
	template <typename T>
	bool operator() (const T* const rot, const T* trans, T* residuals) const {

		// Temporary variable to hold the 3D keypoint
		T P_[3];

		// Initialize the 3D point
		P_[0] = T(X_[0]);P_[1] = T(X_[1]); P_[2] = T(X_[2]);
		for(int i=0;i<numVec_;i++){
			P_[0] = P_[0] + T(l_[i ])*T(v_[3*i+0]);
			P_[1] = P_[1] + T(l_[i ])*T(v_[3*i+1]);
			P_[2] = P_[2] + T(l_[i ])*T(v_[3*i+2]);

		}

		// Rotate the point (and store the result in the same variable)
		// Order of arguments passed: (axis-angle rotation vector (size 3), point (size 3), array where result is to be stored (size 3))
		ceres::AngleAxisRotatePoint(rot, P_, P_);
		// Add the translation
		P_[0] = T(P_[0]) + trans[0];
		P_[1] = T(P_[1]) + trans[1];
		P_[2] = T(P_[2]) + trans[2];

		// Project the obtained 3D point down to the image, using the intrinsics (K)
		T p_[3];
		p_[0] = T(K_[0])*P_[0] + T(K_[1])*P_[1] + T(K_[2])*P_[2];
		p_[1] = T(K_[3])*P_[0] + T(K_[4])*P_[1] + T(K_[5])*P_[2];
		p_[2] = T(K_[6])*P_[0] + T(K_[7])*P_[1] + T(K_[8])*P_[2];

		T px_ = p_[0] / p_[2];
		T py_ = p_[1] / p_[2];

		// Compute the residuals (this one works well)
		residuals[0] = T(1)*sqrt(T(w_))*(px_ - T(x_[0]));
		residuals[1] = T(1)*sqrt(T(w_))*(py_ - T(x_[1]));
		//std::cout<<"reproj resid: " << residuals[0]+residuals[1] << " \n";
		return true;
	}

	// 3D points
	double *X_;
	// 2D observations (keypoints)
	double *x_;
	// Number of vectors
	int numVec_;
	// Top 5 Eigenvectors of the 'current' 3D point
	double *v_;
	// Intrinsic camera matrix
	double *K_;
	// Weight for the current observation
	double w_;
	// Weights for each shape basis vector (lambdas)
	double *l_;

};





// SHAPE ADJUSTMENT
// Cost function to store reprojection error resulting from the values of lambdas
struct LambdaReprojectionError{

	// Constructor
	LambdaReprojectionError(double *X, double *x, int numVec, double *v, double *K, double w, double *trans) \
		: X_(X), x_(x), numVec_(numVec), v_(v), K_(K), w_(w), trans_(trans) {}

	// Operator method. Evaluates the cost function and computes Jacobians.
	template <typename T>
	bool operator() (const T* const rot_, const T* const l, T* residuals) const {


		// 3D wireframe (before applying rotation and translation)
		T P_[3];
		P_[0] = T(X_[0]);P_[1] = T(X_[1]); P_[2] = T(X_[2]);
		
		// Initialize the 3D point
		for(int i=0;i<numVec_;i++){
			P_[0] = P_[0] + T(l[i])*T(v_[3*i+0]);
			P_[1] = P_[1] + T(l[i])*T(v_[3*i+1]);
			P_[2] = P_[2] + T(l[i])*T(v_[3*i+2]);			
		}

		// Apply the rotation and translation
		ceres::AngleAxisRotatePoint(rot_, P_, P_);
		P_[0] += T(trans_[0]);
		P_[1] += T(trans_[1]);
		P_[2] += T(trans_[2]);

		// Project the obtained 3D point down to the image, using the intrinsics (K)
		T p_[3];
		p_[0] = T(K_[0])*P_[0] + T(K_[1])*P_[1] + T(K_[2])*P_[2];
		p_[1] = T(K_[3])*P_[0] + T(K_[4])*P_[1] + T(K_[5])*P_[2];
		p_[2] = T(K_[6])*P_[0] + T(K_[7])*P_[1] + T(K_[8])*P_[2];
		
		T px_ = p_[0] / p_[2];
		T py_ = p_[1] / p_[2];

		// Compute the residuals (this one works well)
		residuals[0] = sqrt(T(w_))*(px_ - T(x_[0]));
		residuals[1] = sqrt(T(w_))*(py_ - T(x_[1]));

		return true;
	}

	// 3D points
	double *X_;
	// 2D observations (keypoints)
	double *x_;
	// Read the number of eigen vectrs
	int numVec_;
	// Top 5 Eigenvectors of the 'current' 3D point
	double *v_;
	// Intrinsic camera matrix
	double *K_;
	// Weight for the current observation
	double w_;
	// Translation estimate (after PnP)
	double *trans_;

};



/*********************************************    REGULARIZERS   ****************************************************/

// Cost function to regularize the translation estimate, to prevent a huge drift from the initial estimate
struct TranslationRegularizer{

	// Constructor
	TranslationRegularizer(double *trans_init) : trans_init_(trans_init) {}

	// Operator method. Evaluates the cost function and computes the Jacobians.
	template <typename T>
	bool operator() (const T* trans, T* residuals) const{

		residuals[0] = T(20)*(trans[0]);
		residuals[1] = T(20)*(trans[1]);
		residuals[2] = T(100)*(trans[2]);

		return true;
	}

	// Initial translation estimate
	double *trans_init_;

};


// Cost function to regularize the rotation estimate, to align the axis of rotation with the axis that we desire.
struct RotationRegularizer{

	// Constructor
	RotationRegularizer() {}

	// Operator method. Evaluates the cost function and computes the Jacobians.
	template <typename T>
	bool operator() (const T* rot, T* residuals) const{

		residuals[0] = T(1000.0)*rot[0];
		residuals[1] = T(0.0);
		residuals[2] = T(1000.0)*rot[2];

		return true;
	}

};

// Cost function to prevent lambdas from deforming the shape strongly
struct LambdaRegularizer{

	// Constructor
	LambdaRegularizer(int numVec, double *v) : numVec_(numVec), v_(v) {}

	// Operator method. Evaluates the cost function and computes the Jacobians.
	template <typename T>
	bool operator() (const T* l, T* residuals) const{

		residuals[0] = T(0.0); residuals[1] = T(0.0); residuals[2] = T(0.0);

		for(int i=0;i<numVec_;i++){
			residuals[0] += T(l[i])*T(v_[3*i+0]);
			residuals[1] += T(l[i])*T(v_[3*i+1]);
			residuals[2] += T(l[i])*T(v_[3*i+2]);

		}

		return true;
	}

	// Number of eigen vectors
	int numVec_;
	// Top numVec eigenvectors for the 'current' 3D point
	double *v_;

};

