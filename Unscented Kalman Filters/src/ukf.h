#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"

class UKF
{
public:
    /**
     * Constructor
     */
    UKF();

    /**
     * Destructor
     */
    virtual ~UKF();

    /**
     * ProcessMeasurement
     * @param meas_package The latest measurement data of either radar or laser
     */
    void ProcessMeasurement(MeasurementPackage meas_package);

    /**
     * Prediction Predicts sigma points, the state, and the state covariance
     * matrix
     * @param delta_t Time between k and k+1 in s
     */
    void Prediction(double delta_t);

    // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    Eigen::VectorXd x_;

private:

    void Initialise();
    void InitialiseSensorNoise();
    bool InitialiseStateFromMeasurement(const MeasurementPackage& meas_package);

    void CreateAugmentedSigmaPoints();
    void PredictSigmaPoints(double delta_t);
    void PredictMeanAndCovariance();

    /**
    * Updates the state and the state covariance matrix using a laser measurement
    * @param meas_package The measurement at k+1
    */
    void UpdateLidar(MeasurementPackage meas_package);
    void PredictLidarMeasurement(Eigen::MatrixXd& Zsig, Eigen::VectorXd& z_pred, Eigen::MatrixXd& S);
    void UpdateLidarFromMeasurement(const MeasurementPackage& meas_package, const Eigen::MatrixXd& Zsig, const Eigen::VectorXd& z_pred, const Eigen::MatrixXd& S);

    /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
    void UpdateRadar(MeasurementPackage meas_package);
    void PredictRadarMeasurement(Eigen::MatrixXd& Zsig, Eigen::VectorXd& z_pred, Eigen::MatrixXd& S);
    void UpdateRadarFromMeasurement(const MeasurementPackage& meas_package, const Eigen::MatrixXd& Zsig, const Eigen::VectorXd& z_pred, const Eigen::MatrixXd& S);

private:

    // initially set to false, set to true in first call of ProcessMeasurement
    bool is_initialized_;

    // if this is false, laser measurements will be ignored (except for init)
    bool use_laser_;

    // if this is false, radar measurements will be ignored (except for init)
    bool use_radar_;

    // state covariance matrix
    Eigen::MatrixXd P_;

    // predicted sigma points matrix
    Eigen::MatrixXd Xsig_pred_;

    // time when the state is true, in us
    long long time_us_;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    double std_a_;

    // Process noise standard deviation yaw acceleration in rad/s^2
    double std_yawdd_;

    // Laser measurement noise standard deviation position1 in m
    double std_laspx_;

    // Laser measurement noise standard deviation position2 in m
    double std_laspy_;

    // Radar measurement noise standard deviation radius in m
    double std_radr_;

    // Radar measurement noise standard deviation angle in rad
    double std_radphi_;

    // Radar measurement noise standard deviation radius change in m/s
    double std_radrd_;

    // Weights of sigma points
    Eigen::VectorXd weights_;

    // State dimension
    int n_x_;

    // Augmented state dimension
    int n_aug_;

    // Sigma point spreading parameter
    double lambda_;

    // Augmented mean vector
    Eigen::VectorXd x_aug_;

    // Augmented state covariance
    Eigen::MatrixXd P_aug_;

    // Augmented sigma point matrix
    Eigen::MatrixXd Xsig_aug_;

    // NIS for radar
    double NIS_radar_;

    // NIS for laser
    double NIS_laser_;
};

#endif  // UKF_H