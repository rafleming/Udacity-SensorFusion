#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;


UKF::UKF()
{
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    Initialise();
}

UKF::~UKF()
{
}

/**
 * Initializes Unscented Kalman filter
 */
void UKF::Initialise()
{
    // State dimension
    n_x_ = 5;

    // Augmented state dimension
    n_aug_ = 7;

    // initial state vector
    x_ = VectorXd(n_x_);

    // initial covariance matrix
    P_ = MatrixXd(n_x_, n_x_);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 1.0;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 1.0;

    // Initially set to false, set to true in first call of ProcessMeasurement
    is_initialized_ = false;

    // Define spreading parameter
    lambda_ = 3 - n_aug_;

    // Matrix to hold sigma points
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

    // Vector for weights
    weights_ = VectorXd(2 * n_aug_ + 1);

    // Start time
    time_us_ = 0;

    // create augmented mean vector
    x_aug_ = VectorXd(n_aug_);

    // create augmented state covariance
    P_aug_ = MatrixXd(n_aug_, n_aug_);

    // create sigma point matrix
    Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    // NIS for radar
    NIS_radar_ = 0.0;

    // NIS for laser
    NIS_laser_ = 0.0;

    InitialiseSensorNoise();
}

void UKF::InitialiseSensorNoise()
{
    /**
     * DO NOT MODIFY measurement noise values below.
     * These are provided by the sensor manufacturer.
     */

     // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    /**
     * End DO NOT MODIFY section for measurement noise values
     */
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
    /**
     * TODO: Complete this function! Make sure you switch between lidar and radar
     * measurements.
     */

     // use the first measurement to initialise state
    if (!is_initialized_)
    {
        is_initialized_ = InitialiseStateFromMeasurement(meas_package);
        time_us_ = meas_package.timestamp_;
    }
    else
    {
        const float delta_t = (meas_package.timestamp_ - time_us_) / 1e6;
        time_us_ = meas_package.timestamp_;

        Prediction(delta_t);

        if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER)
        {
            UpdateLidar(meas_package);
        }

        if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR)
        {
            UpdateRadar(meas_package);
        }
    }
}

bool UKF::InitialiseStateFromMeasurement(const MeasurementPackage& meas_package)
{
    // use the lidar measurement to initialise the state
    if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
        const double px = meas_package.raw_measurements_(0); // x-pos
        const double py = meas_package.raw_measurements_(1); // y-pos

        x_ << px, py, 0, 0, 0;

        // initialise covariance matrix with lidar measurement noise standard deviation
        P_ << std_laspx_ * std_laspx_, 0, 0, 0, 0,
            0, std_laspy_* std_laspy_, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;

        return true;
    }

    // use the radar measurement to initialise the state
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
        const double rho = meas_package.raw_measurements_(0); 	// range
        const double phi = meas_package.raw_measurements_(1);		// bearing
        const double rho_dot = meas_package.raw_measurements_(2);	// range change

        const double px = rho * sin(phi); // x-pos
        const double py = rho * cos(phi); // y-pos

        const double vx = rho_dot * cos(phi);  // velocity x
        const double vy = rho_dot * sin(phi);  // velocity y
        const double v = sqrt(vx * vx + vy * vy);

        x_ << px, py, v, 0, 0;

        // initialise covariance matrix with radar measurement noise standard deviation
        P_ << std_radr_ * std_radr_, 0, 0, 0, 0,
            0, std_radr_* std_radr_, 0, 0, 0,
            0, 0, std_radrd_* std_radrd_, 0, 0,
            0, 0, 0, std_radphi_* std_radphi_, 0,
            0, 0, 0, 0, 1;

        return true;
    }

    return false;
}

void UKF::Prediction(double delta_t)
{
    /**
     * TODO: Complete this function! Estimate the object's location.
     * Modify the state vector, x_. Predict sigma points, the state,
     * and the state covariance matrix.
     */

     // first create the augmented sigma points
    CreateAugmentedSigmaPoints();

    // next predict the sigma points
    PredictSigmaPoints(delta_t);

    // then predict mean and covariance
    PredictMeanAndCovariance();
}

void UKF::CreateAugmentedSigmaPoints()
{
    // set augmented mean state
    x_aug_.head(5) = x_; 	// original state for first 5 rows
    x_aug_(5) = 0;		// longitudinal acceleration noise
    x_aug_(6) = 0;		// yaw acceleration noise

    // set augmented covariance matrix
    P_aug_.fill(0.0);
    P_aug_.topLeftCorner(5, 5) = P_;

    P_aug_(5, 5) = std_a_ * std_a_;			// longitudinal acceleration noise
    P_aug_(6, 6) = std_yawdd_ * std_yawdd_;	// yaw acceleration noise

    // find the square root of P_aug
    MatrixXd L = P_aug_.llt().matrixL();

    // finally set the augmented sigma points
    Xsig_aug_.fill(0.0);
    Xsig_aug_.col(0) = x_aug_;

    for (int i = 0; i < n_aug_; i++)
    {
        Xsig_aug_.col(i + 1)
            = x_aug_ + sqrt(lambda_ + n_aug_) * L.col(i);

        Xsig_aug_.col(i + 1 + n_aug_)
            = x_aug_ - sqrt(lambda_ + n_aug_) * L.col(i);
    }
}

void UKF::PredictSigmaPoints(double delta_t)
{
    const int n_sigma_pts = 2 * n_aug_ + 1;

    for (int i = 0; i < n_sigma_pts; i++)
    {
        const double px = Xsig_aug_(0, i); // x pos
        const double py = Xsig_aug_(1, i); // y pos
        const double v = Xsig_aug_(2, i); // velocity
        const double yaw = Xsig_aug_(3, i); // yaw rate
        const double yawd = Xsig_aug_(4, i); // yaw angle
        const double nu_a = Xsig_aug_(5, i); // acc noise
        const double nu_yawdd = Xsig_aug_(6, i); // yaw rate noise

        // predicted x and y position
        double px_pred, py_pred;

        if (fabs(yawd) > std::numeric_limits<double>::min())
        {
            px_pred = px + (v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw)));
            py_pred = py + (v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t)));
        }
        else
        {
            px_pred = px + (v * delta_t * cos(yaw));
            py_pred = py + (v * delta_t * sin(yaw));
        }

        // add noise
        px_pred += 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_pred += 0.5 * nu_a * delta_t * delta_t * sin(yaw);

        // predicted velocity, yaw rate and angle
        double v_pred = v;
        double yaw_pred = yaw + yawd * delta_t;
        double yawd_pred = yawd;

        v_pred += nu_a * delta_t;

        yaw_pred += 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_pred += nu_yawdd * delta_t;

        // update the predicted sigma point
        Xsig_pred_(0, i) = px_pred;
        Xsig_pred_(1, i) = py_pred;
        Xsig_pred_(2, i) = v_pred;
        Xsig_pred_(3, i) = yaw_pred;
        Xsig_pred_(4, i) = yawd_pred;
    }
}

void UKF::PredictMeanAndCovariance()
{
    const int n_sigma_pts = 2 * n_aug_ + 1;

    // update the weights
    weights_(0) = lambda_ / (lambda_ + n_aug_);

    for (int i = 1; i < n_sigma_pts; i++)
    {
        weights_(i) = 0.5 / (lambda_ + n_aug_);
    }

    // update the state mean
    x_.fill(0.0);
    for (int i = 0; i < n_sigma_pts; i++)
    {
        x_ += weights_(i) * Xsig_pred_.col(i);
    }

    // update the state covariance
    P_.fill(0.0);
    for (int i = 0; i < n_sigma_pts; i++)
    {
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        // angle normalization
        while (x_diff(3) > M_PI)
            x_diff(3) -= 2. * M_PI;

        while (x_diff(3) < -M_PI)
            x_diff(3) += 2. * M_PI;

        P_ += weights_(i) * x_diff * x_diff.transpose();
    }
}

void UKF::UpdateLidar(MeasurementPackage meas_package)
{
    /**
     * TODO: Complete this function! Use lidar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the lidar NIS, if desired.
     */

     // First predict the lidar meaurement
    const int n_z = 2;
    const int n_sigma_pts = 2 * n_aug_ + 1;

    // sigma points matrix (measurement space)
    MatrixXd Zsig = MatrixXd(n_z, n_sigma_pts);

    // predicted lidar measurement mean
    VectorXd z_pred = VectorXd(n_z);

    // predicted lidar measurement covariance matrix
    MatrixXd S = MatrixXd(n_z, n_z);

    PredictLidarMeasurement(Zsig, z_pred, S);

    // Then update lidar state from the measurement
    UpdateLidarFromMeasurement(meas_package, Zsig, z_pred, S);
}

void UKF::PredictLidarMeasurement(MatrixXd& Zsig, VectorXd& z_pred, MatrixXd& S)
{
    const int n_z = 2;
    const int n_sigma_pts = 2 * n_aug_ + 1;

    // first transform the sigma points into measurement space
    for (int i = 0; i < n_sigma_pts; i++)
    {
        // measurement model
        Zsig(0, i) = Xsig_pred_(0, i);
        Zsig(1, i) = Xsig_pred_(1, i);
    }

    // predict mean 
    z_pred.fill(0.0);
    for (int i = 0; i < n_sigma_pts; i++)
    {
        z_pred += weights_(i) * Zsig.col(i);
    }

    // update covariance matrix
    S.fill(0.0);
    for (int i = 0; i < n_sigma_pts; i++)
    {
        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        S += weights_(i) * z_diff * z_diff.transpose();
    }

    // include measurement noise 
    MatrixXd R = MatrixXd(n_z, n_z);
    R << std_laspx_ * std_laspx_, 0,
        0, std_laspy_* std_laspy_;

    S += R;
}

void UKF::UpdateLidarFromMeasurement(const MeasurementPackage& meas_package, const MatrixXd& Zsig, const VectorXd& z_pred, const MatrixXd& S)
{
    const int n_z = 2;
    const int n_sigma_pts = 2 * n_aug_ + 1;

    // calculate the cross correlation matrix
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);

    for (int i = 0; i < n_sigma_pts; i++)
    {
        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        // angle normalization
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        Tc += weights_(i) * x_diff * z_diff.transpose();
    }

    MatrixXd K = Tc * S.inverse();

    // residual
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

    // update state and covariance matrix
    x_ += K * z_diff;
    P_ -= K * S * K.transpose();

    // update the lidar NIS
    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
     * TODO: Complete this function! Use radar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the radar NIS, if desired.
     */

     // First predict the radar meaurement
    const int n_z = 3;
    const int n_sigma_pts = 2 * n_aug_ + 1;

    // sigma points matrix (measurement space)
    MatrixXd Zsig = MatrixXd(n_z, n_sigma_pts);

    // predicted radar measurement mean
    VectorXd z_pred = VectorXd(n_z);

    // predicted radar measurement covariance matrix
    MatrixXd S = MatrixXd(n_z, n_z);

    PredictRadarMeasurement(Zsig, z_pred, S);

    // Then update radar state from the measurement
    UpdateRadarFromMeasurement(meas_package, Zsig, z_pred, S);
}

void UKF::PredictRadarMeasurement(MatrixXd& Zsig, VectorXd& z_pred, MatrixXd& S)
{
    const int n_z = 3;
    const int n_sigma_pts = 2 * n_aug_ + 1;

    // convert sigma points to measurement space
    Zsig.fill(0.0);
    for (int i = 0; i < n_sigma_pts; i++)
    {
        double px = Xsig_pred_(0, i);
        double py = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double vx = cos(yaw) * v;
        double vy = sin(yaw) * v;

        Zsig(0, i) = sqrt(px * px + py * py);                   // r
        Zsig(1, i) = atan2(py, px);							            // phi
        Zsig(2, i) = (px * vx + py * vy) / sqrt(px * px + py * py); // r_dot
    }

    // predict mean
    z_pred.fill(0.0);
    for (int i = 0; i < n_sigma_pts; i++)
    {
        z_pred += weights_(i) * Zsig.col(i);
    }

    // update covariance matrix
    S.fill(0.0);
    for (int i = 0; i < n_sigma_pts; i++)
    {
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // normalise
        while (z_diff(1) > M_PI)
            z_diff(1) -= 2. * M_PI;

        while (z_diff(1) < -M_PI)
            z_diff(1) += 2. * M_PI;

        S += weights_(i) * z_diff * z_diff.transpose();
    }

    // include measurement noise
    MatrixXd R = MatrixXd(n_z, n_z);

    R << std_radr_ * std_radr_, 0, 0,
        0, std_radphi_* std_radphi_, 0,
        0, 0, std_radrd_* std_radrd_;

    S = S + R;
}

void UKF::UpdateRadarFromMeasurement(const MeasurementPackage& meas_package, const MatrixXd& Zsig, const VectorXd& z_pred, const MatrixXd& S)
{
    const int n_z = 3;
    const int n_sigma_pts = 2 * n_aug_ + 1;

    // calculate the cross correlation matrix
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);

    for (int i = 0; i < n_sigma_pts; i++)
    {
        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // normalise 
        while (z_diff(1) > M_PI)
            z_diff(1) -= 2. * M_PI;

        while (z_diff(1) < -M_PI)
            z_diff(1) += 2. * M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        // normalise
        while (x_diff(3) > M_PI)
            x_diff(3) -= 2. * M_PI;

        while (x_diff(3) < -M_PI)
            x_diff(3) += 2. * M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    // kalman gain
    MatrixXd K = Tc * S.inverse();

    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

    // normalise
    while (z_diff(1) > M_PI)
        z_diff(1) -= 2. * M_PI;

    while (z_diff(1) < -M_PI)
        z_diff(1) += 2. * M_PI;

    // update state and covariance matrix
    x_ += K * z_diff;
    P_ -= K * S * K.transpose();

    // update the radar NIS
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}