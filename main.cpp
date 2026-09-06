#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <deque>
#include <string>
#include <chrono>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <functional>
#include <algorithm>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <utility>

#ifdef _WIN32
  #include <direct.h>
  #define MAKE_DIR(d) _mkdir(d)
#else
  #include <sys/stat.h>
  #define MAKE_DIR(d) mkdir(d, 0755)
#endif

using HRC = std::chrono::high_resolution_clock;

struct DetectionResult {
    std::vector<int>                       markerIds;
    std::vector<std::vector<cv::Point2f>>  markerCorners;
    std::vector<std::vector<cv::Point2f>>  rejectedCandidates;
    double                                 detectionTimeMs;
    int                                    frameW, frameH;
};

class ArucoDetector {
public:
    ArucoDetector() {

        dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100);

        cv::aruco::DetectorParameters params;
        params.cornerRefinementMethod    = cv::aruco::CORNER_REFINE_SUBPIX;
        params.adaptiveThreshWinSizeMin  = 3;
        params.adaptiveThreshWinSizeMax  = 23;
        params.adaptiveThreshWinSizeStep = 10;
        params.minMarkerPerimeterRate    = 0.03f;
        params.errorCorrectionRate       = 0.10;

        detector_ = cv::aruco::ArucoDetector(dictionary_, params);
    }

    DetectionResult detect(const cv::Mat& frame) {
        if (frame.empty()) throw std::runtime_error("Empty frame");

        DetectionResult res;
        res.frameW = frame.cols;
        res.frameH = frame.rows;

        cv::Mat grey;
        if (frame.channels() == 3)
            cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
        else
            grey = frame;

        auto t0 = HRC::now();

        detector_.detectMarkers(grey,
                                 res.markerCorners,
                                 res.markerIds,
                                 res.rejectedCandidates);

        res.detectionTimeMs =
            std::chrono::duration<double, std::milli>(HRC::now()-t0).count();
        return res;
    }

    cv::Mat visualise(const cv::Mat& frame, const DetectionResult& res) const {
        cv::Mat out = frame.clone();
        if (!res.markerIds.empty())
            cv::aruco::drawDetectedMarkers(out, res.markerCorners,
                                           res.markerIds);
        if (!res.rejectedCandidates.empty())
            cv::aruco::drawDetectedMarkers(out, res.rejectedCandidates,
                                           cv::noArray(),
                                           cv::Scalar(180,0,180));
        char buf[80];
        std::snprintf(buf, sizeof(buf), "Markers: %d  |  Det: %.1f ms",
                      (int)res.markerIds.size(), res.detectionTimeMs);
        cv::putText(out, buf, {10,26},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,230,0}, 2);
        return out;
    }

private:
    cv::aruco::Dictionary    dictionary_;
    cv::aruco::ArucoDetector detector_;
};

struct CameraParams {
    cv::Mat  cameraMatrix;
    cv::Mat  distCoeffs;
    cv::Size imageSize;

    static std::optional<CameraParams> fromFile(const std::string& path) {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) return std::nullopt;
        CameraParams p;
        fs["camera_matrix"]           >> p.cameraMatrix;
        fs["distortion_coefficients"] >> p.distCoeffs;
        int w=0, h=0;
        fs["image_width"]  >> w;
        fs["image_height"] >> h;
        p.imageSize = {w, h};
        return (p.cameraMatrix.empty() || p.distCoeffs.empty())
               ? std::nullopt : std::optional<CameraParams>(p);
    }

    void save(const std::string& path) const {
        cv::FileStorage fs(path, cv::FileStorage::WRITE);
        fs << "camera_matrix"           << cameraMatrix
           << "distortion_coefficients" << distCoeffs
           << "image_width"             << imageSize.width
           << "image_height"            << imageSize.height;
    }

    static CameraParams defaultWebcam() {
        CameraParams p;
        p.imageSize    = {640, 480};
        p.cameraMatrix = (cv::Mat_<double>(3,3) <<
            600,   0, 320,
              0, 600, 240,
              0,   0,   1);
        p.distCoeffs = cv::Mat::zeros(1, 5, CV_64F);
        return p;
    }
};

struct MarkerPose {
    int       id;
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    cv::Mat   R;
    double    distance;
    double    roll, pitch, yaw;
};

class PoseEstimator {
public:
    PoseEstimator(const CameraParams& cam, double markerLength = 0.05)
        : cam_(cam), markerLength_(markerLength)
    {

        float h = static_cast<float>(markerLength_ / 2.0);

        objPoints_.push_back({-h,  h, 0});
        objPoints_.push_back({ h,  h, 0});
        objPoints_.push_back({ h, -h, 0});
        objPoints_.push_back({-h, -h, 0});
    }

    std::vector<MarkerPose> estimate(const DetectionResult& det) const {
        std::vector<MarkerPose> out;
        if (det.markerIds.empty()) return out;

        out.reserve(det.markerIds.size());
        for (size_t i = 0; i < det.markerIds.size(); ++i) {

            cv::Mat rvec, tvec;
            bool ok = cv::solvePnP(
                objPoints_,
                det.markerCorners[i],
                cam_.cameraMatrix,
                cam_.distCoeffs,
                rvec, tvec,
                false,
                cv::SOLVEPNP_IPPE_SQUARE
            );

            if (!ok || rvec.empty() || tvec.empty() ||
                !cv::checkRange(rvec) || !cv::checkRange(tvec)) {
                std::cerr << "WARNING: solvePnP failed for marker ID "
                          << det.markerIds[i] << " - skipping\n";
                continue;
            }

            MarkerPose mp;
            mp.id   = det.markerIds[i];
            mp.rvec = rvec;
            mp.tvec = tvec;
            cv::Rodrigues(rvec, mp.R);
            mp.distance = cv::norm(tvec);
            toEuler(mp.rvec, mp.roll, mp.pitch, mp.yaw);
            out.push_back(mp);
        }
        return out;
    }

    cv::Mat visualise(const cv::Mat& frame,
                      const std::vector<MarkerPose>& poses) const {
        cv::Mat out = frame.clone();
        for (const auto& mp : poses) {

            cv::drawFrameAxes(out,
                              cam_.cameraMatrix, cam_.distCoeffs,
                              mp.rvec, mp.tvec,
                              static_cast<float>(markerLength_ * 0.5f));

            std::vector<cv::Point3f> obj3 = {{0,0,0}};
            std::vector<cv::Point2f> img2;
            cv::projectPoints(obj3, mp.rvec, mp.tvec,
                              cam_.cameraMatrix, cam_.distCoeffs, img2);
            if (!img2.empty()) {
                cv::Point pt((int)img2[0].x, (int)img2[0].y);
                char buf[64];
                std::snprintf(buf, sizeof(buf),
                              "ID%d  %.2fm", mp.id, mp.distance);
                cv::putText(out, buf, pt+cv::Point(6,-6),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,255,255}, 2);
                std::snprintf(buf, sizeof(buf), "yaw %.1fdeg", mp.yaw);
                cv::putText(out, buf, pt+cv::Point(6,16),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, {255,200,0}, 1);
            }
        }
        return out;
    }

    const CameraParams& camera()       const { return cam_; }
    double              markerLength() const { return markerLength_; }

private:
    CameraParams              cam_;
    double                    markerLength_;
    std::vector<cv::Point3f>  objPoints_;

    static void toEuler(const cv::Vec3d& rvec,
                        double& roll, double& pitch, double& yaw) {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        pitch = std::atan2(-R.at<double>(2,0),
                            std::hypot(R.at<double>(2,1),
                                       R.at<double>(2,2)));
        roll  = std::atan2(R.at<double>(2,1), R.at<double>(2,2));
        yaw   = std::atan2(R.at<double>(1,0), R.at<double>(0,0));
        const double r2d = 180.0 / CV_PI;
        roll  *= r2d;
        pitch *= r2d;
        yaw   *= r2d;
    }
};

struct PredictionResult {
    int                      markerId;
    cv::Point3d              smoothedPos;
    cv::Point3d              velocity;
    std::vector<cv::Point3d> futurePath;
    double                   stepDtMs;
    double                   horizonMs;
    bool                     reliable;
};

struct KFState {
    cv::KalmanFilter        kf;
    bool                    initialised = false;
    double                  lastMs      = 0.0;
    std::deque<cv::Point3d> history;
    static constexpr size_t MAX_HIST = 60;
};

class TrajectoryPredictor {
public:
    explicit TrajectoryPredictor(int    steps  = 10,
                                 double pNoise = 1e-4,
                                 double mNoise = 1e-2)
        : steps_(steps), pNoise_(pNoise), mNoise_(mNoise) {}

    std::vector<PredictionResult> update(
        const std::vector<MarkerPose>& poses, double nowMs)
    {
        std::vector<PredictionResult> results;
        for (const auto& mp : poses) {
            auto& s = kfs_[mp.id];
            cv::Point3d meas(mp.tvec[0], mp.tvec[1], mp.tvec[2]);

            if (!s.initialised) { initKF(s, meas); s.lastMs = nowMs; }

            double dt = clampDt(s.lastMs, nowMs);
            s.lastMs  = nowMs;
            s.kf.transitionMatrix.at<double>(0,3) = dt;
            s.kf.transitionMatrix.at<double>(1,4) = dt;
            s.kf.transitionMatrix.at<double>(2,5) = dt;

            s.kf.predict();

            cv::Mat z = (cv::Mat_<double>(3,1) << meas.x, meas.y, meas.z);
            cv::Mat corrected = s.kf.correct(z);

            cv::Point3d smoothed(corrected.at<double>(0),
                                 corrected.at<double>(1),
                                 corrected.at<double>(2));
            cv::Point3d vel(corrected.at<double>(3),
                            corrected.at<double>(4),
                            corrected.at<double>(5));

            s.history.push_back(meas);
            if (s.history.size() > KFState::MAX_HIST) s.history.pop_front();

            cv::Mat future = corrected.clone();
            std::vector<cv::Point3d> path;
            path.reserve(steps_);
            for (int k = 0; k < steps_; ++k) {
                future = s.kf.transitionMatrix * future;
                path.emplace_back(future.at<double>(0),
                                  future.at<double>(1),
                                  future.at<double>(2));
            }

            PredictionResult pr;
            pr.markerId    = mp.id;
            pr.smoothedPos = smoothed;
            pr.velocity    = vel;
            pr.futurePath  = std::move(path);
            pr.stepDtMs    = dt * 1000.0;
            pr.horizonMs   = steps_ * dt * 1000.0;
            pr.reliable    = (s.history.size() >= 5);
            results.push_back(std::move(pr));
        }
        return results;
    }

    void drawPredictions(cv::Mat& frame,
                         const std::vector<PredictionResult>& results,
                         const cv::Mat& K, const cv::Mat& D) const {
        for (const auto& pr : results) {
            if (!pr.reliable || pr.futurePath.empty()) continue;

            std::vector<cv::Point3f> pts3;
            pts3.emplace_back((float)pr.smoothedPos.x,
                              (float)pr.smoothedPos.y,
                              (float)pr.smoothedPos.z);
            for (const auto& p : pr.futurePath)
                pts3.emplace_back((float)p.x,(float)p.y,(float)p.z);

            std::vector<cv::Point2f> pts2;
            cv::Mat r0 = cv::Mat::zeros(3,1,CV_64F);
            cv::Mat t0 = cv::Mat::zeros(3,1,CV_64F);
            cv::projectPoints(pts3, r0, t0, K, D, pts2);

            for (size_t i = 1; i < pts2.size(); ++i) {
                double a = (double)i / pts2.size();
                cv::Scalar col((int)(255*(1-a)), 255, (int)(255*a));
                cv::line(frame,
                         {(int)pts2[i-1].x,(int)pts2[i-1].y},
                         {(int)pts2[i  ].x,(int)pts2[i  ].y},
                         col, 2, cv::LINE_AA);
            }
            if (!pts2.empty())
                cv::circle(frame,
                           {(int)pts2.back().x,(int)pts2.back().y},
                           5, {0,140,255}, cv::FILLED);

            double spd = cv::norm(
                cv::Vec3d(pr.velocity.x,pr.velocity.y,pr.velocity.z));
            char buf[48];
            std::snprintf(buf, sizeof(buf), "v=%.2fm/s", spd);
            cv::putText(frame, buf,
                        {(int)pts2[0].x+8,(int)pts2[0].y-22},
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, {0,255,255}, 1);
        }
    }

    void resetMarker(int id) { kfs_.erase(id); }
    void resetAll()           { kfs_.clear();  }

private:
    int    steps_;
    double pNoise_, mNoise_;
    std::map<int,KFState> kfs_;

    void initKF(KFState& s, const cv::Point3d& p0) const {
        s.kf.init(6, 3, 0, CV_64F);
        s.kf.transitionMatrix = (cv::Mat_<double>(6,6) <<
            1,0,0, 1,0,0,
            0,1,0, 0,1,0,
            0,0,1, 0,0,1,
            0,0,0, 1,0,0,
            0,0,0, 0,1,0,
            0,0,0, 0,0,1);
        s.kf.measurementMatrix = cv::Mat::zeros(3,6,CV_64F);
        s.kf.measurementMatrix.at<double>(0,0) = 1;
        s.kf.measurementMatrix.at<double>(1,1) = 1;
        s.kf.measurementMatrix.at<double>(2,2) = 1;
        cv::setIdentity(s.kf.processNoiseCov,     cv::Scalar(pNoise_));
        for (int i=3;i<6;++i)
            s.kf.processNoiseCov.at<double>(i,i) = pNoise_*10.0;
        cv::setIdentity(s.kf.measurementNoiseCov, cv::Scalar(mNoise_));
        cv::setIdentity(s.kf.errorCovPost,         cv::Scalar(1.0));
        s.kf.statePost.at<double>(0) = p0.x;
        s.kf.statePost.at<double>(1) = p0.y;
        s.kf.statePost.at<double>(2) = p0.z;
        s.initialised = true;
    }

    static double clampDt(double lastMs, double nowMs) {
        return std::max(0.001, std::min((nowMs-lastMs)/1000.0, 0.5));
    }
};

enum class AlertLevel { SAFE=0, WARNING=1, DANGER=2 };

struct CollisionZone {
    std::string  name;
    cv::Point3d  centre;
    double       dangerRadius;
    double       warnRadius;
    cv::Scalar   colour;
};

struct CollisionAlert {
    int         markerId;
    std::string zoneName;
    AlertLevel  level;
    double      distToZone;
    double      timeToImpactMs;
    cv::Point3d robotPos;
};

using AlertCB = std::function<void(const CollisionAlert&)>;

class CollisionDetector {
public:
    void addZone(const CollisionZone& z) { zones_.push_back(z); }
    void setCallback(AlertCB cb)         { cb_ = std::move(cb); }

    std::vector<CollisionAlert> check(
        const std::vector<PredictionResult>& preds)
    {
        std::vector<CollisionAlert> alerts;
        for (const auto& pr : preds) {
            for (const auto& z : zones_) {
                double     curDist = distToBoundary(pr.smoothedPos, z);
                AlertLevel level   = AlertLevel::SAFE;
                double     tti     = -1.0;

                if (curDist <= 0.0) {
                    level = AlertLevel::DANGER;
                } else {
                    double dw = cv::norm(pr.smoothedPos-z.centre)-z.warnRadius;
                    if (dw <= 0.0) level = AlertLevel::WARNING;
                    for (size_t step=0; step<pr.futurePath.size(); ++step)
                        if (distToBoundary(pr.futurePath[step],z) <= 0.0) {
                            tti   = (double)(step+1)*pr.stepDtMs;
                            level = AlertLevel::WARNING;
                            break;
                        }
                }
                if (level==AlertLevel::SAFE) continue;

                CollisionAlert a;
                a.markerId       = pr.markerId;
                a.zoneName       = z.name;
                a.level          = level;
                a.distToZone     = curDist;
                a.timeToImpactMs = tti;
                a.robotPos       = pr.smoothedPos;
                alerts.push_back(a);
                if (cb_) cb_(a);
            }
        }
        return alerts;
    }

    void drawHUD(cv::Mat& frame,
                 const std::vector<CollisionAlert>& alerts) const {
        int y = frame.rows-14;
        for (auto it=alerts.rbegin(); it!=alerts.rend(); ++it, y-=22) {
            bool danger = (it->level==AlertLevel::DANGER);
            cv::Scalar col = danger ? cv::Scalar(0,0,255):cv::Scalar(0,140,255);
            std::ostringstream ss;
            if (danger)
                ss<<"[DANGER]  ID"<<it->markerId<<" inside "<<it->zoneName;
            else {
                ss<<"[WARN]    ID"<<it->markerId<<" near "<<it->zoneName;
                if (it->timeToImpactMs>0)
                    ss<<"  ~"<<(int)it->timeToImpactMs<<"ms";
            }
            std::string msg=ss.str();
            cv::rectangle(frame,{0,y-16},{(int)msg.size()*9+10,y+4},
                          {0,0,0},cv::FILLED);
            cv::putText(frame,msg,{6,y},
                        cv::FONT_HERSHEY_SIMPLEX,0.52,col,1);
        }
    }

    void drawTopDown(cv::Mat& canvas,
                     const std::vector<PredictionResult>& preds,
                     const std::vector<CollisionAlert>& alerts,
                     double ppm, cv::Point origin) const {
        std::map<std::string,AlertLevel> lut;
        for (const auto& a:alerts) lut[a.zoneName]=a.level;

        for (const auto& z:zones_) {
            cv::Point ctr(origin.x+(int)(z.centre.x*ppm),
                          origin.y-(int)(z.centre.z*ppm));
            cv::Scalar col=z.colour;
            auto it=lut.find(z.name);
            if (it!=lut.end())
                col=(it->second==AlertLevel::DANGER)
                    ?cv::Scalar(0,0,255):cv::Scalar(0,140,255);
            cv::circle(canvas,ctr,(int)(z.warnRadius*ppm),col,1,cv::LINE_AA);
            cv::Mat ov=canvas.clone();
            cv::circle(ov,ctr,(int)(z.dangerRadius*ppm),col,cv::FILLED);
            cv::addWeighted(ov,0.25,canvas,0.75,0,canvas);
            cv::circle(canvas,ctr,(int)(z.dangerRadius*ppm),col,2,cv::LINE_AA);
            cv::putText(canvas,z.name,ctr+cv::Point(4,-4),
                        cv::FONT_HERSHEY_SIMPLEX,0.38,col,1);
        }
        for (const auto& pr:preds) {
            cv::Point cur(origin.x+(int)(pr.smoothedPos.x*ppm),
                          origin.y-(int)(pr.smoothedPos.z*ppm));
            cv::circle(canvas,cur,7,{0,220,0},cv::FILLED);
            cv::putText(canvas,"R"+std::to_string(pr.markerId),
                        cur+cv::Point(9,4),
                        cv::FONT_HERSHEY_SIMPLEX,0.38,{0,220,0},1);
            cv::Point prev=cur;
            for (const auto& p:pr.futurePath) {
                cv::Point next(origin.x+(int)(p.x*ppm),
                               origin.y-(int)(p.z*ppm));
                cv::line(canvas,prev,next,{0,180,255},1);
                prev=next;
            }
        }
    }

private:
    std::vector<CollisionZone> zones_;
    AlertCB                    cb_;

    static double distToBoundary(const cv::Point3d& p,
                                  const CollisionZone& z) {
        cv::Point3d d=p-z.centre;
        return std::sqrt(d.x*d.x+d.y*d.y+d.z*d.z)-z.dangerRadius;
    }
};

enum class NavCmd {
    STOP, MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, HOLD
};

std::string navCmdName(NavCmd c) {
    switch(c) {
        case NavCmd::STOP:          return "STOP";
        case NavCmd::MOVE_FORWARD:  return "FWD";
        case NavCmd::MOVE_BACKWARD: return "BWD";
        case NavCmd::TURN_LEFT:     return "TURN_L";
        case NavCmd::TURN_RIGHT:    return "TURN_R";
        case NavCmd::HOLD:          return "HOLD";
        default:                    return "?";
    }
}

struct NavOutput {
    NavCmd      cmd;
    double      distToTarget;
    double      yawError;
    std::string statusMsg;
    bool        targetVisible;
    bool        dangerActive;
};

struct NavConfig {
    int    targetId       = 0;
    double stopDistM      = 0.30;
    double alignThreshDeg = 5.0;
    double posThreshM     = 0.05;
};

class RobotNavigator {
public:

    explicit RobotNavigator(NavConfig cfg = NavConfig())
        : cfg_(cfg), state_(State::SEARCHING), lostFrames_(0) {}

    NavOutput tick(const std::vector<MarkerPose>& poses,
                   const std::vector<CollisionAlert>& alerts)
    {
        for (const auto& a : alerts)
            if (a.level == AlertLevel::DANGER) {
                state_ = State::EMERGENCY;
                return makeOut(NavCmd::STOP, -1, 0,
                               "!!! EMERGENCY STOP", false, true);
            }
        if (state_ == State::EMERGENCY) state_ = State::SEARCHING;

        const MarkerPose* tgt = nullptr;
        for (const auto& mp : poses)
            if (mp.id == cfg_.targetId) { tgt = &mp; break; }

        if (!tgt) {
            if (++lostFrames_ > 30) state_ = State::SEARCHING;
            return makeOut(NavCmd::TURN_RIGHT, -1, 0,
                           "SEARCHING for ID "+std::to_string(cfg_.targetId),
                           false, false);
        }
        lostFrames_ = 0;

        switch (state_) {
            case State::SEARCHING:
                state_ = State::APPROACHING;

            case State::APPROACHING:
                if (tgt->distance <= cfg_.stopDistM + cfg_.posThreshM)
                    state_ = State::ALIGNING;
                return doApproach(*tgt, alerts);

            case State::ALIGNING:
                if (std::fabs(markerBearingDeg(*tgt)) <= cfg_.alignThreshDeg)
                    state_ = State::HOLDING;
                return doAlign(*tgt);

            case State::HOLDING:
                if (tgt->distance > cfg_.stopDistM + cfg_.posThreshM*3.0) {
                    state_ = State::APPROACHING;
                    return doApproach(*tgt, alerts);
                }
                return makeOut(NavCmd::HOLD, tgt->distance, tgt->yaw,
                               "HOLDING  dist="
                               +std::to_string(tgt->distance).substr(0,4)+"m",
                               true, false);
            default:
                return makeOut(NavCmd::STOP,-1,0,"UNKNOWN",false,false);
        }
    }

    void drawHUD(cv::Mat& frame, const NavOutput& out) const {
        cv::Mat ov=frame.clone();
        cv::rectangle(ov,{0,0},{frame.cols,52},{20,20,20},cv::FILLED);
        cv::addWeighted(ov,0.72,frame,0.28,0,frame);
        cv::Scalar col=out.dangerActive?cv::Scalar(0,0,255):cv::Scalar(60,230,80);
        cv::putText(frame,out.statusMsg,{10,22},
                    cv::FONT_HERSHEY_SIMPLEX,0.55,col,1);
        cv::putText(frame,"CMD: "+navCmdName(out.cmd),{10,44},
                    cv::FONT_HERSHEY_SIMPLEX,0.48,{255,220,0},1);
        std::string sn=stateName(state_);
        int base=0;
        cv::Size sz=cv::getTextSize(sn,cv::FONT_HERSHEY_SIMPLEX,0.48,1,&base);
        cv::putText(frame,sn,{frame.cols-sz.width-10,22},
                    cv::FONT_HERSHEY_SIMPLEX,0.48,{180,180,255},1);
    }

    void reset() { state_=State::SEARCHING; lostFrames_=0; }

private:
    enum class State { SEARCHING, APPROACHING, ALIGNING, HOLDING, EMERGENCY };
    NavConfig cfg_;
    State     state_;
    int       lostFrames_;

    NavOutput doApproach(const MarkerPose& t,
                         const std::vector<CollisionAlert>& alerts) {
        bool warn=false;
        for (const auto& a:alerts)
            if (a.level==AlertLevel::WARNING){warn=true;break;}
        double bearingDeg = markerBearingDeg(t);
        NavCmd cmd;
        if (std::fabs(bearingDeg)>cfg_.alignThreshDeg*2.0)
            cmd=(bearingDeg>0)?NavCmd::TURN_RIGHT:NavCmd::TURN_LEFT;
        else
            cmd=warn?NavCmd::HOLD:NavCmd::MOVE_FORWARD;
        std::ostringstream ss;
        ss<<"APPROACHING  dist="<<std::fixed<<std::setprecision(2)
          <<t.distance<<"m  bearing="<<std::setprecision(1)<<bearingDeg<<"deg";
        if (warn) ss<<"  [COLLISION WARNING]";
        return makeOut(cmd,t.distance,bearingDeg,ss.str(),true,warn);
    }

    NavOutput doAlign(const MarkerPose& t) {
        double bearingDeg = markerBearingDeg(t);
        NavCmd cmd=(bearingDeg>0)?NavCmd::TURN_RIGHT:NavCmd::TURN_LEFT;
        std::ostringstream ss;
        ss<<"ALIGNING  bearing="<<std::fixed<<std::setprecision(1)<<bearingDeg<<"deg";
        return makeOut(cmd,t.distance,bearingDeg,ss.str(),true,false);
    }

    static double markerBearingDeg(const MarkerPose& t) {
        return std::atan2(t.tvec[0], t.tvec[2]) * 180.0 / CV_PI;
    }

    static NavOutput makeOut(NavCmd cmd, double dist, double yaw,
                              const std::string& msg,
                              bool visible, bool danger) {
        return {cmd,dist,yaw,msg,visible,danger};
    }

    static std::string stateName(State s) {
        switch(s) {
            case State::SEARCHING:   return "SEARCHING";
            case State::APPROACHING: return "APPROACHING";
            case State::ALIGNING:    return "ALIGNING";
            case State::HOLDING:     return "HOLDING";
            case State::EMERGENCY:   return "EMERGENCY STOP";
            default:                 return "UNKNOWN";
        }
    }
};

class Logger {
public:
    explicit Logger(const std::string& path)
        : runId_(std::chrono::duration_cast<std::chrono::seconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count())
    {
        bool hasData = false;
        {
            std::ifstream probe(path, std::ios::ate);
            hasData = probe.good() && probe.tellg() > 0;
        }
        file_.open(path, std::ios::out | std::ios::app);
        if (file_.is_open() && !hasData)
            file_ << "run_id,frame,det_ms,pose_ms,pred_ms,total_ms,markers\n";
    }
    ~Logger() { if (file_.is_open()) file_.close(); }

    void log(int frame, double det, double pose,
             double pred, double total, int markers) {
        dV_.push_back(det); pV_.push_back(pose);
        prV_.push_back(pred); tV_.push_back(total);
        if (file_.is_open())
            file_<<runId_<<","<<frame<<","<<std::fixed<<std::setprecision(3)
                 <<det<<","<<pose<<","<<pred<<","<<total<<","<<markers<<"\n";
    }

    void printSummary() const {
        if (tV_.empty()) return;
        auto avg=[](const std::vector<double>& v){
            return std::accumulate(v.begin(),v.end(),0.0)/v.size();
        };
        double mT=avg(tV_);
        std::cout<<"\n========== Performance Summary ==========\n"
                 <<std::fixed<<std::setprecision(2)
                 <<"  Frames      : "<<tV_.size()            <<"\n"
                 <<"  Avg detect  : "<<avg(dV_)              <<" ms\n"
                 <<"  Avg pose    : "<<avg(pV_)              <<" ms\n"
                 <<"  Avg predict : "<<avg(prV_)             <<" ms\n"
                 <<"  Avg total   : "<<mT                    <<" ms\n"
                 <<"  Avg FPS     : "<<(mT>0?1000.0/mT:0)   <<"\n"
                 <<"=========================================\n";
    }

private:
    long long            runId_;
    std::ofstream        file_;
    std::vector<double>  dV_,pV_,prV_,tV_;
};

void generateMarkers(int count=10, int sizePx=200) {
    MAKE_DIR("markers");
    cv::aruco::Dictionary dict =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100);
    for (int id=0; id<count; ++id) {
        cv::Mat img;

        cv::aruco::generateImageMarker(dict, id, sizePx, img, 1);
        int brd=sizePx/8;
        cv::Mat out(sizePx+2*brd+24, sizePx+2*brd, CV_8UC1, cv::Scalar(255));
        img.copyTo(out(cv::Rect(brd,brd,sizePx,sizePx)));
        cv::putText(out,"ID "+std::to_string(id),
                    {brd,sizePx+2*brd+16},
                    cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0),1);
        std::string fn="markers/marker_"+std::to_string(id)+".png";
        cv::imwrite(fn,out);
        std::cout<<"Saved: "<<fn<<"\n";
    }
    std::cout<<"\nDone. Print markers/marker_0.png and hold "
               "it in front of the webcam.\n";
}

cv::Mat makeTopDown(int sz=400) {
    cv::Mat c(sz,sz,CV_8UC3,cv::Scalar(18,18,18));
    for (int i=0;i<sz;i+=50) {
        cv::line(c,{i,0},{i,sz},{35,35,35},1);
        cv::line(c,{0,i},{sz,i},{35,35,35},1);
    }
    cv::putText(c,"Top-Down View",{6,14},
                cv::FONT_HERSHEY_SIMPLEX,0.38,{80,80,80},1);
    return c;
}

int main(int argc, char** argv)
{
    std::cout << "PROGRAM STARTED\n";
    std::string videoPath, calibPath;
    double markerLen   = 0.05;
    int    targetId    = 0;
    int    predSteps   = 10;
    bool   showTopDown = true;
    bool   genMarkers  = false;

    auto errExit = [](const std::string& msg) {
        std::cerr << "ERROR: " << msg << "\n"
                   << "Run with --help for usage.\n";
        std::exit(1);
    };

    for (int i=1; i<argc; ++i) {
        std::string s=argv[i];
        auto needsValue = [&](const char* flag) -> std::string {
            if (i+1>=argc) errExit(std::string(flag)+" requires a value");
            return argv[++i];
        };

        if (s=="--video") {
            videoPath=needsValue("--video");
        } else if (s=="--calib") {
            calibPath=needsValue("--calib");
        } else if (s=="--marker-len") {
            std::string v=needsValue("--marker-len");
            try { markerLen=std::stod(v); }
            catch (...) { errExit("--marker-len expects a number, got '"+v+"'"); }
            if (!(markerLen>0.0) || markerLen>5.0)
                errExit("--marker-len must be a positive value in metres, "
                         "0 < len <= 5.0 (got "+v+")");
        } else if (s=="--target") {
            std::string v=needsValue("--target");
            try { targetId=std::stoi(v); }
            catch (...) { errExit("--target expects an integer, got '"+v+"'"); }
            if (targetId<0 || targetId>=100)
                errExit("--target must be a marker ID between 0 and 99 "
                         "(got "+v+")");
        } else if (s=="--pred") {
            std::string v=needsValue("--pred");
            try { predSteps=std::stoi(v); }
            catch (...) { errExit("--pred expects an integer, got '"+v+"'"); }
            if (predSteps<1 || predSteps>100)
                errExit("--pred must be between 1 and 100 (got "+v+")");
        } else if (s=="--no-topdown") {
            showTopDown=false;
        } else if (s=="--generate-markers") {
            genMarkers=true;
        } else if (s=="--help") {
            std::cout
                <<"Usage: ./aruco_navigation [options]\n\n"
                <<"  --video <file>       video file (default: webcam)\n"
                <<"  --calib <yaml>       camera calibration file\n"
                <<"  --marker-len <m>     marker side in metres (default 0.05)\n"
                <<"  --target <id>        marker ID to navigate to, 0-99 (default 0)\n"
                <<"  --pred <n>           number of future prediction steps, 1-100 (default 10)\n"
                <<"  --no-topdown         disable bird's-eye window\n"
                <<"  --generate-markers   save marker PNGs and exit\n\n"
                <<"Keys while running: q=quit  r=reset  s=save_frame\n";
            return 0;
        } else {
            errExit("unknown option '"+s+"'");
        }
    }

    if (genMarkers) { generateMarkers(); return 0; }

    std::cout
        <<"\n+----------------------------------------------+\n"
        <<"|    ArUco Robot Navigation  —  Starting Up   |\n"
        <<"+----------------------------------------------+\n\n";

    std::cout << "[STEP 1] Opening video source...\n";
    std::cout.flush();

    cv::VideoCapture cap;
    if (videoPath.empty()) {
        std::cout << "  Trying webcam index 0 with DirectShow (Windows)...\n";
        std::cout.flush();
        cap.open(0, cv::CAP_DSHOW);
        if (!cap.isOpened()) {
            std::cout << "  DirectShow failed. Trying default backend...\n";
            std::cout.flush();
            cap.open(0);
        }
        if (!cap.isOpened()) {
            std::cout << "  Index 0 failed. Trying index 1...\n";
            std::cout.flush();
            cap.open(1, cv::CAP_DSHOW);
        }
        if (!cap.isOpened()) {
            std::cerr << "\nERROR: No webcam found!\n"
                      << "  - Make sure your webcam is plugged in\n"
                      << "  - Make sure no other app (Zoom, Teams, OBS) is using it\n"
                      << "  - Or use:  ./app --video yourfile.mp4\n";
            return 1;
        }
        std::cout << "  Webcam opened OK!\n";
        cap.set(cv::CAP_PROP_FRAME_WIDTH,  640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);
    } else {
        cap.open(videoPath);
        if (!cap.isOpened()) {
            std::cerr << "ERROR: Cannot open video file: " << videoPath << "\n";
            return 1;
        }
        std::cout << "  Video file opened OK: " << videoPath << "\n";
    }
    std::cout.flush();

    std::cout << "[STEP 2] Flushing warm-up frames...\n";
    std::cout.flush();
    { cv::Mat dummy; for (int w=0; w<10; ++w) cap.read(dummy); }

    std::cout << "[STEP 3] Reading test frame...\n";
    std::cout.flush();
    cv::Size actualFrameSize;
    {
        cv::Mat test;
        if (!cap.read(test) || test.empty()) {
            std::cerr << "ERROR: Webcam opened but cannot read frames!\n"
                      << "  This is a Windows driver issue. Try:\n"
                      << "  1. Unplug and replug webcam\n"
                      << "  2. Close Zoom/Teams/any app using camera\n"
                      << "  3. Use --video yourfile.mp4 instead\n";
            return 1;
        }
        actualFrameSize = test.size();
        std::cout << "  Frame OK! Size: " << test.cols << "x" << test.rows << "\n";
    }
    std::cout.flush();

    CameraParams cam;
    if (!calibPath.empty()) {
        auto p=CameraParams::fromFile(calibPath);
        cam=p.value_or(CameraParams::defaultWebcam());
        std::cout<<"Calibration: "<<(p?calibPath:"FALLBACK defaults")<<"\n";
    } else {
        cam=CameraParams::defaultWebcam();
        std::cout<<"Calibration: default webcam approximation\n"
                 <<"  WARNING: no --calib file supplied. Distance and pose\n"
                 <<"  values will only be rough approximations, not accurate\n"
                 <<"  measurements. Calibrate your camera and pass --calib\n"
                 <<"  <file.yaml> for real metric accuracy.\n";
    }

    if (cam.imageSize.width>0 && cam.imageSize.height>0 &&
        actualFrameSize.width>0 && actualFrameSize.height>0 &&
        cam.imageSize != actualFrameSize) {
        double sx = (double)actualFrameSize.width  / cam.imageSize.width;
        double sy = (double)actualFrameSize.height / cam.imageSize.height;
        std::cout<<"  WARNING: calibration resolution ("<<cam.imageSize.width
                 <<"x"<<cam.imageSize.height<<") differs from the camera's "
                 <<"actual resolution ("<<actualFrameSize.width<<"x"
                 <<actualFrameSize.height<<"). Scaling the camera matrix to "
                 <<"compensate, but re-calibrating at the actual resolution "
                 <<"is recommended.\n";
        cam.cameraMatrix.at<double>(0,0) *= sx;
        cam.cameraMatrix.at<double>(1,1) *= sy;
        cam.cameraMatrix.at<double>(0,2) *= sx;
        cam.cameraMatrix.at<double>(1,2) *= sy;
        cam.imageSize = actualFrameSize;
    }

    std::cout<<"Marker len : "<<markerLen<<"m  |  Target ID: "<<targetId<<"\n\n";

    std::cout << "[STEP 4] Initialising modules...\n"; std::cout.flush();

    ArucoDetector      detector;
    PoseEstimator      estimator(cam, markerLen);
    TrajectoryPredictor predictor(predSteps);
    CollisionDetector  collider;

    collider.addZone({"Zone-A",{0.50,0.0,1.0},0.15,0.35,{0,0,200}  });
    collider.addZone({"Zone-B",{-0.3,0.0,0.8},0.12,0.30,{0,100,200}});
    collider.setCallback([](const CollisionAlert& a){
        std::cout<<"[COLLISION] "
                 <<(a.level==AlertLevel::DANGER?"DANGER":"WARNING")
                 <<"  ID="<<a.markerId<<"  zone="<<a.zoneName<<"\n";
    });

    NavConfig navCfg;
    navCfg.targetId = targetId;
    RobotNavigator navigator(navCfg);

    MAKE_DIR("evaluation");
    Logger logger("evaluation/metrics.csv");

    cv::Mat frame;
    int frameId=0;
    std::cout<<"[STEP 5] All OK! Opening camera window now...\n"
               "         Hold markers/marker_0.png in front of the webcam.\n"
               "         Keys: q=quit  r=reset  s=save_frame\n\n";
    std::cout.flush();

    while (true) {
        if (!cap.read(frame)||frame.empty()){ std::cout<<"End of stream.\n"; break; }

        auto tStart=HRC::now();
        double nowMs=std::chrono::duration<double,std::milli>(
                         tStart.time_since_epoch()).count();

        auto t0=HRC::now();
        auto det=detector.detect(frame);
        double detMs=std::chrono::duration<double,std::milli>(HRC::now()-t0).count();

        auto t1=HRC::now();
        auto poses=estimator.estimate(det);
        double poseMs=std::chrono::duration<double,std::milli>(HRC::now()-t1).count();

        auto t2=HRC::now();
        auto preds=predictor.update(poses,nowMs);
        double predMs=std::chrono::duration<double,std::milli>(HRC::now()-t2).count();

        auto alerts=collider.check(preds);
        auto navOut=navigator.tick(poses,alerts);

        cv::Mat disp=detector.visualise(frame,det);
        disp=estimator.visualise(disp,poses);
        predictor.drawPredictions(disp,preds,cam.cameraMatrix,cam.distCoeffs);
        collider.drawHUD(disp,alerts);
        navigator.drawHUD(disp,navOut);

        double totalMs=std::chrono::duration<double,std::milli>(
                           HRC::now()-tStart).count();
        char fpsBuf[24];
        std::snprintf(fpsBuf,sizeof(fpsBuf),"%.0f fps",1000.0/totalMs);
        cv::putText(disp,fpsBuf,{disp.cols-72,disp.rows-8},
                    cv::FONT_HERSHEY_SIMPLEX,0.5,{160,160,160},1);

        cv::imshow("ArUco Robot Navigation",disp);
        if (showTopDown) {
            cv::Mat td=makeTopDown();
            collider.drawTopDown(td,preds,alerts,100.0,{td.cols/2,td.rows-20});
            cv::imshow("Top-Down View",td);
        }

        logger.log(frameId,detMs,poseMs,predMs,totalMs,(int)det.markerIds.size());
        ++frameId;

        char k=static_cast<char>(cv::waitKey(1));
        if (k=='q'||k==27) break;
        if (k=='r'){ navigator.reset(); predictor.resetAll();
                     std::cout<<"Reset.\n"; }
        if (k=='s'){ std::string fn="frame_"+std::to_string(frameId)+".png";
                     cv::imwrite(fn,disp); std::cout<<"Saved: "<<fn<<"\n"; }
    }

    logger.printSummary();
    std::cout<<"Done. "<<frameId<<" frames processed.\n"
               "Metrics saved to evaluation/metrics.csv\n";
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
