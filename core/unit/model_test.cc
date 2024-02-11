#include <gtest/gtest.h>

#include "jiminy/core/fwd.h"
#include "jiminy/core/utilities/helpers.h"
#include "jiminy/core/robot/model.h"

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

using namespace jiminy;


class ModelTestFixture : public testing::TestWithParam<bool>
{
};


TEST_P(ModelTestFixture, CreateFlexible)
{
    // Create a flexible model, and verify that it makes sense
    const bool hasFreeflyer = GetParam();

    // Double pendulum model
    const std::string dataDirPath(UNIT_TEST_DATA_DIR);
    const auto urdfPath = dataDirPath + "/branching_pendulum.urdf";

    // All joints actuated
    auto model = std::make_shared<Model>();
    model->initialize(urdfPath, hasFreeflyer, std::vector<std::string>(), true);

    // We now have a rigid robot: perform rigid computation on this
    auto q = pinocchio::randomConfiguration(model->pinocchioModel_);
    if (hasFreeflyer)
    {
        q.head<3>().setZero();
    }
    auto pinocchioData = pinocchio::Data(model->pinocchioModel_);
    pinocchio::framesForwardKinematics(model->pinocchioModel_, pinocchioData, q);

    // Model is rigid, so configuration should not change
    Eigen::VectorXd qflex;
    model->getFlexiblePositionFromRigid(q, qflex);
    ASSERT_TRUE(qflex.isApprox(q));

    auto visualData = pinocchio::GeometryData(model->visualModel_);
    pinocchio::updateGeometryPlacements(
        model->pinocchioModel_, pinocchioData, model->visualModel_, visualData);

    auto collisionData = pinocchio::GeometryData(model->collisionModel_);
    pinocchio::updateGeometryPlacements(
        model->pinocchioModel_, pinocchioData, model->collisionModel_, collisionData);

    // Add flexibility to joint and frame
    auto options = model->getOptions();
    FlexibilityConfig flexConfig;
    Eigen::Vector3d v = Eigen::Vector3d::Ones();
    flexConfig.push_back({"PendulumJoint", v, v, v});
    flexConfig.push_back({"PendulumMassJoint", v, v, v});
    GenericConfig & dynamicsOptions = boost::get<GenericConfig>(options.at("dynamics"));
    boost::get<FlexibilityConfig>(dynamicsOptions.at("flexibilityConfig")) = flexConfig;
    model->setOptions(options);

    model->getFlexiblePositionFromRigid(q, qflex);
    ASSERT_EQ(qflex.size(),
              q.size() + Eigen::Quaterniond::Coefficients::RowsAtCompileTime * flexConfig.size());

    // Recompute frame, geometry and collision pose, and check that nothing has moved.
    pinocchio::framesForwardKinematics(model->pinocchioModel_, model->pinocchioData_, qflex);
    pinocchio::updateGeometryPlacements(
        model->pinocchioModel_, model->pinocchioData_, model->visualModel_, model->visualData_);
    pinocchio::updateGeometryPlacements(model->pinocchioModel_,
                                        model->pinocchioData_,
                                        model->collisionModel_,
                                        model->collisionData_);

    for (uint32_t i = 0; i < model->pinocchioModelOrig_.frames.size(); i++)
    {
        const pinocchio::Frame & frame = model->pinocchioModelOrig_.frames[i];
        const pinocchio::FrameIndex flexIndex = model->pinocchioModel_.getFrameId(frame.name);
        ASSERT_TRUE(pinocchioData.oMf[i].isApprox(model->pinocchioData_.oMf[flexIndex]));
    }

    for (uint32_t i = 0; i < model->visualData_.oMg.size(); i++)
    {
        ASSERT_TRUE(model->visualData_.oMg[i].isApprox(visualData.oMg[i]));
    }

    for (uint32_t i = 0; i < model->collisionData_.oMg.size(); i++)
    {
        ASSERT_TRUE(model->collisionData_.oMg[i].isApprox(collisionData.oMg[i]));
    }
}

INSTANTIATE_TEST_SUITE_P(ModelTests, ModelTestFixture, testing::Values(true, false));
