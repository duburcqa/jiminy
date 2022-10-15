// Test the sanity of the simulation engine.
// The tests in this file verify that the behavior of a simulated system matches
// real-world physics, and that no memory is allocated by Eigen during a simulation.
// The test system is a double inverted pendulum.
#include <gtest/gtest.h>

#define EIGEN_RUNTIME_NO_MALLOC

#include "jiminy/core/robot/Model.h"
#include "jiminy/core/utilities/Helpers.h"
#include "jiminy/core/Types.h"

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/geometry.hpp"

using namespace jiminy;

float64_t const TOLERANCE = 1e-9;


TEST(Model, CreateFlexible)
{
    // Create a flexible model, and very that it makes sense.

    // Double pendulum model
    std::string const dataDirPath(UNIT_TEST_DATA_DIR);
    auto const urdfPath = dataDirPath + "/branching_pendulum.urdf";

    // All joints actuated.
    auto model = std::make_shared<Model>();
    model->initialize(urdfPath, true, std::vector<std::string>(), true);

    // We now have a rigid robot: perform rigid computation on this.
    auto q = pinocchio::randomConfiguration(model->pncModel_);
    // Reset freeflyer to finite value.
    q[0] = 42;
    q[1] = 42;
    q[2] = 42;
    auto pncData = pinocchio::Data(model->pncModel_);
    pinocchio::framesForwardKinematics(model->pncModel_, pncData, q);

    // Model is rigid, so configuration should not change.
    vectorN_t qflex;
    ASSERT_TRUE(model->getFlexibleConfigurationFromRigid(q, qflex) == hresult_t::SUCCESS);
    ASSERT_TRUE(qflex.isApprox(q));

    auto visualData = pinocchio::GeometryData(model->visualModel_);
    pinocchio::updateGeometryPlacements(model->pncModel_, pncData, model->visualModel_, visualData);

    auto collisionData = pinocchio::GeometryData(model->collisionModel_);
    pinocchio::updateGeometryPlacements(model->pncModel_, pncData, model->collisionModel_, collisionData);


    // Add flexibility to joint and frame.
    auto options = model->getOptions();
    flexibilityConfig_t flexConfig;
    vector3_t v = vector3_t::Constant(42);
    flexConfig.push_back(flexibleJointData_t{"PendulumJoint", v, v, v});
    flexConfig.push_back(flexibleJointData_t{"PendulumMassJoint", v, v, v});
    boost::get<flexibilityConfig_t>(boost::get<configHolder_t>(options.at("dynamics")).at("flexibilityConfig")) = flexConfig;
    model->setOptions(options);
    model->reset();


    ASSERT_TRUE(model->getFlexibleConfigurationFromRigid(q, qflex) == hresult_t::SUCCESS);
    ASSERT_EQ(qflex.size(), q.size() + 8);

    // Recompute frame, geometry and collision pose, and check that nothing has moved.
    pinocchio::framesForwardKinematics(model->pncModel_, model->pncData_, qflex);
    pinocchio::updateGeometryPlacements(model->pncModel_, model->pncData_, model->visualModel_, model->visualData_);
    pinocchio::updateGeometryPlacements(model->pncModel_, model->pncData_, model->collisionModel_, model->collisionData_);

    for (unsigned int i = 0; i < model->pncModelOrig_.frames.size(); i++)
    {
        long unsigned int flexId = model->pncModel_.getFrameId(model->pncModelOrig_.frames[i].name);
        ASSERT_TRUE(pncData.oMf[i].toHomogeneousMatrix().isApprox(model->pncData_.oMf[flexId].toHomogeneousMatrix()));
    }

    for (unsigned int i = 0; i < model->visualData_.oMg.size(); i++)
    {
        ASSERT_TRUE(model->visualData_.oMg[i].toHomogeneousMatrix().isApprox(visualData.oMg[i].toHomogeneousMatrix()));
    }

    for (unsigned int i = 0; i < model->collisionData_.oMg.size(); i++)
    {
        ASSERT_TRUE(model->collisionData_.oMg[i].toHomogeneousMatrix().isApprox(collisionData.oMg[i].toHomogeneousMatrix()));
    }


}
