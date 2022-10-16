#include <gtest/gtest.h>

#include "jiminy/core/robot/Model.h"
#include "jiminy/core/utilities/Helpers.h"
#include "jiminy/core/Types.h"

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

using namespace jiminy;


class ModelTestFixture :
    public testing::TestWithParam<bool> {
};


TEST_P(ModelTestFixture, CreateFlexible)
{
    // Create a flexible model, and verify that it makes sense.
    bool const hasFreeflyer = GetParam();

    // Double pendulum model
    std::string const dataDirPath(UNIT_TEST_DATA_DIR);
    auto const urdfPath = dataDirPath + "/branching_pendulum.urdf";

    // All joints actuated.
    auto model = std::make_shared<Model>();
    model->initialize(urdfPath, hasFreeflyer, std::vector<std::string>(), true);

    // We now have a rigid robot: perform rigid computation on this.
    auto q = pinocchio::randomConfiguration(model->pncModel_);
    if (hasFreeflyer)
        q.head<3>().setZero();
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
    vector3_t v = vector3_t::Ones();
    flexConfig.push_back(flexibleJointData_t{"PendulumJoint", v, v, v});
    flexConfig.push_back(flexibleJointData_t{"PendulumMassJoint", v, v, v});
    boost::get<flexibilityConfig_t>(boost::get<configHolder_t>(options.at("dynamics")).at("flexibilityConfig")) = flexConfig;
    model->setOptions(options);
    model->reset();


    ASSERT_TRUE(model->getFlexibleConfigurationFromRigid(q, qflex) == hresult_t::SUCCESS);
    ASSERT_EQ(qflex.size(), q.size() + quaternion_t::Coefficients::RowsAtCompileTime * flexConfig.size());

    // Recompute frame, geometry and collision pose, and check that nothing has moved.
    pinocchio::framesForwardKinematics(model->pncModel_, model->pncData_, qflex);
    pinocchio::updateGeometryPlacements(model->pncModel_, model->pncData_, model->visualModel_, model->visualData_);
    pinocchio::updateGeometryPlacements(model->pncModel_, model->pncData_, model->collisionModel_, model->collisionData_);

    for (uint32_t i = 0; i < model->pncModelOrig_.frames.size(); i++)
    {
        uint32_t const flexId = static_cast<uint32_t>(model->pncModel_.getFrameId(model->pncModelOrig_.frames[i].name));
        ASSERT_TRUE(pncData.oMf[i].isApprox(model->pncData_.oMf[flexId]));
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

INSTANTIATE_TEST_SUITE_P(ModelTests,
                         ModelTestFixture,
                         testing::Values(true, false));

