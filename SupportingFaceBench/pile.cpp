// Stress scene: a pile of convex hulls falling into a box, single threaded.
// Reports steps / second, when the pile falls asleep, a hash of the final state and how deep the hull vertices end up below the floor.
// Usage: pile <bodies> <points per hull> <steps> <small|default> <seed> <bodies per row (8: tall pile, 32: one layer)>
//   small: rocks of 0.3 m with penetration slop 2 mm and convex radius error 1 mm (small objects, tuned settings)
//   default: rocks of 1 m with all Jolt default settings
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemSingleThreaded.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayerInterfaceTable.h>
#include <Jolt/Physics/Collision/BroadPhase/ObjectVsBroadPhaseLayerFilterTable.h>
#include <Jolt/Physics/Collision/ObjectLayerPairFilterTable.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace JPH;

int main(int argc, char **argv)
{
	int num_bodies = argc > 1? atoi(argv[1]) : 1000;
	int num_points = argc > 2? atoi(argv[2]) : 64;
	int num_steps = argc > 3? atoi(argv[3]) : 600;
	bool small = argc > 4? strcmp(argv[4], "default") != 0 : true;
	float s = small? 1.0f : 10.0f / 3.0f;
	uint seed = argc > 5? uint(atoi(argv[5])) : 4321;
	int per_row = argc > 6? atoi(argv[6]) : 8;
	float wall = 0.125f * per_row + 0.1f;

	RegisterDefaultAllocator();
	Factory::sInstance = new Factory();
	RegisterTypes();

	TempAllocatorImpl temp_allocator(64 * 1024 * 1024);
	JobSystemSingleThreaded job_system(cMaxPhysicsJobs);

	ObjectLayerPairFilterTable object_filter(2);
	object_filter.EnableCollision(0, 1);
	object_filter.EnableCollision(1, 1);
	BroadPhaseLayerInterfaceTable bp_interface(2, 2);
	bp_interface.MapObjectToBroadPhaseLayer(0, BroadPhaseLayer(0));
	bp_interface.MapObjectToBroadPhaseLayer(1, BroadPhaseLayer(1));
	ObjectVsBroadPhaseLayerFilterTable bp_filter(bp_interface, 2, object_filter, 2);

	PhysicsSystem system;
	system.Init(10240, 0, 65536, 65536, bp_interface, bp_filter, object_filter);
	if (small)
	{
		PhysicsSettings settings;
		settings.mPenetrationSlop = 0.002f;
		system.SetPhysicsSettings(settings);
	}
	float slop = system.GetPhysicsSettings().mPenetrationSlop;
	BodyInterface &bi = system.GetBodyInterface();

	// Floor (top at y = 0) and walls of a box of 2 x 2 m (scaled by s)
	bi.CreateAndAddBody(BodyCreationSettings(new BoxShape(s * Vec3(5, 0.5f, 5)), RVec3(0, -0.5f * s, 0), Quat::sIdentity(), EMotionType::Static, 0), EActivation::DontActivate);
	for (int i = 0; i < 4; ++i)
	{
		Vec3 half = i < 2? Vec3(0.1f, 5, wall) : Vec3(wall, 5, 0.1f);
		Vec3 pos = i == 0? Vec3(-wall, 5, 0) : i == 1? Vec3(wall, 5, 0) : i == 2? Vec3(0, 5, -wall) : Vec3(0, 5, wall);
		bi.CreateAndAddBody(BodyCreationSettings(new BoxShape(s * half), RVec3(s * pos), Quat::sIdentity(), EMotionType::Static, 0), EActivation::DontActivate);
	}

	// 20 rock-like hulls: points on an ellipsoid of 0.15 x 0.05 x 0.075 m (scaled by s) with 5% radial noise
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> u(-1, 1);
	Array<Ref<Shape>> shapes;
	uint total_points = 0;
	while (shapes.size() < 20)
	{
		Array<Vec3> points;
		while ((int)points.size() < num_points)
		{
			Vec3 v(u(rng), u(rng), u(rng));
			if (v.LengthSq() < 1.0e-6f || v.LengthSq() > 1)
				continue;
			v = v.Normalized() * (1.0f + 0.05f * u(rng));
			points.push_back(s * Vec3(0.15f * v.GetX(), 0.05f * v.GetY(), 0.075f * v.GetZ()));
		}
		ConvexHullShapeSettings settings(points);
		if (small)
		{
			settings.mMaxConvexRadius = 0.005f;
			settings.mMaxErrorConvexRadius = 0.001f;
		}
		Shape::ShapeResult result = settings.Create();
		if (result.IsValid())
		{
			total_points += static_cast<const ConvexHullShape *>(result.Get().GetPtr())->GetNumPoints();
			shapes.push_back(result.Get());
		}
	}

	// Drop the bodies in layers of per_row x per_row above the box
	Array<BodyID> bodies;
	for (int i = 0; i < num_bodies; ++i)
	{
		int layer = i / (per_row * per_row), cell = i % (per_row * per_row);
		float first = -0.125f * (per_row - 1);
		Vec3 pos(first + 0.25f * (cell % per_row), 0.3f + 0.25f * layer, first + 0.25f * (cell / per_row));
		Quat rot = Quat(u(rng), u(rng), u(rng), u(rng)).Normalized();
		BodyCreationSettings bcs(shapes[i % shapes.size()], RVec3(s * pos), rot, EMotionType::Dynamic, 1);
		bodies.push_back(bi.CreateAndAddBody(bcs, EActivation::Activate));
	}
	system.OptimizeBroadPhase();

	// The piles settle differently in both versions, so the cost is reported per active body per step
	int asleep_at = -1;
	uint64 active_body_steps = 0;
	auto start = std::chrono::high_resolution_clock::now();
	for (int step = 0; step < num_steps; ++step)
	{
		active_body_steps += system.GetNumActiveBodies(EBodyType::RigidBody);
		system.Update(1.0f / 60.0f, 1, &temp_allocator, &job_system);
		if (asleep_at < 0 && system.GetNumActiveBodies(EBodyType::RigidBody) == 0)
			asleep_at = step + 1;
	}
	auto end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(end - start).count();

	// Depth of the hull vertices below the floor, and a hash of the final state
	std::vector<float> depths;
	uint64 hash = HashBytes(nullptr, 0);
	for (BodyID id : bodies)
	{
		RMat44 com = bi.GetCenterOfMassTransform(id);
		const ConvexHullShape *hull = static_cast<const ConvexHullShape *>(bi.GetShape(id).GetPtr());
		float lowest = FLT_MAX;
		for (uint i = 0; i < hull->GetNumPoints(); ++i)
			lowest = min(lowest, float((com * hull->GetPoint(i)).GetY()));
		depths.push_back(-lowest);
		RVec3 p = bi.GetPosition(id);
		Quat q = bi.GetRotation(id);
		hash = HashBytes(&p, sizeof(p), hash);
		hash = HashBytes(&q, sizeof(q), hash);
	}
	std::sort(depths.begin(), depths.end());
	float threshold = slop + 0.005f * s;
	int deep = int(depths.end() - std::upper_bound(depths.begin(), depths.end(), threshold));
	printf("%s: bodies %d, hull points %.0f, %.1f steps/s, %.0f ns per active body step, asleep at step %d, depth below floor p99 %.1f mm max %.1f mm, deeper than %.0f mm: %d, hash 0x%016llx\n",
		small? "small" : "default", num_bodies, double(total_points) / shapes.size(), num_steps / seconds, 1.0e9 * seconds / double(active_body_steps), asleep_at,
		depths[size_t(0.99 * (depths.size() - 1))] * 1000, depths.back() * 1000, threshold * 1000, deep, (unsigned long long)hash);
	return 0;
}
