// Supporting face probe: for random convex hulls in random orientations, measure how far the deepest vertex
// lies beyond the face returned by ConvexHullShape::GetSupportingFace (the "miss"). A face that misses the
// deepest vertex makes the contact manifold miss it too, so the body rests on the wrong face and sinks.
#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/SubShapeID.h>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace JPH;

struct Stats { int samples = 0, misses = 0; float max_miss = 0; std::vector<float> all; };

static void Probe(const ConvexHullShape *shape, int rotations, std::mt19937 &rng, Stats &stats, float threshold)
{
	std::uniform_real_distribution<float> u(0, 1);
	for (int r = 0; r < rotations; ++r)
	{
		float u1 = u(rng), u2 = u(rng), u3 = u(rng);
		Quat q(sqrt(1 - u1) * sin(2 * JPH_PI * u2), sqrt(1 - u1) * cos(2 * JPH_PI * u2), sqrt(u1) * sin(2 * JPH_PI * u3), sqrt(u1) * cos(2 * JPH_PI * u3));
		Vec3 direction = q.Conjugated() * Vec3(0, 1, 0);
		Shape::SupportingFace face;
		shape->GetSupportingFace(SubShapeID(), direction, Vec3::sOne(), Mat44::sIdentity(), face);
		float deepest = -FLT_MAX;
		for (uint i = 0; i < shape->GetNumPoints(); ++i)
			deepest = max(deepest, shape->GetPoint(i).Dot(-direction));
		float face_best = -FLT_MAX;
		for (Vec3 v : face)
			face_best = max(face_best, v.Dot(-direction));
		float miss = deepest - face_best;
		stats.samples++;
		stats.all.push_back(miss);
		stats.max_miss = max(stats.max_miss, miss);
		if (miss > threshold)
			stats.misses++;
	}
}

static Ref<ConvexHullShape> Hull(const Array<Vec3> &points)
{
	ConvexHullShapeSettings settings(points, 0.005f);
	settings.mHullTolerance = 1.0e-3f;
	Shape::ShapeResult result = settings.Create();
	return result.IsValid()? static_cast<ConvexHullShape *>(result.Get().GetPtr()) : nullptr;
}

static void Report(const char *name, Stats &s, float threshold)
{
	std::sort(s.all.begin(), s.all.end());
	float p99 = s.all[size_t(0.99 * (s.all.size() - 1))];
	printf("%-28s samples %6d  miss > %.0f mm: %6.2f%%  p99 %6.2f mm  max %7.2f mm\n", name, s.samples, threshold * 1000, 100.0 * s.misses / s.samples, p99 * 1000, s.max_miss * 1000);
}

int main()
{
	RegisterDefaultAllocator();
	Factory::sInstance = new Factory();
	RegisterTypes();

	const float threshold = 0.001f;
	const int rotations = 1000;
	std::mt19937 rng(12345);
	std::uniform_real_distribution<float> u(-1, 1);

	// Square pyramid from PerformanceTest ConvexVsMeshScene, scaled to 0.2 m
	{
		Stats s;
		Ref<ConvexHullShape> shape = Hull({ Vec3(0, 0.2f, 0), Vec3(0.2f, 0, 0), Vec3(-0.2f, 0, 0), Vec3(0, 0, 0.2f), Vec3(0, 0, -0.2f) });
		Probe(shape, rotations, rng, s, threshold);
		Report("pyramid", s, threshold);
	}

	// Long box 0.3 x 0.06 x 0.1 m with a small chamfer at one end (a long sharp edge next to a small bevel face)
	{
		Stats s;
		Array<Vec3> points;
		for (float x : { -0.15f, 0.15f }) for (float y : { -0.03f, 0.03f }) for (float z : { -0.05f, 0.05f })
			if (x < 0 && y < 0)
			{
				points.push_back(Vec3(x + 0.006f, y, z));
				points.push_back(Vec3(x, y + 0.004f, z));
			}
			else
				points.push_back(Vec3(x, y, z));
		Ref<ConvexHullShape> shape = Hull(points);
		Probe(shape, rotations, rng, s, threshold);
		Report("long box, chamfered end", s, threshold);
	}

	// Random point clouds in a 0.3 x 0.1 x 0.15 m box (sharp corners and slivers)
	for (int count : { 12, 40, 150 })
	{
		Stats s;
		for (int h = 0; h < 100; ++h)
		{
			Array<Vec3> points;
			for (int i = 0; i < count; ++i)
				points.push_back(Vec3(0.15f * u(rng), 0.05f * u(rng), 0.075f * u(rng)));
			Ref<ConvexHullShape> shape = Hull(points);
			if (shape != nullptr)
				Probe(shape, rotations / 10, rng, s, threshold);
		}
		char name[64];
		snprintf(name, sizeof(name), "random cloud, %d points", count);
		Report(name, s, threshold);
	}

	// Points on an ellipsoid surface (smooth, finely tessellated hull)
	{
		Stats s;
		for (int h = 0; h < 20; ++h)
		{
			Array<Vec3> points;
			for (int i = 0; i < 200; ++i)
			{
				Vec3 v(u(rng), u(rng), u(rng));
				if (v.LengthSq() < 1.0e-6f)
					continue;
				v = v.Normalized();
				points.push_back(Vec3(0.15f * v.GetX(), 0.05f * v.GetY(), 0.075f * v.GetZ()));
			}
			Ref<ConvexHullShape> shape = Hull(points);
			if (shape != nullptr)
				Probe(shape, rotations / 10, rng, s, threshold);
		}
		Report("ellipsoid, 200 points", s, threshold);
	}

	// Cost of GetSupportingFace itself
	{
		Array<Vec3> points;
		for (int i = 0; i < 256; ++i)
		{
			Vec3 v(u(rng), u(rng), u(rng));
			points.push_back(0.1f * v.NormalizedOr(Vec3::sAxisX()));
		}
		Ref<ConvexHullShape> shape = Hull(points);
		std::vector<Vec3> directions;
		for (int i = 0; i < 4096; ++i)
			directions.push_back(Vec3(u(rng), u(rng), u(rng)).NormalizedOr(Vec3::sAxisY()));
		float sink = 0;
		const int calls = 2000000;
		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < calls; ++i)
		{
			Shape::SupportingFace face;
			shape->GetSupportingFace(SubShapeID(), directions[i & 4095], Vec3::sOne(), Mat44::sIdentity(), face);
			sink += face[0].GetX();
		}
		auto end = std::chrono::high_resolution_clock::now();
		printf("GetSupportingFace, hull with %u points, %u faces: %.1f ns per call (%g)\n", shape->GetNumPoints(), shape->GetNumFaces(), std::chrono::duration<double, std::nano>(end - start).count() / calls, sink);
	}
	return 0;
}
