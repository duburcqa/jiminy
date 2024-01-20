#include "hpp/fcl/BVH/BVH_model.h"  // `hpp::fcl::CollisionGeometry`, `hpp::fcl::BVHModel`, `hpp::fcl::OBBRSS`
#include "hpp/fcl/shape/geometric_shapes.h"  // `hpp::fcl::Halfspace`
#include "jiminy/core/utilities/geometry.h"


namespace jiminy
{
    namespace internal
    {
        const double TOL = 1e-8;

        using Vector10d = Eigen::Matrix<double, 10, 1>;

        class Symmetric4
        {
        public:
            Symmetric4() = default;

            template<typename Derived>
            Symmetric4(const Eigen::MatrixBase<Derived> & data) :
            data_(data){EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 10)

            }

            Symmetric4(const Eigen::Vector3d & n, const double & d) :
            data_()
            {
                // Make plane a x + b y + c z + d = 0, i.e. Kp = p * p.T with p = [a, b, c, d]
                const double a = n[0], b = n[1], c = n[2];
                data_ << a * a, a * b, a * c, a * d, b * b, b * c, b * d, c * c, c * d, d * d;
            }

            static Symmetric4 Zero() { return Symmetric4(Vector10d::Zero()); }

            void setZero() { data_.setZero(); }

            const Vector10d & data() const { return data_; }

            Vector10d & data() { return data_; }

            Symmetric4 operator+(const Symmetric4 & other) const
            {
                return Symmetric4(data_ + other.data_);
            }

            Symmetric4 & operator+=(const Symmetric4 & other)
            {
                data_ += other.data_;
                return *this;
            }

            template<int i, int j>
            double coeff() const
            {
                constexpr int k = (j >= i) ? 4 * i - (i * (i + 1)) / 2 + j :
                                             4 * j - (j * (j + 1)) / 2 + i;
                return data_[k];
            }

            Eigen::Matrix4d matrix() const
            {
                // clang-format off
                return (Eigen::Matrix4d() <<
                    coeff<0, 0>(), coeff<0, 1>(), coeff<0, 2>(), coeff<0, 3>(),
                    coeff<1, 0>(), coeff<1, 1>(), coeff<1, 2>(), coeff<1, 3>(),
                    coeff<2, 0>(), coeff<2, 1>(), coeff<2, 2>(), coeff<2, 3>(),
                    coeff<3, 0>(), coeff<3, 1>(), coeff<3, 2>(), coeff<3, 3>()
                ).finished();
                // clang-format on
            }

            /// @brief Compute the minor defined as the determinant of the sub-matrix formed by
            ///        deleting row i and col j
            template<int i, int j>
            double cofactor() const
            {
                // clang-format off
                constexpr int i1 = (i + 1) % 4, i2 = (i + 2) % 4, i3 = (i + 3) % 4;
                constexpr int j1 = (j + 1) % 4, j2 = (j + 2) % 4, j3 = (j + 3) % 4;
                double minor =
                    coeff<i1, j1>() * (coeff<i2, j2>() * coeff<i3, j3>() - coeff<i2, j3>() * coeff<i3, j2>()) +
                    coeff<i1, j2>() * (coeff<i2, j3>() * coeff<i3, j1>() - coeff<i2, j1>() * coeff<i3, j3>()) +
                    coeff<i1, j3>() * (coeff<i2, j1>() * coeff<i3, j2>() - coeff<i2, j2>() * coeff<i3, j1>());
                if constexpr ((i + j) & 1)
                {
                    minor *= -1;
                }
                return minor;
                // clang-format on
            }

            double vtQv(const Eigen::Vector3d & v) const
            {
                // clang-format off
                // [v_x, v_y, v_z, 1].T * Q * [v_x, v_y, v_z, 1]
                const double x = v[0], y = v[1], z = v[2];
                return data_[0] * x * x + data_[4] * y * y + data_[7] * z * z + 2 * (
                       data_[1] * x * y + data_[2] * x * z + data_[5] * y * z +
                       data_[3] * x + data_[6] * y + data_[8] * z) + data_[9];
                // clang-format on
            }

        protected:
            Vector10d data_{};
        };

        struct Triangle
        {
            Vector3<std::size_t> v;
            std::array<double, 4> err;
            Eigen::Vector3d n;
            bool is_deleted;
            bool is_dirty;
        };

        struct Vertex
        {
            Eigen::Vector3d p;
            Symmetric4 q;
            bool is_border;
            std::size_t t_start;
            std::size_t t_count;
        };

        struct Ref
        {
            std::size_t t_id;
            std::size_t t_vertex;
        };

        /// @brief Fast mesh simplification utility.
        ///
        /// @details The original algorithm has been developed by Michael Garland and Paul
        ///          Heckbert. The technical details can be found in their paper, "Surface
        ///          Simplification Using Quadric Error Metrics.", 1997:
        ///          http://www.cs.cmu.edu/~garland/Papers/quadrics.pdf
        ///
        /// @sa The proposed implementation is based on code from Sven Forstmann released in 2014
        ///     under the MIT License:
        ///     https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification
        class MeshSimplifier
        {
        public:
            MeshSimplifier(const Matrix3X<double> & verts, const Matrix3X<Eigen::Index> & tris)
            {
                vertices.reserve(verts.cols());
                for (Eigen::Index i = 0; i < verts.cols(); ++i)
                {
                    Vertex v{};
                    v.p = verts.col(i);
                    vertices.push_back(std::move(v));
                }
                triangles.reserve(tris.cols());
                for (Eigen::Index i = 0; i < tris.cols(); ++i)
                {
                    Triangle t{};
                    t.v = tris.col(i).cast<std::size_t>();
                    triangles.push_back(std::move(t));
                }
            }

            /// @brief Compute error for one specific edge.
            std::pair<double, Eigen::Vector3d> computeError(std::size_t id_v1, std::size_t id_v2)
            {
                // Extract the relevant vertices
                const Vertex & v1 = vertices[id_v1];
                const Vertex & v2 = vertices[id_v2];

                // compute interpolated vertex
                const Symmetric4 q = v1.q + v2.q;
                if (!(v1.is_border && v2.is_border))
                {
                    const double det = q.cofactor<3, 3>();
                    if (det > TOL)
                    {
                        // q_delta is invertible
                        Eigen::Vector3d p;
                        p << q.cofactor<3, 0>(), q.cofactor<3, 1>(), q.cofactor<3, 2>();
                        p /= det;

                        // Compute resulting error
                        const double err = p[0] * q.coeff<3, 0>() + p[1] * q.coeff<3, 1>() +
                                           p[2] * q.coeff<3, 2>() + q.coeff<3, 3>();

                        return std::make_pair(err, std::move(p));
                    }
                }

                // det = 0 -> try to find best result
                const Eigen::Vector3d & p1 = v1.p;
                const double err1 = q.vtQv(p1);
                if (err1 < TOL)
                {
                    return {err1, p1};
                }
                const Eigen::Vector3d & p2 = v2.p;
                const double err2 = q.vtQv(p2);
                if (err2 < TOL)
                {
                    return {err2, p2};
                }
                const Eigen::Vector3d p3 = 0.5 * (p1 + p2);
                const double err3 = q.vtQv(p3);
                if (err3 < err2)
                {
                    if (err3 < err1)
                    {
                        return {err3, p3};
                    }
                    return {err1, p1};
                }
                if (err2 < err1)
                {
                    return {err2, p2};
                }
                return {err1, p1};
            }

            /// @brief Check if a triangle flips when this edge is removed.
            bool flipped(const Eigen::Vector3d & p,
                         std::size_t i1,
                         const Vertex & v0,
                         std::vector<bool> & is_deleted)
            {
                for (std::size_t k = 0; k < v0.t_count; ++k)
                {
                    const Triangle & t = triangles[refs[v0.t_start + k].t_id];
                    if (t.is_deleted)
                    {
                        continue;
                    }

                    const std::size_t s = refs[v0.t_start + k].t_vertex;
                    const std::size_t id1 = t.v[(s + 1) % 3];
                    const std::size_t id2 = t.v[(s + 2) % 3];
                    if (id1 == i1 || id2 == i1)  // delete ?
                    {
                        is_deleted[k] = true;
                        continue;
                    }
                    is_deleted[k] = false;

                    if ((vertices[id1].p - p).cross(vertices[id2].p - p).dot(t.n) < 0.0)
                    {
                        return true;
                    }
                }
                return false;
            }

            /// @brief Update triangle connections and edge error after a edge is collapsed.
            void collapseTriangles(const std::size_t & i0,
                                   const Vertex & v,
                                   std::vector<bool> & is_deleted,
                                   int64_t & num_deleted)
            {
                for (std::size_t k = 0; k < v.t_count; ++k)
                {
                    const Ref & r = refs[v.t_start + k];
                    Triangle & t = triangles[r.t_id];
                    if (t.is_deleted)
                    {
                        continue;
                    }
                    if (is_deleted[k])
                    {
                        t.is_deleted = true;
                        num_deleted++;
                        continue;
                    }
                    t.v[r.t_vertex] = i0;
                    t.is_dirty = true;
                    for (std::size_t j = 0; j < 3; ++j)
                    {
                        std::tie(t.err[j], std::ignore) = computeError(t.v[j], t.v[(j + 1) % 3]);
                    }
                    t.err[3] = std::min(t.err[0], std::min(t.err[1], t.err[2]));
                    refs.push_back(r);
                }
            }

            void update_mesh(int64_t iter)
            {
                // compact triangles, compute edge error and build reference list
                if (iter > 0)
                {
                    std::size_t dst = 0;
                    for (Triangle & t : triangles)
                    {
                        if (!t.is_deleted)
                        {
                            std::swap(triangles[dst++], t);
                        }
                    }
                    triangles.resize(dst);
                }

                // Init Reference ID list
                for (Vertex & v : vertices)
                {
                    v.t_start = 0;
                    v.t_count = 0;
                }
                for (const Triangle & t : triangles)
                {
                    for (const std::size_t j : t.v)
                    {
                        ++vertices[j].t_count;
                    }
                }
                std::size_t t_start = 0;
                for (Vertex & v : vertices)
                {
                    v.t_start = t_start;
                    t_start += v.t_count;
                    v.t_count = 0;
                }

                // Write References
                refs.resize(triangles.size() * 3);
                for (std::size_t i = 0; i < triangles.size(); ++i)
                {
                    const Triangle & t = triangles[i];
                    for (std::size_t j = 0; j < 3; ++j)
                    {
                        Vertex & v = vertices[t.v[j]];
                        Ref & ref = refs[v.t_start + v.t_count];
                        ref.t_id = i;
                        ref.t_vertex = j;
                        v.t_count++;
                    }
                }

                if (iter == 0)
                {
                    // Identify boundary : vertices[].is_border=0,1
                    for (Vertex & v : vertices)
                    {
                        v.is_border = false;
                    }
                    std::vector<std::pair<std::size_t, std::size_t>> v_info;  // id, count
                    for (Vertex & v : vertices)
                    {
                        v_info.clear();  // Defined outside and cleared inside to optimize cache
                        for (std::size_t j = 0; j < v.t_count; ++j)
                        {
                            const Triangle & t = triangles[refs[v.t_start + j].t_id];
                            for (const std::size_t id : t.v)
                            {
                                auto v_info_it = std::find_if(
                                    v_info.begin(),
                                    v_info.end(),
                                    [id](const std::pair<std::size_t, std::size_t> & item) -> bool
                                    { return item.first == id; });
                                if (v_info_it == v_info.end())
                                {
                                    v_info.emplace_back(id, 1);
                                    continue;
                                }
                                ++(v_info_it->second);
                            }
                        }
                        for (const auto & [id, count] : v_info)
                        {
                            if (count == 1)
                            {
                                vertices[id].is_border = true;
                            }
                        }
                    }

                    // Init Quadrics by Plane & Edge Errors.
                    // It is required at first iteration but not during the simplification.
                    // Yet, doing it anyway tends to improve the result for closed meshes.
                    for (Vertex & v : vertices)
                    {
                        v.q.setZero();
                    }

                    for (Triangle & t : triangles)
                    {
                        const Eigen::Vector3d & p0 = vertices[t.v[0]].p;
                        const Eigen::Vector3d & p1 = vertices[t.v[1]].p;
                        const Eigen::Vector3d & p2 = vertices[t.v[2]].p;
                        t.n = (p1 - p0).cross(p2 - p0).normalized();
                        const Symmetric4 plan(t.n, -t.n.dot(p0));
                        for (const std::size_t j : t.v)
                        {
                            vertices[j].q += plan;
                        }
                        for (std::size_t j = 0; j < 3; ++j)
                        {
                            Vertex & v0 = vertices[t.v[j]];
                            Vertex & v1 = vertices[t.v[(j + 1) % 3]];
                            if (v0.is_border && v1.is_border)
                            {
                                const Eigen::Vector3d n = t.n.cross(v1.p - v0.p).normalized();
                                const Symmetric4 border(n, -n.dot(v0.p));
                                v0.q += border;
                                v1.q += border;
                            }
                        }
                    }
                    for (Triangle & t : triangles)
                    {
                        // Calc Edge Error
                        for (std::size_t j = 0; j < 3; ++j)
                        {
                            std::tie(t.err[j], std::ignore) =
                                computeError(t.v[j], t.v[(j + 1) % 3]);
                        }
                        t.err[3] = std::min(t.err[0], std::min(t.err[1], t.err[2]));
                    }
                }
            }

            /// @brief Main simplification function.
            void simplify(std::size_t mesh_update_rate = 5,
                          double aggressiveness = 7,
                          int64_t max_iter = 100,
                          double alpha = 1e-9,
                          int64_t K = 3,
                          bool preserve_border = false,
                          bool verbose = false)
            {
                // init
                for (Triangle & t : triangles)
                {
                    t.is_deleted = false;
                }

                // main iteration loop
                int64_t num_deleted = 0;
                std::vector<bool> deleted0;
                std::vector<bool> deleted1;
                std::size_t triangle_count = triangles.size();
                for (int64_t iter = 0; iter < max_iter; iter++)
                {
                    // update mesh once in a while
                    if (iter % mesh_update_rate == 0)
                    {
                        update_mesh(iter);
                    }

                    // clear is_dirty flag
                    for (Triangle & t : triangles)
                    {
                        t.is_dirty = false;
                    }

                    // All triangles with edges below the threshold will be removed.
                    // The following numbers works well for most models.
                    // If it does not, try to adjust the 3 parameters.
                    double threshold =
                        alpha * std::pow(static_cast<double>(iter + K), aggressiveness);

                    // target number of triangles reached ? Then break
                    if (verbose)
                    {
                        std::cout << "iteration " << iter << " - triangles "
                                  << triangle_count - num_deleted << " - threshold " << threshold
                                  << std::endl;
                    }

                    // remove vertices & mark is_deleted triangles
                    for (Triangle & t : triangles)
                    {
                        if (t.err[3] > threshold || t.is_deleted || t.is_dirty)
                        {
                            continue;
                        }

                        for (std::size_t j = 0; j < 3; ++j)
                        {
                            if (t.err[j] < threshold)
                            {
                                const std::size_t i0 = t.v[j];
                                const std::size_t i1 = t.v[(j + 1) % 3];
                                Vertex & v0 = vertices[i0];
                                const Vertex & v1 = vertices[i1];

                                // Border check
                                if (v0.is_border != v1.is_border)
                                {
                                    // base behaviour
                                    continue;
                                }
                                if (preserve_border && v0.is_border && v1.is_border &&
                                    t.err[j] > TOL)
                                {
                                    continue;
                                }

                                // Compute vertex to collapse to
                                Eigen::Vector3d p;
                                std::tie(std::ignore, p) = computeError(i0, i1);
                                deleted0.resize(v0.t_count);  // normals temporarily
                                deleted1.resize(v1.t_count);  // normals temporarily
                                // don't remove if flipped
                                if (flipped(p, i1, v0, deleted0))
                                {
                                    continue;
                                }
                                if (flipped(p, i0, v1, deleted1))
                                {
                                    continue;
                                }

                                // not flipped, so remove edge
                                v0.p = p;
                                v0.q += v1.q;
                                const std::size_t t_start = refs.size();
                                collapseTriangles(i0, v0, deleted0, num_deleted);
                                collapseTriangles(i0, v1, deleted1, num_deleted);
                                const std::size_t t_count = refs.size() - t_start;
                                if (t_count <= v0.t_count)
                                {
                                    if (t_count)
                                    {
                                        // save ram
                                        std::move(std::next(refs.begin(), t_start),
                                                  std::next(refs.begin(), t_start + t_count),
                                                  std::next(refs.begin(), v0.t_start));
                                    }
                                }
                                else
                                {
                                    // append
                                    v0.t_start = t_start;
                                }
                                v0.t_count = t_count;
                                break;
                            }
                        }
                    }
                }

                if (verbose)
                {
                    std::cout << "done: triangles " << triangle_count - num_deleted << std::endl;
                }

                // Finally compact mesh before exiting
                for (Vertex & v : vertices)
                {
                    v.t_count = 0;
                }
                std::size_t dst = 0;
                for (Triangle & t : triangles)
                {
                    if (!t.is_deleted)
                    {
                        for (const std::size_t j : t.v)
                        {
                            vertices[j].t_count = 1;
                        }
                        std::swap(triangles[dst++], t);
                    }
                }
                triangles.resize(dst);
                dst = 0;
                for (Vertex & v : vertices)
                {
                    if (v.t_count)
                    {
                        v.t_start = dst;
                        std::swap(vertices[dst++].p, v.p);
                    }
                }
                vertices.resize(dst);
                for (Triangle & t : triangles)
                {
                    for (std::size_t & j : t.v)
                    {
                        j = vertices[j].t_start;
                    }
                }
            }

        public:
            std::vector<Triangle> triangles{};
            std::vector<Vertex> vertices{};
            std::vector<Ref> refs{};
        };
    }

    hpp::fcl::CollisionGeometryPtr_t discretizeHeightmap(const HeightmapFunctor & heightmap,
                                                         double x_min,
                                                         double x_max,
                                                         double x_unit,
                                                         double y_min,
                                                         double y_max,
                                                         double y_unit,
                                                         bool must_simplify)
    {
        // Allocate vertices on a regular grid
        const Eigen::Index x_dim =
            static_cast<Eigen::Index>(std::ceil((x_max - x_min) / x_unit)) + 1;
        const Eigen::Index y_dim =
            static_cast<Eigen::Index>(std::ceil((y_max - y_min) / y_unit)) + 1;
        Matrix3X<double> vertices(3, x_dim * y_dim);

        // Fill x and y query coordinates over the grid
        Eigen::Map<Eigen::MatrixXd>(vertices.row(0).data(), x_dim, y_dim).rowwise() =
            Eigen::VectorXd::LinSpaced(x_dim, x_min, x_max).transpose().eval();
        Eigen::Map<Eigen::MatrixXd>(vertices.row(1).data(), x_dim, y_dim).colwise() =
            Eigen::VectorXd::LinSpaced(y_dim, y_min, y_max).eval();

        // Evaluate z coordinate over the grid
        for (Eigen::Index i = 0; i < vertices.cols(); ++i)
        {
            auto vertex = vertices.col(i);
            Eigen::Vector3d normal;
            heightmap(vertex.head<2>(), vertex[2], normal);
        }

        // Check if the heightmap is flat
        if (((vertices.row(2).array() - vertices(2, 0)).abs() < EPS).all())
        {
            return hpp::fcl::CollisionGeometryPtr_t(
                new hpp::fcl::Halfspace(Eigen::Vector3d::UnitZ(), vertices(2, 0)));
        }

        // Compute the face indices
        Matrix3X<Eigen::Index> triangles(3, 2 * (x_dim - 1) * (y_dim - 1));
        Eigen::Index tri_index = 0;
        for (Eigen::Index i = 0; i < x_dim - 1; ++i)
        {
            for (Eigen::Index j = 0; j < y_dim - 1; ++j)
            {
                const Eigen::Index k = j * x_dim + i;
                triangles.middleCols<2>(tri_index) << k, k + 1, k + x_dim, k + 1, k + 1 + x_dim,
                    k + x_dim;
                tri_index += 2;
            }
        }

        // Simplify the mesh if requested
        if (must_simplify)
        {
            // The border must be preserved to avoid changing the boundary of the surface
            internal::MeshSimplifier mesh_simplifier(vertices, triangles);
            mesh_simplifier.simplify(4, 4, 21, 3.0e-9, 3, true, true);
            vertices.resize(Eigen::NoChange, mesh_simplifier.vertices.size());
            for (Eigen::Index i = 0; i < vertices.cols(); ++i)
            {
                vertices.col(i) = mesh_simplifier.vertices[i].p;
            }
            triangles.resize(Eigen::NoChange, mesh_simplifier.triangles.size());
            for (Eigen::Index i = 0; i < triangles.cols(); ++i)
            {
                triangles.col(i) = mesh_simplifier.triangles[i].v.cast<Eigen::Index>();
            }
        }

        /* Wrap the vertices and triangles in a geometry object.
           Do not use `addVertices` and `addTriangles`to avoid extra copy. */
        hpp::fcl::BVHModelPtr_t mesh_ptr(new hpp::fcl::BVHModel<hpp::fcl::OBBRSS>);
        mesh_ptr->beginModel();
        mesh_ptr->addVertices(vertices.transpose());    // Beware it performs a copy
        mesh_ptr->addTriangles(triangles.transpose());  // Beware it performs a copy
        mesh_ptr->endModel();
        mesh_ptr->computeLocalAABB();
        return mesh_ptr;
    }

    HeightmapFunctor sumHeightmaps(std::vector<HeightmapFunctor> heightmaps)
    {
        if (heightmaps.size() == 1)
        {
            return heightmaps[0];
        }
        return [heightmaps](
                   const Eigen::Vector2d & pos, double & height, Eigen::Vector3d & normal) -> void
        {
            thread_local static double height_i;
            thread_local static Eigen::Vector3d normal_i;

            height = 0.0;
            normal.setZero();
            for (HeightmapFunctor const & heightmap : heightmaps)
            {
                heightmap(pos, height_i, normal_i);
                height += height_i;
                normal += normal_i;
            }
            normal.normalize();
        };
    }

    HeightmapFunctor mergeHeightmaps(std::vector<HeightmapFunctor> heightmaps)
    {
        if (heightmaps.size() == 1)
        {
            return heightmaps[0];
        }
        return [heightmaps](
                   const Eigen::Vector2d & pos, double & height, Eigen::Vector3d & normal) -> void
        {
            thread_local static double height_i;
            thread_local static Eigen::Vector3d normal_i;

            height = -INF;
            bool is_dirty = false;
            for (HeightmapFunctor const & heightmap : heightmaps)
            {
                heightmap(pos, height_i, normal_i);
                if (std::abs(height_i - height) < EPS)
                {
                    normal += normal_i;
                    is_dirty = true;
                }
                else if (height_i > height)
                {
                    height = height_i;
                    normal = normal_i;
                    is_dirty = false;
                }
            }
            if (is_dirty)
            {
                normal.normalize();
            }
        };
    }
}