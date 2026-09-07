#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/vector3i.hpp>

namespace globe {

// Loads graph/drainage.json, graph/basins.json and the coarse basin_id /
// flow_dir rasters; answers downstream / basin / nearest-reach queries
// (PLAN 12.1).  Cells are Vector3i(face, i, j) on the coarse grid.
class DrainageGraph : public godot::RefCounted {
    GDCLASS(DrainageGraph, godot::RefCounted)

    struct Edge {
        int id = -1, from = -1, to = -1, order = 0;
        double length_m = 0.0, mean_discharge = 0.0;
        std::vector<int32_t> cells;  // flat coarse cell ids
    };

    int N = 0;
    std::vector<int32_t> basin_id;   // 6*N*N, -1 ocean
    std::vector<uint8_t> flow_dir;   // 6*N*N, 255 = ocean
    std::vector<Edge> edges;
    godot::Array nodes;
    godot::Dictionary basins;        // id -> Dictionary
    std::unordered_map<int32_t, int> cell_to_edge;
    bool loaded = false;

    int32_t flat(int f, int i, int j) const { return int32_t((f * N + i) * N + j); }
    godot::Vector3i unflat(int32_t c) const {
        int f = c / (N * N), r = c % (N * N);
        return godot::Vector3i(f, r / N, r % N);
    }

protected:
    static void _bind_methods();

public:
    bool load(const godot::String &world_dir);
    bool is_loaded() const { return loaded; }
    int grid_size() const { return N; }
    // downstream cell of (face, i, j); (-1,-1,-1) when it flows into the ocean / off the grid
    godot::Vector3i downstream(const godot::Vector3i &cell) const;
    // cells from `cell` downstream until the ocean (inclusive of `cell`), at most max_steps
    godot::Array downstream_path(const godot::Vector3i &cell, int max_steps) const;
    int basin_of(int face, double u, double v) const;
    int basin_of_cell(const godot::Vector3i &cell) const;
    godot::Dictionary basin(int id) const;
    godot::Dictionary outlet(int basin_id) const;
    // nearest channel reach within `radius` coarse cells of the unit vector p
    godot::Dictionary nearest_reach(const godot::Vector3 &p, int radius) const;
    godot::Dictionary edge(int id) const;
    int edge_count() const { return int(edges.size()); }
    godot::Array get_nodes() const { return nodes; }
    godot::Vector3i cell_of(const godot::Vector3 &p) const;
    godot::Vector3 cell_center(const godot::Vector3i &cell) const;
};

}  // namespace globe
