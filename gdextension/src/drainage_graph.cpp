#include "drainage_graph.h"

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/json.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include "cubesphere_math.h"
#include "npy.h"

using namespace godot;

namespace globe {

static const int D8_OFFSETS[8][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};

void DrainageGraph::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load", "world_dir"), &DrainageGraph::load);
    ClassDB::bind_method(D_METHOD("is_loaded"), &DrainageGraph::is_loaded);
    ClassDB::bind_method(D_METHOD("grid_size"), &DrainageGraph::grid_size);
    ClassDB::bind_method(D_METHOD("downstream", "cell"), &DrainageGraph::downstream);
    ClassDB::bind_method(D_METHOD("downstream_path", "cell", "max_steps"), &DrainageGraph::downstream_path);
    ClassDB::bind_method(D_METHOD("basin_of", "face", "u", "v"), &DrainageGraph::basin_of);
    ClassDB::bind_method(D_METHOD("basin_of_cell", "cell"), &DrainageGraph::basin_of_cell);
    ClassDB::bind_method(D_METHOD("basin", "id"), &DrainageGraph::basin);
    ClassDB::bind_method(D_METHOD("outlet", "basin_id"), &DrainageGraph::outlet);
    ClassDB::bind_method(D_METHOD("nearest_reach", "p", "radius"), &DrainageGraph::nearest_reach);
    ClassDB::bind_method(D_METHOD("edge", "id"), &DrainageGraph::edge);
    ClassDB::bind_method(D_METHOD("edge_count"), &DrainageGraph::edge_count);
    ClassDB::bind_method(D_METHOD("get_nodes"), &DrainageGraph::get_nodes);
    ClassDB::bind_method(D_METHOD("cell_of", "p"), &DrainageGraph::cell_of);
    ClassDB::bind_method(D_METHOD("cell_center", "cell"), &DrainageGraph::cell_center);
}

static Variant parse_json_file(const String &path) {
    if (!FileAccess::file_exists(path)) return Variant();
    return JSON::parse_string(FileAccess::get_file_as_string(path));
}

bool DrainageGraph::load(const String &world_dir) {
    loaded = false;
    Variant man = parse_json_file(world_dir + String("/manifest.json"));
    if (man.get_type() != Variant::DICTIONARY) {
        UtilityFunctions::push_error(String("DrainageGraph: no manifest at ") + world_dir);
        return false;
    }
    N = int(Dictionary(man).get("N_c", 0));
    if (N <= 0) return false;
    basin_id.assign(size_t(6) * N * N, -1);
    flow_dir.assign(size_t(6) * N * N, 255);
    for (int f = 0; f < 6; ++f) {
        NpyArray b = read_npy(world_dir + String("/coarse/basin_id.f") + String::num_int64(f) + String(".npy"));
        if (b.ok && b.count() == int64_t(N) * N && b.dtype == "<i4") memcpy(basin_id.data() + size_t(f) * N * N, b.data.data(), size_t(N) * N * 4);
        NpyArray d = read_npy(world_dir + String("/coarse/flow_dir.f") + String::num_int64(f) + String(".npy"));
        if (d.ok && d.count() == int64_t(N) * N && d.dtype == "|u1") memcpy(flow_dir.data() + size_t(f) * N * N, d.data.data(), size_t(N) * N);
    }
    edges.clear();
    cell_to_edge.clear();
    nodes = Array();
    basins = Dictionary();
    Variant g = parse_json_file(world_dir + String("/graph/drainage.json"));
    if (g.get_type() == Variant::DICTIONARY) {
        Dictionary gd = g;
        nodes = gd.get("nodes", Array());
        Array es = gd.get("edges", Array());
        for (int k = 0; k < es.size(); ++k) {
            Dictionary e = es[k];
            Edge E;
            E.id = int(e.get("id", k));
            E.from = int(e.get("from", -1));
            E.to = int(e.get("to", -1));
            E.order = int(e.get("order", 1));
            E.length_m = double(e.get("length_m", 0.0));
            E.mean_discharge = double(e.get("mean_discharge", 0.0));
            Array cells = e.get("cells", Array());
            for (int c = 0; c < cells.size(); ++c) {
                Array cc = cells[c];
                if (cc.size() < 3) continue;
                int32_t id = flat(int(cc[0]), int(cc[1]), int(cc[2]));
                E.cells.push_back(id);
                cell_to_edge[id] = int(edges.size());
            }
            edges.push_back(E);
        }
    }
    Variant b = parse_json_file(world_dir + String("/graph/basins.json"));
    if (b.get_type() == Variant::DICTIONARY) {
        Array bs = Dictionary(b).get("basins", Array());
        for (int k = 0; k < bs.size(); ++k) {
            Dictionary bd = bs[k];
            basins[int(bd.get("id", k))] = bd;
        }
    }
    loaded = true;
    return true;
}

Vector3i DrainageGraph::cell_of(const Vector3 &p) const {
    double u, v;
    int face = globe::from_sphere(double(p.x), double(p.y), double(p.z), u, v);
    int i = std::min(int(std::floor(u * N)), N - 1), j = std::min(int(std::floor(v * N)), N - 1);
    return Vector3i(face, std::max(i, 0), std::max(j, 0));
}

Vector3 DrainageGraph::cell_center(const Vector3i &cell) const {
    Vec3 p = globe::to_sphere(cell.x, (cell.y + 0.5) / N, (cell.z + 0.5) / N);
    return Vector3(real_t(p.x), real_t(p.y), real_t(p.z));
}

Vector3i DrainageGraph::downstream(const Vector3i &cell) const {
    if (!loaded || cell.x < 0 || cell.x >= 6 || cell.y < 0 || cell.y >= N || cell.z < 0 || cell.z >= N) return Vector3i(-1, -1, -1);
    uint8_t code = flow_dir[size_t(flat(cell.x, cell.y, cell.z))];
    if (code > 7) return Vector3i(-1, -1, -1);
    int i = cell.y + D8_OFFSETS[code][0], j = cell.z + D8_OFFSETS[code][1];
    if (i >= 0 && i < N && j >= 0 && j < N) return Vector3i(cell.x, i, j);
    // crossed a face edge: map the extended cell centre through the sphere
    Vec3 p = globe::to_sphere(cell.x, (i + 0.5) / N, (j + 0.5) / N);
    double u, v;
    int f2 = globe::from_sphere(p.x, p.y, p.z, u, v);
    int i2 = std::min(std::max(int(std::floor(u * N)), 0), N - 1), j2 = std::min(std::max(int(std::floor(v * N)), 0), N - 1);
    return Vector3i(f2, i2, j2);
}

Array DrainageGraph::downstream_path(const Vector3i &cell, int max_steps) const {
    Array out;
    Vector3i c = cell;
    for (int s = 0; s < max_steps && c.x >= 0; ++s) {
        out.push_back(c);
        c = downstream(c);
    }
    return out;
}

int DrainageGraph::basin_of_cell(const Vector3i &cell) const {
    if (!loaded || cell.x < 0 || cell.x >= 6 || cell.y < 0 || cell.y >= N || cell.z < 0 || cell.z >= N) return -1;
    return basin_id[size_t(flat(cell.x, cell.y, cell.z))];
}

int DrainageGraph::basin_of(int face, double u, double v) const {
    int i = std::min(std::max(int(std::floor(u * N)), 0), N - 1), j = std::min(std::max(int(std::floor(v * N)), 0), N - 1);
    return basin_of_cell(Vector3i(face, i, j));
}

Dictionary DrainageGraph::basin(int id) const {
    if (basins.has(id)) return basins[id];
    return Dictionary();
}

Dictionary DrainageGraph::outlet(int basin_id_) const {
    Dictionary out;
    Dictionary b = basin(basin_id_);
    if (b.is_empty()) return out;
    Variant ov = b.get("outlet", Array());
    Array o = ov;
    if (o.size() >= 3) {
        Vector3i c = Vector3i(int(o[0]), int(o[1]), int(o[2]));
        out["cell"] = c;
        out["position"] = cell_center(c);
    }
    out["downstream_basin"] = b.get("downstream_basin", -1);
    out["order"] = b.get("order", 0);
    return out;
}

Dictionary DrainageGraph::edge(int id) const {
    Dictionary out;
    if (id < 0 || id >= int(edges.size())) return out;
    const Edge &E = edges[size_t(id)];
    out["id"] = E.id;
    out["from"] = E.from;
    out["to"] = E.to;
    out["order"] = E.order;
    out["length_m"] = E.length_m;
    out["mean_discharge"] = E.mean_discharge;
    Array cells;
    for (int32_t c : E.cells) cells.push_back(unflat(c));
    out["cells"] = cells;
    return out;
}

Dictionary DrainageGraph::nearest_reach(const Vector3 &p, int radius) const {
    Dictionary out;
    if (!loaded) return out;
    Vector3i c = cell_of(p);
    int best = -1;
    double best_d = 1e30;
    Vector3i best_cell;
    for (int di = -radius; di <= radius; ++di) {
        for (int dj = -radius; dj <= radius; ++dj) {
            int i = c.y + di, j = c.z + dj, f = c.x;
            if (i < 0 || i >= N || j < 0 || j >= N) {
                Vec3 q = globe::to_sphere(f, (i + 0.5) / N, (j + 0.5) / N);
                double u, v;
                f = globe::from_sphere(q.x, q.y, q.z, u, v);
                i = std::min(std::max(int(std::floor(u * N)), 0), N - 1);
                j = std::min(std::max(int(std::floor(v * N)), 0), N - 1);
            }
            auto it = cell_to_edge.find(flat(f, i, j));
            if (it == cell_to_edge.end()) continue;
            Vec3 q = globe::to_sphere(f, (i + 0.5) / N, (j + 0.5) / N);
            double d = std::acos(std::min(1.0, std::max(-1.0, q.x * p.x + q.y * p.y + q.z * p.z)));
            if (d < best_d) {
                best_d = d;
                best = it->second;
                best_cell = Vector3i(f, i, j);
            }
        }
    }
    if (best < 0) return out;
    const Edge &E = edges[size_t(best)];
    out["edge_id"] = E.id;
    out["order"] = E.order;
    out["mean_discharge"] = E.mean_discharge;
    out["cell"] = best_cell;
    out["distance_cells"] = best_d / (HALF_PI / N);
    return out;
}

}  // namespace globe
