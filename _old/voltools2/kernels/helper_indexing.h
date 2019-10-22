inline __device__
int get_z_idx(const int i, const int4* const dims) {
    return i / (dims[0].y * dims[0].z);
}

inline __device__
int get_y_idx(const int i, const int4* const dims) {
    return (i % (dims[0].y * dims[0].z)) / dims[0].z;
}

inline __device__
int get_x_idx(const int i, const int4* const dims) {
    return (i % (dims[0].y * dims[0].z)) % dims[0].z;
}