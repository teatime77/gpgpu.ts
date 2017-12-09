﻿var Shaders = {};

Shaders["AttribTest"] = `
precision highp sampler3D;

in    float    w;
in vec4      x;
in vec4 y;

uniform float biases[4];
uniform float f;
uniform vec4 ff;
uniform sampler3D tt;

out vec4 z;

void main() {
    uint idx = uint(w) % 4u;

    vec4  txl = texelFetch(tt, ivec3(2, 3, 1), 0);
    z = (x +y +ff +txl) +biases[idx]+f;
}`;


Shaders["Test"] = `
precision highp sampler3D;

in float idx_f;

uniform float biases[4];

uniform sampler3D prev_activation;

out float z;
out float activation;

void main() {
    uint idx = uint(idx_f);

    uint Z = idx / uint(4 * 3 * 28);
    idx %= uint(4 * 3 * 28);

    uint y = idx / uint(4 * 3);
    idx %= uint(4 * 3);

    uint x = idx / uint(4);
    idx %= uint(4);

    vec4  txl = texelFetch(prev_activation, ivec3(x, y, Z), 0);

    z = idx_f;
    activation = txl[idx] + biases[idx];
}`;


Shaders["ConvolutionalLayer-forward"] = `
precision highp sampler3D;

uniform float weights[featureCount * filterSize * filterSize];
uniform float biases[featureCount];

uniform sampler3D prev_activation;

in float idx_f;

out vec4 z;
out vec4 activation;

void main() {
    uint idx = uint(idx_f);

    uint feature_idx  = idx / (batchVec4Count * colCount * rowCount);
    idx -= feature_idx * (batchVec4Count * colCount * rowCount);

    uint r1 = idx / (batchVec4Count * colCount);
    idx -= r1 * (batchVec4Count * colCount);

    uint c1 = idx / batchVec4Count;
    uint batch_vec4_idx = idx - c1 * batchVec4Count;

    uint r2, c2;
    vec4 sum = vec4(0.0);
    int err_flg = 0;
    for (r2 = 0u; r2 < filterSize; r2++) {

        for (c2 = 0u; c2 < filterSize; c2++) {

            uint c3 = c1 + c2;
            uint r3 = r1 + r2;

/*
            uint ii = uint(idx_f);
            uint r3 = ii / uint(3 * 28);
            ii %= uint(3 * 28);

            uint c3 = ii / uint(3);
            batch_vec4_idx = ii % uint(3);
*/

            if(batch_vec4_idx < batchVec4Count && c3 < colCount + filterSize - 1u  && r3 < rowCount + filterSize - 1u) {

                vec4  txl = texelFetch(prev_activation, ivec3(batch_vec4_idx, c3, r3), 0);

                uint weight_idx = (feature_idx * filterSize + r2) * filterSize + c2;
                sum += txl * weights[weight_idx];
            }
            else {

                err_flg = 1;
                break;
            }
        }
    }

    if(err_flg == 0) {

        z = sum +biases[feature_idx];
        activation = 1.0 / (1.0 +exp(-z));
    }
    else {

        z          = vec4(12345.0);
        activation = vec4(12345.0);
    }

}`;


Shaders["ConvolutionalLayer-backward"] = `
precision highp sampler3D;

uniform float weights[featureCount * filterSize * filterSize];

uniform sampler3D prev_activation;
uniform sampler3D delta_z;

in float idx_f;

out float nablaWeights;

void main() {
    uint idx = uint(idx_f);

    uint feature_idx  = idx / (filterSize * filterSize);
    idx -= feature_idx * (filterSize * filterSize);

    uint r2 = idx / filterSize;
    uint c2 = idx -r2 * filterSize;

    vec4 sum = vec4(0.0);

    uint r1;
    for (r1 = 0u; r1 < rowCount; r1++) {

        uint c1;
        for (c1 = 0u; c1 < colCount; c1++) {

            uint c3 = c1 + c2;
            uint r3 = r1 + r2;

            uint batch_vec4_idx;
            for(batch_vec4_idx = 0u; batch_vec4_idx < batchVec4Count; batch_vec4_idx++) {

                vec4  delta  = texelFetch(delta_z, ivec3(batch_vec4_idx + c1 * batchVec4Count, r1, feature_idx), 0);
                vec4  prev_a = texelFetch(prev_activation, ivec3(batch_vec4_idx, c3, r3), 0);

                sum += delta * prev_a;
            }
        }
    }

    nablaWeights = sum.x +sum.y +sum.z +sum.w;
}`;


Shaders["vs-Texture"] = `
uniform int B_Cols;

uniform sampler2D A_Tex;
uniform sampler2D B_Tex;

in float idx_f;

out float dot_val;

void main() {
    uint idx = uint(idx_f);
    int i   = int(idx / uint(B_Cols));
    int j   = int(idx % uint(B_Cols));

    int k;
    float sum = 0.0;
    for(k = 0; k < _repeat_; k++) {
        vec4  A_txl, B_txl;

        A_txl = texelFetch(A_Tex, ivec2(k, i), 0);
        B_txl = texelFetch(B_Tex, ivec2(k, j), 0);
        sum   += dot(A_txl, B_txl);
    }

    dot_val = sum;
}`;

Shaders["vs-Uniform"] =
`
uniform int B_Cols;

uniform vec4 A[_A_len_];
uniform vec4 B[_B_len_];

in float idx_f;

out float dot_val;

void main() {
    uint idx = uint(idx_f);
    int i   = int(idx / uint(B_Cols));
    int j   = int(idx % uint(B_Cols));

    int k;
    float sum = 0.0;
    for(k = 0; k < _repeat_; k++) {
        sum += dot(A[_repeat_*i +k], B[_repeat_*j +k]);
    }
    dot_val = sum;
}`;