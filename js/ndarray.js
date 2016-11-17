﻿class Mat {
    constructor(rows, cols, init, column_major, depth) {
        this.Rows = rows;
        this.Cols = cols;
        this.Depth = (depth == undefined ? 1 : depth);
        this.shape = [rows, cols];
        this.columnMajor = (column_major == undefined ? false : column_major);

        if (init) {

            Assert(init instanceof Float32Array && init.length == rows * cols * this.Depth, "Mat-init");
            this.dt = init;
        }
        else {

            this.dt = new Float32Array(rows * cols);
            /*
            for (var r = 0; r < rows; r++) {
                for (var c = 0; c < cols; c++) {
                    //                            this.dt[r * cols + c] = r * 1000 + c;
                    this.dt[r * cols + c] = Math.random();
                }
            }
            */
        }
    }

    map(f) {
        return new Mat(this.Rows, this.Cols, this.dt.map(f), this.columnMajor);
    }

    T() {
        var v = new Float32Array(this.Cols * this.Rows);
        for (var r = 0; r < this.Cols; r++) {
            for (var c = 0; c < this.Rows; c++) {
                v[r * this.Rows + c] = this.dt[c * this.Cols + r];
            }
        }

        return new Mat(this.Cols, this.Rows, v);
    }

    transpose() {
        return this.T();
    }

    At(r, c) {
        return this.dt[r * this.Cols + c];
    }

    Set(r, c, val) {
        this.dt[r * this.Cols + c] = val;
    }

    Col(c) {
        var v = new Float32Array(this.Rows);
        for (var r = 0; r < this.Rows; r++) {
            v[r] = this.dt[r * this.Cols + c];
        }

        return new Mat(this.Rows, 1, v);
    }

    Add(m) {
        Assert(m instanceof Mat && m.Rows == this.Rows && m.Cols == this.Cols, "Mat-add");
        var v = new Float32Array(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = this.dt[k] + m.dt[k];
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    AddV(m) {
        Assert(m instanceof Mat && m.Rows == this.Rows && m.Cols == 1, "Mat-add-V");
        var v = new Float32Array(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = this.dt[k] + m.dt[r];
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    SubV(m) {
        Assert(m instanceof Mat && m.Rows == this.Rows && m.Cols == 1, "Mat-sub-V");
        var v = new Float32Array(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = this.dt[k] - m.dt[r];
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    reduce(f) {
        var v = new Float32Array(this.Rows);
        for (var r = 0; r < this.Rows; r++) {
            var x;
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                if (c == 0) {

                    x = this.dt[k];
                }
                else {

                    x = f(x, this.dt[k]);
                }
            }
            v[r] = x;
        }

        return new Mat(this.Rows, 1, v);
    }

    Sub(m) {
        Assert(m instanceof Mat && m.Rows == this.Rows && m.Cols == this.Cols, "Mat-Sub");
        var v = new Float32Array(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = this.dt[k] - m.dt[k];
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    Mul(m) {
        if (m instanceof Number) {

            return new Mat(this.Rows, this.Cols, this.dt.map(x => x * m));
        }
        Assert(m instanceof Mat && m.Rows == this.Rows && m.Cols == this.Cols && m.columnMajor == this.columnMajor, "Mat-Mul");
        var v = new Float32Array(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = this.dt[k] * m.dt[k];
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    Abs() {
        var v = new Float32Array(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = Math.abs(this.dt[k]);
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    Sum() {
        var sum = 0;
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                sum += this.dt[r * this.Cols + c];
            }
        }

        return sum;
    }

    Dot(m) {
        Assert(m instanceof Mat && m.Rows == this.Cols, "Mat-Dot");

        var v = new Float32Array(this.Rows * m.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < m.Cols; c++) {
                var sum = 0;
                for (var k = 0; k < this.Cols; k++) {
                    sum += this.dt[r * this.Cols + k] * m.dt[k * m.Cols + c];
                }
                v[r * m.Cols + c] = sum;
            }
        }
        return new Mat(this.Rows, m.Cols, v);
    }

    toString() {
        var s = "[";
        for (var r = 0; r < this.Rows; r++) {
            if (r == 0) {

                s = s + " [";
            }
            else {

                s = s + "\r\n, [";
            }

            for (var c = 0; c < this.Cols; c++) {
                if (c != 0) {

                    s = s + ", ";
                }

                s = s + this.dt[r * this.Cols + c].toFixed(7);
            }

            s = s + "]";
        }

        s = s + " ]";

        return s;
    }

    MakeProgram(gl, vshaderTransform, fshaderTransform, varyings) {
        var prg = gl.createProgram();
        gl.attachShader(prg, vshaderTransform);
        gl.attachShader(prg, fshaderTransform);

        gl.transformFeedbackVaryings(prg, varyings, gl.SEPARATE_ATTRIBS);   // gl.INTERLEAVED_ATTRIBS 
        gl.linkProgram(prg);

        // check
        var msg = gl.getProgramInfoLog(prg);
        if (msg) {
            console.log(msg);
        }

        msg = gl.getShaderInfoLog(vshaderTransform);
        if (msg) {
            console.log(msg);
        }

        gl.deleteShader(vshaderTransform);
        gl.deleteShader(fshaderTransform);

        return prg;
    }

    MakeFloat32Array(n) {
        var v = new Float32Array(n);

        for (var k = 0; k < n; k++) {
            v[k] = k;
        }

        return v;
    }

    MakeIdxBuffer(gl, gpu, element_count) {
        var idx_buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, idx_buffer);
        gpu.vidx = this.MakeFloat32Array(element_count);
        for (var i = 0; i < element_count; i++) {
            gpu.vidx[i] = i;
        }
        gl.bufferData(gl.ARRAY_BUFFER, gpu.vidx, gl.STATIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);

        return idx_buffer;
    }

    MakeShader(gl, type, source) {
        var shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);

        return shader;
    }

    MakeTex(gl, tex_id, dim) {
        var texture = gl.createTexture();

        gl.activeTexture(tex_id);
        gl.bindTexture(dim, texture);

        gl.texParameteri(dim, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(dim, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(dim, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(dim, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        return texture;
    }

    SetTex(gl, m, tex_id, dim, texture) {
        gl.activeTexture(tex_id);
        gl.bindTexture(dim, texture);
        if (dim == gl.TEXTURE_2D) {

            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, m.Cols / 4, m.Rows, 0, gl.RGBA, gl.FLOAT, m.dt);
        }
        else {
            Assert(dim == gl.TEXTURE_3D, "Set-Tex");

            gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA32F, m.Cols / 4, m.Rows, m.Depth, 0, gl.RGBA, gl.FLOAT, m.dt);
        }
    }

    Calc(param) {
        var gl = Mat.prototype.WebGL;

        var TEXTUREs = [gl.TEXTURE0, gl.TEXTURE1, gl.TEXTURE2, gl.TEXTURE3];

        var gpu = Mat.prototype.Prg[param.key];
        if (!gpu) {

            gpu = {};
            Mat.prototype.Prg[param.key] = gpu;

            gpu.key = param.key;

            var fsrc = Shaders['fs-transform'];
            var vshader = this.MakeShader(gl, gl.VERTEX_SHADER, param.vsrc);
            var fshader = this.MakeShader(gl, gl.FRAGMENT_SHADER, fsrc);
            gpu.program = this.MakeProgram(gl, vshader, fshader, param.varyings);
            gl.useProgram(gpu.program);

            // ユニフォーム変数の初期処理
            gpu.locUniforms = [];
            for(let u of param.uniforms) {

                var loc = gl.getUniformLocation(gpu.program, u.name);
                gpu.locUniforms.push(loc);
            }

            // テクスチャの初期処理
            gpu.locTextures = [];
            gpu.Textures = [];
            for (var i = 0; i < param.textures.length; i++) {

                var loc = gl.getUniformLocation(gpu.program, param.textures[i].name);
                gpu.locTextures.push(loc);

                var tex = this.MakeTex(gl, TEXTUREs[i], param.textures[i].dim);
                gpu.Textures.push(tex);
            }

            gpu.idxBuffer = this.MakeIdxBuffer(gl, gpu, param.elementCount);
            gpu.arrayBuffers = [];
            gpu.outBuffers = [];

            for (var i = 0; i < param.varyings.length; i++) {
                var sz = param.elementCount * Float32Array.BYTES_PER_ELEMENT;

                gpu.arrayBuffers.push( new ArrayBuffer(sz) );

                // Feedback empty buffer
                var buf = gl.createBuffer();
                gl.bindBuffer(gl.ARRAY_BUFFER, buf);
                gl.bufferData(gl.ARRAY_BUFFER, sz, gl.STATIC_COPY);
                gl.bindBuffer(gl.ARRAY_BUFFER, null);

                gpu.outBuffers.push(buf);
            }

            // -- Init TransformFeedback 
            gpu.transformFeedback = gl.createTransformFeedback();
        }
        else {

            gl.useProgram(gpu.program);
        }

        // -- Init Buffer

        gl.bindBuffer(gl.ARRAY_BUFFER, gpu.idxBuffer);
        gl.vertexAttribPointer(0, 1, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(0);

        gl.useProgram(gpu.program);

        gl.enable(gl.RASTERIZER_DISCARD);

        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, gpu.transformFeedback);

        // テクスチャの値のセット
        for (var i = 0; i < param.textures.length; i++) {

            this.SetTex(gl, param.textures[i].value, TEXTUREs[i], param.textures[i].dim, gpu.Textures[i]);
            gl.uniform1i(gpu.locTextures[i], i);
        }

        // ユニフォーム変数のセット
        for (var i = 0; i < param.uniforms.length; i++) {
            var u = param.uniforms[i];
            if (u.value instanceof Mat) {

                gl.uniform4fv(gpu.locUniforms[i], new Float32Array(u.value.dt));
            }
            else if (u.value instanceof Float32Array) {

                gl.uniform1fv(gpu.locUniforms[i], new Float32Array(u.value));
            }
            else {

                gl.uniform1i(gpu.locUniforms[i], u.value);
            }
        }

        for (var i = 0; i < param.varyings.length; i++) {

            gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, gpu.outBuffers[i]);
        }

        // 計算開始
        gl.beginTransformFeedback(gl.POINTS);    // TRIANGLES
        gl.drawArrays(gl.POINTS, 0, param.elementCount);
        gl.endTransformFeedback();

        gl.disable(gl.RASTERIZER_DISCARD);

        var ret = [];
        for (var i = 0; i < param.varyings.length; i++) {

            gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, null);

            // 処理結果を表示
            gl.bindBuffer(gl.ARRAY_BUFFER, gpu.outBuffers[i]);

            gl.getBufferSubData(gl.ARRAY_BUFFER, 0, gpu.arrayBuffers[i]);

            ret.push( new Float32Array(gpu.arrayBuffers[i]) );

            gl.bindBuffer(gl.ARRAY_BUFFER, null);
        }

        // 終了処理
        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);

        gl.useProgram(null);

        return ret;
    }
}

Mat.prototype.Clear = function () {
    var gl = Mat.prototype.WebGL;
    for (key in Mat.prototype.Prg) {
        var gpu = Mat.prototype.Prg[key];

        gl.bindBuffer(gl.ARRAY_BUFFER, null);
        gl.deleteBuffer(gpu.idxBuffer);
        gl.deleteBuffer(gpu.outBuffers);
        gl.deleteTransformFeedback(gpu.transformFeedback);

        gl.bindTexture(gl.TEXTURE_2D, null);
        gl.bindTexture(gl.TEXTURE_3D, null);
        for(let tex of gpu.Textures) {

            gl.deleteTexture(tex);
        }

        gl.deleteProgram(gpu.program);
        console.log("clear gpu:" + gpu.key);
    }
}

Mat.prototype.Init = function () {
    console.log("init WebGL");

    Mat.prototype.Prg = {};

    // -- Init Canvas
    var canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 32;
    document.body.appendChild(canvas);

    // -- Init WebGL Context
    var gl = canvas.getContext('webgl2', { antialias: false });
    var isWebGL2 = !!gl;
    if (!isWebGL2) {
        console.log("WebGL 2 is not available. See How to get a WebGL 2 implementation");
        console.log("https://www.khronos.org/webgl/wiki/Getting_a_WebGL_Implementation");

        throw "WebGL 2 is not available.";
    }

    Mat.prototype.WebGL = gl;
}
