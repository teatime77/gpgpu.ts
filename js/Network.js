﻿// import
// import
var isDebug = false;

class Layer {
    constructor() {
        this.fwCnt = 0;
        this.fwTime = 0;
        this.bwCnt = 0;
        this.bwTime = 0;
    }

    init(prev_layer) {
        this.prevLayer = prev_layer;
        if (prev_layer) {
            prev_layer.nextLayer = this;
        }
    }

    forward() {
    }

    backward(Y, eta2) {
    }

    forward2() {
        var startTime = new Date();
        this.forward();
        this.fwCnt++;
        this.fwTime += new Date() - startTime;
    }

    backward2(Y, eta2) {
        var startTime = new Date();
        this.backward(Y, eta2);
        this.bwCnt++;
        this.bwTime += new Date() - startTime;
    }

    updateParameter(eta2) {
    }
}

class InputLayer extends Layer {
    constructor(rows, cols) {
        super();

        this.imgRows = rows;
        this.imgCols = cols;
        this.unitSize = rows * cols;
    }
}

class FullyConnectedLayer extends Layer{
    constructor(size) {
        super();

        this.unitSize = size;
    }

    init(prev_layer) {
        super.init(prev_layer);

        this.bias = np.random.randn(this.unitSize, 1);
        this.weight = np.random.randn(this.unitSize, this.prevLayer.unitSize);
    }

    forward() {
        this.batchLength = this.prevLayer.activation.Cols;
        this.z = np.dot(this.weight, this.prevLayer.activation).AddV(this.bias);
        this.activation = sigmoid(this.z);
    }

    backward(Y, eta2) {
        if (!this.nextLayer) {
            // 最後のレイヤーの場合

            this.costDerivative = cost_derivative(this.activation, Y);
            this.Delta = this.costDerivative.Mul(sigmoid_prime(this.z));

            if (isDebug) {

                this.cost = xrange(this.costDerivative.Cols).map(c => this.costDerivative.Col(c).dt.map(x => x * x).reduce((x, y) => x + y));
            }
        }
        else {
            // 最後のレイヤーでない場合

            this.Delta = np.dot(this.nextLayer.weight.transpose(), this.nextLayer.Delta).Mul(sigmoid_prime(this.z));
        }
        this.nabla_b = this.Delta.reduce((x, y) => x + y);
        this.nabla_w = np.dot(this.Delta, this.prevLayer.activation.transpose());

        if (isDebug) {

            this.nablaBiases = this.Delta;
            // constructor(rows, cols, init, column_major, depth)
            this.nablaWeights = new Mat(this.weight.Rows, this.weight.Cols, null, false, this.batchLength);
            for (var batch_idx = 0; batch_idx < this.batchLength; batch_idx++) {
                for (var r = 0; r < this.weight.Rows; r++) {
                    for (var c = 0; c < this.weight.Cols; c++) {
                        var f = this.Delta.At(r, batch_idx) * this.prevLayer.activation.At(c, batch_idx);
                        this.nablaWeights.Set3(batch_idx, r, c, f);
                    }
                }
            }
        }
    }

    updateParameter(eta2) {
        this.weight = this.weight.Sub(eta2.Mul(this.nabla_w));
        this.bias = this.bias.Sub(eta2.Mul(this.nabla_b));
    }
}

class ConvolutionalLayer extends Layer{
    constructor(filter_size, filter_count) {
        super();

        this.filterSize = filter_size;
        this.filterCount = filter_count;
    }

    init(prev_layer) {
        super.init(prev_layer);

        Assert(this.prevLayer instanceof InputLayer, "Convolutional-Layer-init");

        this.imgRows = this.prevLayer.imgRows - this.filterSize + 1;
        this.imgCols = this.prevLayer.imgCols - this.filterSize + 1;
        this.unitSize = this.imgRows * this.imgCols * this.filterCount;

        this.biases = xrange(this.filterCount).map(x => np.random.randn());
        this.weights = xrange(this.filterCount).map(x => np.random.randn(this.filterSize, this.filterSize));
    }

    gpuForwardTest(prev_Layer) {
        var dt = new Float32Array(2 * 3 * 4 * 4);
        for (var i = 0; i < dt.length; i++) {
            dt[i] = i;
        }
        // (rows, cols, init, column_major, depth)
        var m = new Mat(3, 4 * 4, dt, false, 2);
        var param = {};

        param.elementCount = m.dt.length;

        var vs_id = "Test";
        param.textures = [
            { name: "prev_activation", value: m, dim: WebGL2.GL.TEXTURE_3D }
        ];

        param.uniforms = [];

        param.vsrc = Shaders[vs_id];
        param.varyings = ["z", "activation"];
        param.key = vs_id + ":" + this.batchLength + ":" + this.imgRows + ":" + this.imgCols + ":" + this.filterCount + ":" + this.filterSize;

        var ret = WebGL2.Calc(param);
    }

    gpuForward() {
        var prev_Layer = this.prevLayer;
        var weights = new Float32Array(this.filterCount * this.filterSize * this.filterSize);
        var biases = new Float32Array(this.filterCount);

        for(var filter_idx = 0; filter_idx < this.filterCount; filter_idx++){
            var weight = this.weights[filter_idx];
            for(var i = 0; i < weight.dt.length; i++){
                weights[filter_idx * weight.dt.length + i] = weight.dt[i];
            }
            biases[filter_idx] = this.biases[filter_idx];
        }

        // (rows, cols, init, column_major, depth)
        var m = new Mat(prev_Layer.imgRows, prev_Layer.imgCols, prev_Layer.activation.dt, false, this.batchLength);

        var param = {};

        param.elementCount = this.batchLength * this.imgRows * this.imgCols * this.filterCount;
        var vs_id = "ConvolutionalLayer-forward";
        param.textures = [
            { name: "prev_activation", value: m, dim: WebGL2.GL.TEXTURE_3D }
        ];

        param.uniforms = [
            { name: "weights", value: weights },
            { name: "biases", value: biases }
        ];

        if (this.forwardSrc == undefined) {

            this.forwardSrc = Shaders[vs_id]
                .replace(/imgCols_filterCount/g, (this.imgCols * this.filterCount).toString() + "u")
                .replace(/filterSize_filterSize/g, (this.filterSize * this.filterSize).toString() + "u")
                .replace(/unitSize/g   , this.unitSize.toString()+"u")
                .replace(/imgCols/g    , this.imgCols.toString())
                .replace(/filterCount/g, this.filterCount.toString()+"u")
                .replace(/filterSize/g, this.filterSize.toString() + "u");
        }
        param.vsrc = this.forwardSrc;

        param.varyings = ["z", "activation"];
        param.arrayBuffers = [this.zArrayBuffer, this.activationArrayBuffer];
        param.key = vs_id + ":" + this.imgRows + ":" + this.imgCols + ":" + this.filterCount + ":" + this.filterSize;

        var ret = WebGL2.Calc(param);

        return ret;
    }

    cpuForward() {
        var prev_Layer = this.prevLayer;

        var prev_activation_dt = prev_Layer.activation.dt;
        var z_dt = this.z.dt;
        var activation_dt = this.activation.dt;

        // バッチ内のデータに対し
        for (var batch_idx = 0; batch_idx < this.batchLength; batch_idx++) {

            // すべての行に対し
            for (var r1 = 0; r1 < this.imgRows; r1++) {

                // すべての列に対し
                for (var c1 = 0; c1 < this.imgCols; c1++) {

                    var output_base = batch_idx * this.unitSize + this.filterCount * (r1 * this.imgCols + c1);

                    // すべてのフィルターに対し
                    for (var filter_idx = 0; filter_idx < this.filterCount; filter_idx++) {

                        var weight = this.weights[filter_idx];
                        var bias = this.biases[filter_idx];

                        var sum = 0.0;

                        // フィルターの行に対し
                        for (var r2 = 0; r2 < this.filterSize; r2++) {

                            // フィルターの列に対し
                            for (var c2 = 0; c2 < this.filterSize; c2++) {
                                var weight_idx = r2 * this.filterSize + c2;
                                var prev_activation_idx = batch_idx * prev_Layer.unitSize + (r1 + r2) * prev_Layer.imgCols + (c1 + c2);
                                sum += prev_activation_dt[ prev_activation_idx ] * weight.dt[weight_idx];
                            }
                        }

                        // 出力先
                        var output_idx = output_base + filter_idx;

                        var z_val = sum + bias;

                        z_dt[output_idx] = z_val;
                        activation_dt[output_idx] = sigmoidF(z_val);

                    }
                }
            }
        }
    }

    forward() {
        this.batchLength = this.prevLayer.activation.Cols;

        if (!this.z || this.z.Rows != this.unitSize || this.z.Cols != this.batchLength){

            this.zArrayBuffer          = new ArrayBuffer(4 * this.unitSize * this.batchLength);
            this.activationArrayBuffer = new ArrayBuffer(4 * this.unitSize * this.batchLength);

            this.z          = new Mat(this.unitSize, this.batchLength, new Float32Array(this.zArrayBuffer), true);
            this.activation = new Mat(this.unitSize, this.batchLength, new Float32Array(this.activationArrayBuffer), true);

            //this.z             = new Mat(this.unitSize, this.batchLength, null, true);
            //this.activation    = new Mat(this.unitSize, this.batchLength, null, true);
        }

        if (this.batchLength == 12) {

            this.gpuForward();
        }
        else {

            this.cpuForward();
        }
        if (true) return;

        var t0 = new Date();
        this.gpuForward();
        var t1 = new Date();

        var z_gpu_dt = new Float32Array(this.z.dt);
        var activation_gpu_dt = new Float32Array(this.activation.dt);

        this.cpuForward();
        var t2 = new Date();

        var max_diff = 0;
        for (var k = 0; k < this.z.dt.length; k++) {
            var diff = Math.max(Math.abs(z_gpu_dt[k] - this.z.dt[k]), Math.abs(activation_gpu_dt[k] - this.activation.dt[k]));
            if(max_diff < diff){
                max_diff = diff;
            }
        }
        Assert(max_diff < 0.00001, "Convolutional-Layer-forward-diff");
        if (this.forwardCnt == undefined) {

            this.forwardCnt = 0;
            this.forwardGPU = 0;
            this.forwardCPU = 0;
        }
        this.forwardCnt++;
        this.forwardGPU += t1 - t0;
        this.forwardCPU += t2 - t1;
        if (this.forwardCnt % 100 == 0) {

            console.log("forward diff:%f cnt:%d GPU:%dms CPU:%dms", max_diff, this.forwardCnt, Math.round(this.forwardGPU / this.forwardCnt), Math.round(this.forwardCPU / this.forwardCnt));
        }
    }

    backward(Y, eta2) {
        var eta3 = eta2 / (this.filterSize * this.filterSize);

        //this.Delta = this.nextLayer.Delta.Mul(sigmoid_prime(this.z));
        var deltaT = new Float32Array(this.nextLayer.DeltaT);
        for (var i = 0; i < deltaT.length; i++) {
            deltaT[i] *= sigmoid_primeF(this.z.dt[i]);
        }


        var prev_Layer = this.prevLayer;
        var prev_activation_dt = prev_Layer.activation.dt;

        for(var filter_idx = 0; filter_idx < this.filterCount; filter_idx++) {

            var nabla_b = 0.0;
            for (var r2 = 0; r2 < this.filterSize; r2++) {
                for (var c2 = 0; c2 < this.filterSize; c2++) {

                    var nabla_w = 0.0;

                    for (var batch_idx = 0; batch_idx < this.batchLength; batch_idx++) {

                        for (var r1 = 0; r1 < this.imgRows; r1++) {
                            for (var c1 = 0; c1 < this.imgCols; c1++) {

                                var output_base = batch_idx * this.unitSize + this.filterCount * (r1 * this.imgCols + c1);
                                var out_idx = output_base + filter_idx;

                                var delta = deltaT[out_idx];
                                if (delta != 0) {

                                    var prev_activation_idx = batch_idx * prev_Layer.unitSize + (r1 + r2) * prev_Layer.imgCols + (c1 + c2);

                                    nabla_w += delta * prev_activation_dt[ prev_activation_idx ];

                                    nabla_b += delta;
                                }
                            }
                        }
                    }

                    var weight_idx = r2 * this.filterSize + c2;
                    this.weights[filter_idx].dt[weight_idx] -= eta3 * nabla_w;
                }
            }

            this.biases[filter_idx] -= eta3 * nabla_b;
        }
    }
}

class PoolingLayer extends Layer {
    constructor(filter_size) {
        super();
        this.filterSize = filter_size;
    }

    init(prev_layer) {
        super.init(prev_layer);

        Assert(this.prevLayer instanceof ConvolutionalLayer, "Pooling-Layer-init");

        this.imgRows = this.prevLayer.imgRows / this.filterSize;
        this.imgCols = this.prevLayer.imgCols / this.filterSize;
        this.filterCount = this.prevLayer.filterCount;

        this.unitSize = this.imgRows * this.imgCols * this.filterCount;
    }

    forward() {
        var prev_Layer = this.prevLayer;
        var prev_activation_dt = prev_Layer.activation.dt;
        this.batchLength = prev_Layer.batchLength;
        var out_dt = new Float32Array(this.unitSize * this.batchLength);

        this.maxIdx = new Int8Array(this.unitSize * this.batchLength);

        // バッチ内のデータに対し
        for (var batch_idx = 0; batch_idx < this.batchLength; batch_idx++) {

            // すべての行に対し
            for (var r1 = 0; r1 < this.imgRows; r1++) {

                // すべての列に対し
                for (var c1 = 0; c1 < this.imgCols; c1++) {

                    var output_base = batch_idx * this.unitSize + this.filterCount * (r1 * this.imgCols + c1);

                    // すべてのフィルターに対し
                    for (var filter_idx = 0; filter_idx < this.filterCount; filter_idx++) {

                        var max_val = -10000;
                        var max_idx;

                        // フィルターの行に対し
                        for (var r2 = 0; r2 < this.filterSize; r2++) {

                            // フィルターの列に対し
                            for (var c2 = 0; c2 < this.filterSize; c2++) {

                                var prev_activation_idx = batch_idx * prev_Layer.unitSize + prev_Layer.filterCount * ((r1 + r2) * prev_Layer.imgCols + (c1 + c2)) + filter_idx;
                                var val = prev_activation_dt[prev_activation_idx];
                                if (max_val < val) {

                                    max_val = val;
                                    max_idx = r2 * this.filterSize + c2;
                                }
                            }
                        }

                        // 出力先
                        var output_idx = output_base + filter_idx;

                        out_dt[output_idx] = max_val;
                        this.maxIdx[output_idx] = max_idx;
                    }
                }
            }
        }

        this.activation = (new Mat(this.batchLength, this.unitSize, out_dt)).T();
    }

    backward(Y, eta2) {
        var deltaT = np.dot(this.nextLayer.weight.transpose(), this.nextLayer.Delta).T();
        Assert(deltaT.Rows == this.batchLength && deltaT.Cols == this.unitSize, "Pooling-Layer-backward");

        this.DeltaT = new Float32Array(this.prevLayer.activation.dt.length);

        var filter_stride = this.filterSize * this.filterSize;
        for (var i = 0; i < deltaT.dt.length; i++) {
            var k = i * filter_stride + this.maxIdx[i];
            this.DeltaT[k] = deltaT.dt[i];
        }
    }
}

class Network {
    constructor(layers) {
        this.layers = layers;

        var prev_layer = null;
        for(let layer of layers) {
            layer.init(prev_layer);
            prev_layer = layer;
        }
    }

    SGD(training_data, epochs, mini_batch_size, eta, test_data) {
        this.miniBatchSize = mini_batch_size;
        var n_test;//??
        if(test_data == undefined){ test_data = None;}
        if(test_data){
            n_test = test_data["count"];
        }
        var n=len(training_data);//??
        for (let j of xrange(epochs)) {

            var startTime = new Date();
            np.random.shuffle(training_data);//??
            console.log("shuffle:" + (new Date() - startTime) + "ms");

            startTime = new Date();
            var mini_batches = xrange(0, n, mini_batch_size).map(k => Slice(training_data, [k, k + mini_batch_size]));//??
            console.log("mini_batches:" + (new Date() - startTime) + "ms");

            startTime = new Date();
            for (let mini_batch of mini_batches) {
                var X = this.Laminate(mini_batch, 0);
                var Y = this.Laminate(mini_batch, 1);
                this.update_mini_batch(X, Y, eta);
                if (this.layers[1].fwCnt % 1000 == 0) {

                    var s = "" + this.layers[1].fwCnt + " ";
                    for(let layer of this.layers.slice(1)) {
                        s += " (" + Math.floor(layer.fwTime / layer.fwCnt) + " " + Math.floor(layer.bwTime / layer.bwCnt) + ")";
                    }
                    console.log("update mini batch:" + s);
                }
            }
            console.log("update_mini_batch:" + (new Date() - startTime) + "ms");

            if(test_data){
                //??                console.log("Epoch {0}: {1} / {2}".format(j, this.evaluate(test_data), n_test));
                startTime = new Date();
                var e = this.evaluate(test_data);
                console.log("evaluate:" + (new Date() - startTime) + "ms");

                console.log("Epoch %d: %d / %d", j, e, n_test);
            }
            else{
//??                console.log("Epoch {0} complete".format(j));
                console.log("Epoch %d complete", j);
            }
        }
    }

    Laminate(mini_batch, i) {
        var x_rows = mini_batch[0][i].Rows;
        var X = new Mat(x_rows, mini_batch.length);
        for (var idx = 0; idx < mini_batch.length; idx++) {
            var x = mini_batch[idx][i];
            for (var r = 0; r < x_rows; r++) {
                X.dt[r * X.Cols + idx] = x.dt[r];
            }
        }

        return X;
    }

    update_mini_batch(X, Y, eta) {
        this.layers[0].activation = X;
        this.layers.forEach(x => x.forward2());

        var eta2 = eta / X.Cols;

        for (var i = this.layers.length - 1; 1 <= i; i--) {
            this.layers[i].backward2(Y, eta2);
        }


        if (!isDebug) {
            this.layers.forEach(x => x.updateParameter(eta2));
        }
        else {
            var delta = 0.001;

            var last_layer = this.layers[this.layers.length - 1];
            var last_z_sv = new Float32Array(last_layer.z.dt);
            var activation_sv = new Float32Array(last_layer.activation.dt);
            var cost_sv = last_layer.cost;

            for(var layer_idx = this.layers.length - 1; 0 <= layer_idx; layer_idx--) {
                var layer = this.layers[layer_idx];
                if (layer.nablaBiases) {
                    for (var batch_idx = 0; batch_idx < this.miniBatchSize; batch_idx++) {

                        for (var r = 0; r < layer.nablaBiases.Rows; r++) {
                            var b = layer.bias.dt[r];
                            var cd = layer.costDerivative.dt[r];
                            var nb = layer.nablaBiases.dt[r];
                            layer.bias.dt[r] -= delta;

                            console.log("nable検証 : %f %f", nb, cd);//, layer.prevLayer.activation.At(c, 0)
                            console.log("activation :" + layer.activation.dt);
                            console.log("z :" + layer.z.dt);

                            this.layers.forEach(x => x.forward2());

                            for (var i = this.layers.length - 1; 1 <= i; i--) {
                                this.layers[i].backward2(Y, eta2);
                            }

                            var cost = this.layers[this.layers.length - 1].cost;
                            var cost_diff = xrange(cost.length).map(i => cost_sv[i] - cost[i]);
                            console.log("C = Σ(ai - yi)^2 = " + cost_sv[batch_idx]);
                            console.log("δC/δa0 = 2 * (a0 - y0) = 2 * (" + activation_sv[r] + "-" + Y.dt[r] + ") = " + (2 * (activation_sv[r] - Y.dt[r])));
                            
                            var deltaC1 = cost[batch_idx] - cost_sv[batch_idx];
                            console.log("ΔC = " + deltaC1);

                            console.log("Δa0 = " + (layer.activation.dt[r] - activation_sv[r]));

                            var deltaC2 = (activation_sv[r] - layer.activation.dt[r]) * 2 * (activation_sv[r] - Y.dt[r]);
                            console.log("ΔC ≒ Δa0 * 2 * (a0 - y0) = " + deltaC2);

                            console.log("ΔC誤差 = " + (deltaC1 - deltaC2));

                            console.log(cost);
                            console.log(cost_diff);
                            console.log("cost-Derivative : %f", layer.costDerivative.dt[r]);
                            console.log("activation :" + layer.activation.dt);
                            console.log("activation-diff :" + xrange(activation_sv.length).map(i => activation_sv[i] - layer.activation.dt[i]));
                            console.log("last_z_sv :" + last_z_sv);
                            console.log("z    :" + layer.z.dt);
                            console.log("z-diff :" + xrange(last_z_sv.length).map(i => last_z_sv[i] - layer.z.dt[i]));

                            for (var r2 = 0; r2 < layer.nablaBiases.Rows; r2++) {
                                if (r2 != r) {
                                    Assert(last_z_sv[r2] - last_layer.z.dt[r2] == 0 && activation_sv[r2] - last_layer.activation.dt[r2] == 0, "z-activation-diff");
                                }
                            }

                            layer.bias.dt[r] = b;
                        }
                    }
                }
            }

            var cost1 = this.layers[this.layers.length - 1].costDerivative;
            console.log(this.costAvg(cost1));

            this.layers.forEach(x => x.forward2());
            var cost2 = cost_derivative(this.layers[this.layers.length - 1].activation, Y);
            console.log(this.costAvg( cost2 ));
        }
    }

    costAvg(cost) {
        return xrange(cost.Cols).map(c => cost.Col(c).dt.map(x => Math.abs(x)).reduce((x, y) => x + y) / cost.Rows);
    }

    feedforward(a) {
        for(let l of this.layers.slice(1)) {
            if (l instanceof FullyConnectedLayer) {

                a = sigmoid(np.dot(l.weight, a).Add(l.bias));
            }
        }

        return a;
    }

    evaluate(test_data) {
        var cnt = test_data["count"];
        var labels = test_data["label"];

        var X = new Mat(cnt, 28 * 28, TestData["image"]).T();
        this.layers[0].activation = X;
        this.layers.forEach(x => x.forward());

        var result = this.layers[this.layers.length - 1].activation;

        return sum(xrange(cnt).map(c => np.argmax(result.Col(c)) == labels[c] ? 1 : 0));

//        var test_results = test_data.map($ => { var x = $[0]; var y = $[1]; return [np.argmax(this.feedforward(x)), y]; });
//        return sum(test_results.map($ => {var x = $[0];var y = $[1];return /*int*/(x == y ? 1 : 0);}));
    }
}

function cost_derivative(output_activations, y){
    return (output_activations.Sub(y));
}

function sigmoid(z){
//??    return 1.0 / (1.0 + np.exp(-z));
    return z.map(x => sigmoidF(x));
}

function sigmoid_prime(z){
//??    return sigmoid(z) * (1 - sigmoid(z));
    return z.map(x => sigmoid_primeF(x));
}

function sigmoid_primeF(z) {
    var f = sigmoidF(z);
    return f * (1 - f);
}

//??
function sigmoidF(z){
    return 1.0 / (1.0 + Math.exp(-z));
}

//??
Array.prototype.GetAt = function (i) {
    if (0 <= i) {
        return this[i];
    }
    else {
        return this[this.length + i];
    }
}
