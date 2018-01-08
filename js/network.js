﻿var ActivationFunction = {
    none : 0,
    sigmoid: 1,
    ReLU : 2,
};

function CreateNeuralNetwork(gpgpu){
    var miniBatchSize;
    var miniBatchIdx;
    var learningRate;
    var useSoftMax = true;
    var WebGL2;
    var isTest = false;
    var TrainingDataCnt;
    var WeightDecay = 5.0;
    var Momentum = 0.9;
    var useGradientCheck = false;
    var inGradientCheck = false;
    var Shaders = CreateNeuralNetworkShaders();
    var random = new RandomHelper();

    function Stats(tm, idx){
        switch(tm.length){
        case 0:
            return "-"
        case 1:
            return "" + Math.floor(tm[0] / idx)
        default:
            return "[" + tm.map(x => Math.floor(x / idx)).reduce((x,y) => x + "," + y) + "]";
        }
    }

    class Lap {
        constructor(v){
            this.lastTime = new Date();
            this.lapIdx = 0;
            this.lapTimes = v;
        }

        Time(){
            var prev_last_time = this.lastTime;
            this.lastTime = new Date();

            if(this.lapTimes.length <= this.lapIdx){
                this.lapTimes.push(0);
            }
            this.lapTimes[this.lapIdx] += this.lastTime - prev_last_time;
            this.lapIdx++;
        }
    }

    class Layer {
        constructor() {
        }

        init(prev_layer) {
            this.prevLayer = prev_layer;
            if (prev_layer) {
                prev_layer.nextLayer = this;
            }
        }

        miniBatchSizeChanged(){
            this.fwTime = [];
            this.bwTime = [];
            this.udTime = [];
        }

        forward() {
        }

        Backpropagation() {
        }

        updateParameter() {
        }

        clear(){
            if(this.params){

                for(var key in this.params){
                    WebGL2.clear(this.params[key].id);
                }
                this.params = {};
            }
        }

        GradientCheck(net, batch_Y, exp_work, cost, batch_idx, layer_idx){
            if(! this.prevLayer){
                return;
            }

            var last_layer = net.layers[net.layers.length - 1];
            var cost_derivative_work = new Float32Array(last_layer.costDerivative.dt.length);

            // deltaX
            if(this.deltaX){

                var prev_Layer = this.prevLayer;

                var idx_list = random.RandomSampling(prev_Layer.unitSize, 3);

                for(let i of idx_list){
                    var k = batch_idx * prev_Layer.unitSize + i;
                    var x = prev_Layer.activation.dt[k];
                    var dx = this.deltaX.dt[k];
                    var eps = x * 0.01;

                    prev_Layer.activation.dt[k] = x - eps;
                    var cost1 = net.forwardCost(batch_Y, exp_work, batch_idx, layer_idx, cost_derivative_work);

                    prev_Layer.activation.dt[k] = x + eps;
                    var cost2 = net.forwardCost(batch_Y, exp_work, batch_idx, layer_idx, cost_derivative_work);

                    var diff = dx * 2 * eps - (cost2 - cost1);
                    console.log("delta-X : %f dC:%f eps:%f cost:%f,%f,%f", diff, dx, eps, cost1, cost - cost1, cost2 - cost);

                    prev_Layer.activation.dt[k] = x;
                }
            }

            if(this.deltaBias){

                net.ParamGradientCheck("bias", this.bias.dt, this.deltaBias.dt, batch_Y, exp_work, cost, batch_idx, layer_idx, cost_derivative_work);
            }

            if(this.deltaWeight){

                net.ParamGradientCheck("weight", this.weight.dt, this.deltaWeight.dt, batch_Y, exp_work, cost, batch_idx, layer_idx, cost_derivative_work);
            }
        }
    }

    class InputLayer extends Layer {
        constructor(channel_size, rows, cols) {
            super();

            this.channelSize = channel_size;
            this.imgRows = rows;
            this.imgCols = cols;
            this.unitSize = rows * cols;
        }
    }

    class AffineTransformationLayer extends Layer{
        cpuDeltaZ(){

            // 活性化関数
            switch(this.activationFunction){
            case ActivationFunction.none:
                for (var i = 0; i < this.deltaZ.dt.length; i++) {
                    this.deltaZ.dt[i] = this.costDerivative.dt[i];
                }
                break;

            case ActivationFunction.sigmoid:
                for (var i = 0; i < this.deltaZ.dt.length; i++) {
                    this.deltaZ.dt[i] = this.costDerivative.dt[i] * sigmoid_primeF(this.z.dt[i]);
                }
                break;

            case ActivationFunction.ReLU:
                for (var i = 0; i < this.deltaZ.dt.length; i++) {
                    this.deltaZ.dt[i] = (this.z.dt[i] <= 0 ? 0 : this.costDerivative.dt[i]);
                }
                break;
            }
        }
    }

    class FullyConnectedLayer extends AffineTransformationLayer {
        constructor(size, activation_function) {
            super();

            this.unitSize = size;
            this.activationFunction = activation_function;
            this.params = {};
        }

        init(prev_layer) {
            super.init(prev_layer);

            this.bias = random.randn(this.unitSize, 1);
            this.weight = random.randn(this.unitSize, this.prevLayer.unitSize);

            if(this.activationFunction == ActivationFunction.ReLU){
                var sd = Math.sqrt(prev_layer.unitSize);
                for(var i = 0; i < this.weight.dt.length; i++){
                    this.weight.dt[i] /= sd;
                }
            }

            this.deltaWeight = new ArrayView(this.unitSize, this.prevLayer.unitSize);
        }

        miniBatchSizeChanged(){
            super.miniBatchSizeChanged();

            this.outZero    = new Float32Array(miniBatchSize * this.unitSize);
            this.z          = new ArrayView(miniBatchSize,  this.unitSize);
            this.activation = new ArrayView(miniBatchSize,  this.unitSize);
            this.deltaX     = new ArrayView(miniBatchSize,  this.prevLayer.unitSize);
            this.prevLayerActivation = new ArrayView(miniBatchSize,  this.prevLayer.unitSize);

            this.deltaZ     = new ArrayView(miniBatchSize,  this.unitSize);

            if(!this.nextLayer){
                // 最後の場合

                this.costDerivative = new ArrayView(miniBatchSize,  this.unitSize);
            }
        }

        gpuForward(){
            var vertex_shader = Shaders.FullyConnectedLayer_Forward;

            this.param = {
                id : "Fully-Connected-Layer-forward," + miniBatchSize + "," + this.prevLayer.unitSize + "," + this.unitSize,
                vertexShader: vertex_shader,
                args : {
                    "activationFunction": this.activationFunction,
                    "zero": this.outZero,
                    "X": WebGL2.makeTextureInfo("float", [ miniBatchSize, this.prevLayer.unitSize], this.prevLayer.activation.dt),
                    "W": WebGL2.makeTextureInfo("float", this.weight.shape, this.weight.dt),
                    "Bias": WebGL2.makeTextureInfo("float", [ 1, this.bias.dt.length ], this.bias.dt),
                    "z": this.z.dt,
                    "activation" : this.activation.dt
                }
            };

            WebGL2.compute(this.param);
        }

        forward() {
            var lap = new Lap(this.fwTime);
            this.gpuForward();

            lap.Time();
        }

        gpuDeltaX(){
            var vertex_shader = Shaders.FullyConnectedLayer_DeltaX;

            var param_id = "Fully-Connected-Layer-gpu-delta-X," + miniBatchSize + "," + this.prevLayer.unitSize + "," + this.unitSize;
            if (this.params[param_id] == undefined){

                this.params[param_id] = {
                    id : param_id,
                    vertexShader: vertex_shader,
                    args : {
                        "zero": new Float32Array(miniBatchSize * this.prevLayer.unitSize),
                        "W": makeTextureInfo(WebGL2, "float", this.weight),
                        "deltaZ": makeTextureInfo(WebGL2, "float", this.deltaZ),
                        "deltaX" : this.deltaX.dt
                    }
                };
            }

            var param = this.params[param_id];
            param.args["deltaZ"].value = this.deltaZ.dt;

            WebGL2.compute(param);
        }

        cpuDeltaX(){

            // 出力先
            var output_idx = 0;

            // バッチ内のデータに対し
            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                // 入力に対し
                for (var x_idx = 0; x_idx < this.prevLayer.unitSize; x_idx++) {

                    var sum = 0.0;

                    // 重みの行とδzの内積
                    for (var k = 0; k < this.weight.nrow; k++) {
                        var weight_idx = k * this.weight.ncol + x_idx;
                        var delta_z_idx = batch_idx * this.unitSize + k;
                        sum += this.deltaZ.dt[delta_z_idx] * this.weight.dt[weight_idx];
                    }

                    this.deltaX.dt[output_idx] = sum;
                    output_idx++;
                }
            }
        }

        gpuDeltaWeight(){
            var vertex_shader = Shaders.FullyConnectedLayer_DeltaWeight;

            var prev_Layer = this.prevLayer;

            var vertex_shader = vertex_shader
                .replace(/miniBatchSize/g, miniBatchSize.toString())
                .replace(/WeightColSize/g, prev_Layer.unitSize.toString());

            var param_id = "Fully-Connected-Layer-delta-weight," + miniBatchSize + "," + prev_Layer.unitSize + "," + this.unitSize;
            if (this.params[param_id] == undefined){

                this.params[param_id] = {
                    id : param_id,
                    vertexShader: vertex_shader,
                    args : {
                        "zero": new Float32Array(this.deltaWeight.dt.length),
                        "prev_activation": makeTextureInfo(WebGL2, "float", new ArrayView(miniBatchSize, prev_Layer.unitSize)),
                        "deltaZ": makeTextureInfo(WebGL2, "float", this.deltaZ),
                        "deltaWeight" : this.deltaWeight.dt
                    }
                };
            }

            var param = this.params[param_id];

            param.args["prev_activation"].value = prev_Layer.activation.dt;;
            param.args["deltaZ"].value = this.deltaZ.dt;
            param.args["deltaWeight"].value = this.deltaWeight.dt;

            WebGL2.compute(param);
        }

        Backpropagation() {
            var lap = new Lap(this.bwTime);

            if (this.nextLayer) {
                // 最後のレイヤーでない場合

                this.costDerivative = this.nextLayer.deltaX;
            }

            this.cpuDeltaZ();

            lap.Time();

            this.deltaBias = this.deltaZ.Reduce((x, y) => x + y);
            lap.Time();

            this.gpuDeltaWeight();
            lap.Time();

            if(this.prevLayer instanceof InputLayer){

                return;
            }

            this.gpuDeltaX();
            if(Math.random() < 0.01){

                var gpu_delta_x = new Float32Array(this.deltaX.dt);
                this.cpuDeltaX();

                var diff = this.deltaX.diff(gpu_delta_x);
                Assert(diff < 0.01, "delta-X");
            }
            lap.Time();
        }

        updateParameter() {
            var eta = learningRate / miniBatchSize;
            var c = 1.0 - learningRate * WeightDecay / TrainingDataCnt;

            var lap = new Lap(this.udTime);

            for(var i = 0; i < this.weight.dt.length; i++){
                this.weight.dt[i] -= eta * this.deltaWeight.dt[i];
    /*
                var v = Momentum * this.weightV.dt[i] - eta * this.deltaWeight.dt[i];
                this.weightV.dt[i] = v;

                this.weight.dt[i] = c * this.weight.dt[i] + v;
    */
            }
            lap.Time();

            for(var i = 0; i < this.bias.dt.length; i++){
                this.bias.dt[i] -= eta * this.deltaBias.dt[i];
            }
            lap.Time();
        }
    }

    class ConvolutionalLayer extends AffineTransformationLayer {
        constructor(filter_size, channel_size, activation_function) {
            super();

            this.filterSize = filter_size;
            this.channelSize = channel_size;
            this.activationFunction = activation_function;

            this.params = {};
        }

        init(prev_layer) {
            super.init(prev_layer);

            this.imgRows = this.prevLayer.imgRows - this.filterSize + 1;
            this.imgCols = this.prevLayer.imgCols - this.filterSize + 1;
            this.unitSize = this.channelSize * this.imgRows * this.imgCols;

            this.bias = new ArrayView(this.channelSize);
            for (var i = 0; i < this.bias.dt.length; i++) {
                this.bias.dt[i] = random.randn();
            }

            this.weight = new ArrayView(this.channelSize, prev_layer.channelSize, this.filterSize, this.filterSize);
            for (var i = 0; i < this.weight.dt.length; i++) {
                this.weight.dt[i] = random.randn();
            }

            if(this.activationFunction == ActivationFunction.ReLU){
                var sd = Math.sqrt(prev_layer.unitSize);
                for(var i = 0; i < this.weight.dt.length; i++){
                    this.weight.dt[i] /= sd;
                }
            }

            this.deltaBias    = new ArrayView(this.channelSize);
            this.deltaWeight   = new ArrayView(this.channelSize, prev_layer.channelSize, this.filterSize, this.filterSize);
        }

        miniBatchSizeChanged(){
            super.miniBatchSizeChanged();

            this.z = new ArrayView(miniBatchSize, this.unitSize);
            this.activation = new ArrayView(miniBatchSize, this.unitSize);
            this.zero = new Float32Array(miniBatchSize * this.unitSize);

            this.deltaZ     = new ArrayView(miniBatchSize,  this.unitSize);

            if(this.prevLayer instanceof InputLayer){

                this.deltaX = undefined;
            }
            else{

                this.deltaX = new ArrayView(miniBatchSize, this.prevLayer.unitSize);
            }

        }

        gpuForward() {
            var prev_Layer = this.prevLayer;

            var prev_activation = new ArrayView(miniBatchSize, prev_Layer.unitSize, prev_Layer.activation.dt);

            var param_id = "ConvolutionalLayer-forward:" + this.filterSize + ":" + prev_Layer.channelSize + ":" + this.channelSize + ":" + this.imgRows + ":" + this.imgCols + ":" + miniBatchSize;

            if (this.params[param_id] == undefined) {

                var shader_src = Shaders.ConvolutionalLayer_Forward
                    .replace(/channelSize/g, this.channelSize.toString() + "u")
                    .replace(/prevChannelSize/g, prev_Layer.channelSize.toString() + "u")
                    .replace(/rowCount/g, this.imgRows.toString() + "u")
                    .replace(/colCount/g, this.imgCols.toString() + "u")
                    .replace(/filterSize/g, this.filterSize.toString() + "u");

                this.params[param_id]  = {
                    id : param_id,
                    vertexShader: shader_src,
                    args : {
                        "activationFunction": this.activationFunction,
                        "zero": this.zero,
                        "prev_activation": makeTextureInfo(WebGL2, "float", new ArrayView(miniBatchSize * prev_Layer.channelSize, prev_Layer.imgRows, prev_Layer.imgCols)),
                        "weights": makeTextureInfo(WebGL2, "float", new ArrayView(this.channelSize * prev_Layer.channelSize, this.filterSize, this.filterSize)),
    //                    "weights": this.weight.dt,
                        "biases": this.bias.dt,
                        "z": this.z.dt,
                        "activation": this.activation.dt
                    }
                };
            }

            var param = this.params[param_id];

            param.args["prev_activation"].value = prev_activation.dt;
            param.args["weights"].value = this.weight.dt;
            WebGL2.compute(param);
        }

        cpuForward() {
            var prev_Layer = this.prevLayer;

            var prev_activation_dt = prev_Layer.activation.dt;
            var z_dt = this.z.dt;
            var activation_dt = this.activation.dt;

            // 出力先
            var output_idx = 0;

            // バッチ内のデータに対し
            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                // すべての特徴マップに対し
                for (var channel_idx = 0; channel_idx < this.channelSize; channel_idx++) {

                    // 出力の行に対し
                    for (var r1 = 0; r1 < this.imgRows; r1++) {

                        // 出力の列に対し
                        for (var c1 = 0; c1 < this.imgCols; c1++) {

                            var sum = 0.0;
                            var weight_idx = channel_idx * prev_Layer.channelSize * this.filterSize * this.filterSize;
                            var prev_activation_base = batch_idx * prev_Layer.channelSize * prev_Layer.imgRows * prev_Layer.imgCols;

                            // 入力のチャネルに対し
                            for(var prev_channel_idx = 0; prev_channel_idx < prev_Layer.channelSize; prev_channel_idx++){

                                // フィルターの行に対し
                                for (var r2 = 0; r2 < this.filterSize; r2++) {

                                    // フィルターの列に対し
                                    for (var c2 = 0; c2 < this.filterSize; c2++) {
                                        var prev_activation_idx = prev_activation_base + (r1 + r2) * prev_Layer.imgCols + (c1 + c2);
                                        sum += prev_activation_dt[prev_activation_idx] * this.weight.dt[weight_idx];
                                        weight_idx++;
                                    }
                                }
                                prev_activation_base += prev_Layer.imgRows * prev_Layer.imgCols;
                            }

                            var z_val = sum + this.bias.dt[channel_idx];

                            z_dt[output_idx] = z_val;

                            // 活性化関数
                            switch(this.activationFunction){
                            case ActivationFunction.none:
                                activation_dt[output_idx] = z_val;
                                break;

                            case ActivationFunction.sigmoid:
                                activation_dt[output_idx] =  sigmoidF(z_val);
                                break;

                            case ActivationFunction.ReLU:
                                activation_dt[output_idx] = (0.0 < z_val ? z_val : 0.0);
                                break;
                            }

                            output_idx++;
                        }

                    }
                }
            }
        }

        forward() {
            var lap = new Lap(this.fwTime);

            this.gpuForward();

            lap.Time();

            if(miniBatchIdx != 0 && 0.1 < Math.random()){

                return;
            }

            var z_gpu_dt          = new Float32Array(this.z.dt);
            var activation_gpu_dt = new Float32Array(this.activation.dt);

            this.cpuForward();

            var max_diff = 0;

            // 出力先
            var output_idx = 0;
            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {
                for (var channel_idx = 0; channel_idx < this.channelSize; channel_idx++) {
                    for (var r1 = 0; r1 < this.imgRows; r1++) {
                        for (var c1 = 0; c1 < this.imgCols; c1++) {
                            var diff = Math.max(Math.abs(z_gpu_dt[output_idx] - this.z.dt[output_idx]), Math.abs(activation_gpu_dt[output_idx] - this.activation.dt[output_idx]));
                            if (max_diff < diff) {
                                if(0.001 < diff){
                                    console.log("CNN forward : %dx%d %d %d %d %d %f", this.imgRows, this.imgCols, batch_idx, channel_idx, r1, c1, diff)
                                }
                                max_diff = diff;
                            }
                            output_idx++;
                        }
                    }
                }
            }
        }

        gpuDeltaWeights() {
            var prev_Layer = this.prevLayer;

            var param_id = "ConvolutionalLayer-dabla-weights:" + this.filterSize + ":" + prev_Layer.channelSize + ":" + this.channelSize + ":" + this.imgRows + ":" + this.imgCols + ":" + miniBatchSize;

            if (this.params[param_id] == undefined) {

                var shader_src = Shaders.ConvolutionalLayer_DeltaWeights
                    .replace(/miniBatchSize/g, miniBatchSize.toString() + "u")
                    .replace(/channelSize/g, this.channelSize.toString() + "u")
                    .replace(/prevChannelSize/g, prev_Layer.channelSize.toString() + "u")
                    .replace(/rowCount/g, this.imgRows.toString() + "u")
                    .replace(/colCount/g, this.imgCols.toString() + "u")
                    .replace(/filterSize/g, this.filterSize.toString() + "u");

                this.params[param_id]  = {
                    id : param_id,
                    vertexShader: shader_src,
                    args : {
                        "zero": new Float32Array(this.weight.dt.length),
                        "prev_activation": makeTextureInfo(WebGL2, "float", new ArrayView(miniBatchSize * prev_Layer.channelSize, prev_Layer.imgRows, prev_Layer.imgCols)),
                        "delta_z": makeTextureInfo(WebGL2, "float", new ArrayView(miniBatchSize * this.channelSize, this.imgRows, this.imgCols)),
                        "nabla_w": this.deltaWeight.dt
                    }
                };
            }

            var param = this.params[param_id];

            param.args["prev_activation"].value = prev_Layer.activation.dt;
            param.args["delta_z"].value = this.deltaZ.dt;
            WebGL2.compute(param);
        }


        gpuDeltaX() {
            var prev_Layer = this.prevLayer;

            var param_id = "ConvolutionalLayer-delta-X:" + this.filterSize + ":" + prev_Layer.channelSize + ":" + this.channelSize + ":" + this.imgRows + ":" + this.imgCols + ":" + miniBatchSize;

            if (this.params[param_id] == undefined) {

                var shader_src = Shaders.ConvolutionalLayer_DeltaX
                    .replace(/miniBatchSize/g, miniBatchSize.toString() + "u")
                    .replace(/channelSize/g, this.channelSize.toString() + "u")
                    .replace(/prevChannelSize/g, prev_Layer.channelSize.toString() + "u")
                    .replace(/prevRowCount/g, prev_Layer.imgRows.toString() + "u")
                    .replace(/prevColCount/g, prev_Layer.imgCols.toString() + "u")
                    .replace(/rowCount/g, this.imgRows.toString() + "u")
                    .replace(/colCount/g, this.imgCols.toString() + "u")
                    .replace(/filterSize/g, this.filterSize.toString() + "u");

                this.params[param_id]  = {
                    id : param_id,
                    vertexShader: shader_src,
                    args : {
                        "zero": new Float32Array(this.deltaX.dt.length),
                        "weights": makeTextureInfo(WebGL2, "float", new ArrayView(this.channelSize * prev_Layer.channelSize, this.filterSize, this.filterSize)),
                        "delta_z": makeTextureInfo(WebGL2, "float", new ArrayView(miniBatchSize * this.channelSize, this.imgRows, this.imgCols)),
                        "delta_x": this.deltaX.dt
                    }
                };
            }

            var param = this.params[param_id];

            param.args["weights"].value = this.weight.dt;
            param.args["delta_z"].value = this.deltaZ.dt;
            WebGL2.compute(param);
        }

        cpuDeltaWeights() {
            var prev_Layer = this.prevLayer;
            var RC = this.imgRows * this.imgCols;
            var prev_RC = prev_Layer.imgRows * prev_Layer.imgCols;

            // 出力のチャネルに対し
            var weights_idx = 0;
            for (var channel_idx = 0; channel_idx < this.channelSize; channel_idx++) {

                // 入力のチャネルに対し
                for(var prev_channel_idx = 0; prev_channel_idx < prev_Layer.channelSize; prev_channel_idx++){

                    // フィルターの行に対し
                    for (var r2 = 0; r2 < this.filterSize; r2++) {

                        // フィルターの列に対し
                        for (var c2 = 0; c2 < this.filterSize; c2++) {

                            var nabla_w = 0.0;

                            // 出力の行に対し
                            for (var r1 = 0; r1 < this.imgRows; r1++) {

                                // 出力の列に対し
                                for (var c1 = 0; c1 < this.imgCols; c1++) {

                                    var delta_z_idx = channel_idx * RC + r1 * (this.imgCols | 0) + c1;
                                    var prev_activation_idx = prev_channel_idx * prev_RC + (r1 + r2) * (prev_Layer.imgCols | 0) + (c1 + c2);

                                    // バッチ内のデータに対し
                                    for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                                        var delta = this.deltaZ.dt[delta_z_idx];
                                        if (delta != 0) {

                                            nabla_w += delta * prev_Layer.activation.dt[prev_activation_idx];
                                        }

                                        delta_z_idx += this.unitSize;
                                        prev_activation_idx += (prev_Layer.unitSize | 0);
                                    }
                                }
                            }

                            this.deltaWeight.dt[weights_idx] = nabla_w;
                            weights_idx++;
                        }
                    }
                }
            }
            Assert(weights_idx == this.deltaWeight.dt.length);
        }

        cpuDeltaBiases(){
            var RC = this.imgRows * this.imgCols;

            // すべての特徴マップに対し
            for (var channel_idx = 0; channel_idx < this.channelSize; channel_idx++) {

                var nabla_b = 0.0;

                // 出力の行に対し
                for (var r1 = 0; r1 < this.imgRows; r1++) {

                    // 出力の列に対し
                    for (var c1 = 0; c1 < this.imgCols; c1++) {

                        // バッチ内のデータに対し
    //                    var delta_z_idx = channel_idx * (r1 * c1) + r1 * this.imgCols;
                        var delta_z_idx = channel_idx * RC + r1 * (this.imgCols | 0) + c1;
                        for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                            nabla_b += this.deltaZ.dt[delta_z_idx];
                            delta_z_idx += this.unitSize;

                            //!!!!!! 直前が入力層なら必要なし !!!!!
                            // this.costDerivative.dt[output_idx] = this.nextLayer.DeltaT[output_idx];
                        }
                    }
                }

                this.deltaBias.dt[channel_idx] = nabla_b;
            }
        }


        cpuDeltaX2() {
            var prev_Layer = this.prevLayer;
            var delta_x = new Float32Array(miniBatchSize * prev_Layer.unitSize);

            var prev_activation_dt = prev_Layer.activation.dt;
            var z_dt = this.z.dt;
            var activation_dt = this.activation.dt;

            // 出力先
            var output_idx = 0;

            // バッチ内のデータに対し
            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                // すべての特徴マップに対し
                for (var channel_idx = 0; channel_idx < this.channelSize; channel_idx++) {

                    // 出力の行に対し
                    for (var r1 = 0; r1 < this.imgRows; r1++) {

                        // 出力の列に対し
                        for (var c1 = 0; c1 < this.imgCols; c1++) {

                            var sum = 0.0;
                            var weight_idx = channel_idx * prev_Layer.channelSize * this.filterSize * this.filterSize;
                            var prev_activation_base = batch_idx * prev_Layer.channelSize * prev_Layer.imgRows * prev_Layer.imgCols;

                            // 入力のチャネルに対し
                            for(var prev_channel_idx = 0; prev_channel_idx < prev_Layer.channelSize; prev_channel_idx++){

                                // フィルターの行に対し
                                for (var r2 = 0; r2 < this.filterSize; r2++) {

                                    // フィルターの列に対し
                                    for (var c2 = 0; c2 < this.filterSize; c2++) {
                                        var prev_activation_idx = prev_activation_base + (r1 + r2) * prev_Layer.imgCols + (c1 + c2);
                                        sum += prev_activation_dt[prev_activation_idx] * this.weight.dt[weight_idx];

                                        delta_x[prev_activation_idx] += this.deltaZ.dt[output_idx] * this.weight.dt[weight_idx]
                                        weight_idx++;
                                    }
                                }
                                prev_activation_base += prev_Layer.imgRows * prev_Layer.imgCols;
                            }

                            output_idx++;
                        }
                    }
                }
            }

            return delta_x;
        }

        cpuDeltaX() {
            var prev_Layer = this.prevLayer;
            var RC = this.imgRows * this.imgCols;

            var prev_activation_idx = 0;

            // バッチ内のデータに対し
            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                // 入力のチャネルに対し
                for(var prev_channel_idx = 0; prev_channel_idx < prev_Layer.channelSize; prev_channel_idx++){

                    // 入力の行に対し
                    for (var r3 = 0; r3 < prev_Layer.imgRows; r3++) {

                        // 入力の列に対し
                        for (var c3 = 0; c3 < prev_Layer.imgCols; c3++) {

                            var sum = 0.0;

                            // 出力のチャネルに対し
                            for(var channel_idx = 0; channel_idx < this.channelSize; channel_idx++){
                                var delta_z_base = batch_idx * this.unitSize + channel_idx * RC;
                                var weight_base = (channel_idx * prev_Layer.channelSize + prev_channel_idx) * this.filterSize * this.filterSize;

                                // フィルターの行に対し
                                for (var r2 = 0; r2 < this.filterSize; r2++) {

                                    // 出力の行
                                    var r1 = r3 - r2;

                                    if(0 <= r1 && r1 < this.imgRows){

                                        // フィルターの列に対し
                                        for (var c2 = 0; c2 < this.filterSize; c2++) {

                                            // 出力の列
                                            var c1 = c3 - c2;

                                            if(0 <= c1 && c1 < this.imgCols){

                                                var delta_z_idx = delta_z_base + r1 * this.imgCols + c1;
                                                var weight_idx = weight_base + r2 * this.filterSize + c2;
                                                sum += this.deltaZ.dt[delta_z_idx] * this.weight.dt[weight_idx];
                                            }
                                        }
                                    }
                                }
                            }

                            this.deltaX.dt[prev_activation_idx] = sum;
                            prev_activation_idx++;
                        }
                    }
                }
            }
            Assert(prev_activation_idx == miniBatchSize * prev_Layer.unitSize);
        }


        Backpropagation() {
            var lap = new Lap(this.bwTime);

            if (this.nextLayer) {
                // 最後のレイヤーでない場合

                this.costDerivative = this.nextLayer.deltaX;
            }

            this.cpuDeltaZ();
            lap.Time();

            this.cpuDeltaBiases();
            lap.Time();

            this.gpuDeltaWeights();

            if(miniBatchIdx == 0 || Math.random() < 0.01){

                var delta_w = new Float32Array(this.deltaWeight.dt);
                this.cpuDeltaWeights();

                var diff = this.deltaWeight.diff(delta_w);
                if(0.00001 < diff){
                    console.log("CNN delta-W diff:%f", diff);
                }
            }

    //        AssertEq(gpu_nabla_weights, this.deltaWeight.dt);
            lap.Time();

            if(this.prevLayer instanceof InputLayer){

                return;
            }

            this.gpuDeltaX();

            if(miniBatchIdx == 0 || Math.random() < 0.01){

                var delta_x1 = new Float32Array(this.deltaX.dt);

                this.cpuDeltaX();

                var delta_x2 = this.cpuDeltaX2();
                var diff1 = this.deltaX.diff(delta_x1);
                var diff2 = this.deltaX.diff(delta_x2);
                if(0.00001 < diff1 || 0.00001 < diff2){
                    console.log("CNN delta-X diff:%f %f", diff1, diff2);
                }
            }

            lap.Time();
        }

        updateParameter() {
            var lap = new Lap(this.udTime);

            var prev_Layer = this.prevLayer;
            var eta = learningRate / miniBatchSize;

            var weights_idx = 0;

            // 出力のチャネルに対し
            for (var channel_idx = 0; channel_idx < this.channelSize; channel_idx++) {

                this.bias.dt[channel_idx] -= eta * this.deltaBias.dt[channel_idx];

                // 入力のチャネルに対し
                for(var prev_channel_idx = 0; prev_channel_idx < prev_Layer.channelSize; prev_channel_idx++){

                    // フィルターの行に対し
                    for (var r2 = 0; r2 < this.filterSize; r2++) {

                        // フィルターの列に対し
                        for (var c2 = 0; c2 < this.filterSize; c2++) {
                            this.weight.dt[weights_idx] -= eta * this.deltaWeight.dt[weights_idx];
                            weights_idx++;
                        }
                    }
                }
            }
            Assert(weights_idx == this.weight.dt.length);
            lap.Time();
        }

        clear(){
            for(var key in this.params){
                WebGL2.clear(this.params[key].id);
            }
            this.params = {};
        }
    }

    class MaxPoolingLayer extends Layer {
        constructor(filter_size) {
            super();
            this.filterSize = filter_size;
        }

        init(prev_layer) {
            super.init(prev_layer);

            Assert(this.prevLayer instanceof ConvolutionalLayer, "Pooling-Layer-init");

            this.channelSize = this.prevLayer.channelSize;
            this.imgRows = this.prevLayer.imgRows / this.filterSize;
            this.imgCols = this.prevLayer.imgCols / this.filterSize;

            Assert(Math.ceil(this.imgRows) == this.imgRows && Math.ceil(this.imgCols) == this.imgCols);

            this.unitSize = this.channelSize * this.imgRows * this.imgCols;
        }

        miniBatchSizeChanged(){
            super.miniBatchSizeChanged();

            this.activation = new ArrayView(miniBatchSize, this.unitSize);
            this.maxRow     = new Int8Array(miniBatchSize * this.unitSize);
            this.maxCol     = new Int8Array(miniBatchSize * this.unitSize);
            this.deltaX     = new ArrayView(miniBatchSize, this.prevLayer.unitSize);
        }

        forward() {
            var lap = new Lap(this.fwTime);

            var prev_Layer = this.prevLayer;

            var prev_activation_dt = prev_Layer.activation.dt;

            // 出力先
            var output_idx = 0;

            // バッチ内のデータに対し
            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                // すべての特徴マップに対し
                for (var channel_idx = 0; channel_idx < this.channelSize; channel_idx++) {

                    // 出力の行に対し
                    for (var r1 = 0; r1 < this.imgRows; r1++) {
                        var r0 = r1 * this.filterSize;

                        // 出力の列に対し
                        for (var c1 = 0; c1 < this.imgCols; c1++) {
                            var c0 = c1 * this.filterSize;

                            if(inGradientCheck){

                                var r2 = this.maxRow[output_idx];
                                var c2 = this.maxCol[output_idx];
                                var prev_activation_idx = batch_idx * prev_Layer.unitSize + (channel_idx * prev_Layer.imgRows + (r0 + r2)) * prev_Layer.imgCols + (c0 + c2);
                                this.activation.dt[output_idx] = prev_activation_dt[prev_activation_idx];
                            }
                            else{

                                var max_val = -10000;
                                var max_row, max_col;

                                // フィルターの行に対し
                                for (var r2 = 0; r2 < this.filterSize; r2++) {

                                    // フィルターの列に対し
                                    for (var c2 = 0; c2 < this.filterSize; c2++) {

                                        var prev_activation_idx = batch_idx * prev_Layer.unitSize + (channel_idx * prev_Layer.imgRows + (r0 + r2)) * prev_Layer.imgCols + (c0 + c2);
                                        var val = prev_activation_dt[prev_activation_idx];
                                        if (max_val < val) {

                                            max_val = val;
                                            max_row = r2;
                                            max_col = c2;
                                        }
                                    }
                                }

                                this.activation.dt[output_idx] = max_val;
                                this.maxRow[output_idx] = max_row;
                                this.maxCol[output_idx] = max_col;
                            }

                            output_idx++;
                        }
                    }
                }
            }

            Assert(output_idx == this.activation.dt.length);
            lap.Time();
        }

        Backpropagation() {
            var lap = new Lap(this.bwTime);

            var prev_Layer = this.prevLayer;

            Assert(this.activation.dt.length == this.nextLayer.deltaX.dt.length);

            for(var i = 0; i < this.deltaX.dt.length; i++){
                this.deltaX.dt[i] = 0;
            }
            lap.Time();

            // 出力先
            var output_idx = 0;

            // バッチ内のデータに対し
            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {

                // すべての特徴マップに対し
                for (var channel_idx = 0; channel_idx < this.channelSize; channel_idx++) {
                    var offset = batch_idx * prev_Layer.unitSize + channel_idx * prev_Layer.imgRows * prev_Layer.imgCols;

                    // 出力の行に対し
                    for (var r1 = 0; r1 < this.imgRows; r1++) {
                        var r0 = r1 * this.filterSize;

                        // 出力の列に対し
                        for (var c1 = 0; c1 < this.imgCols; c1++) {
                            var c0 = c1 * this.filterSize;

                            var delta = this.nextLayer.deltaX.dt[output_idx];
                            if(delta != 0){

                                var r2 = this.maxRow[output_idx] | 0;
                                var c2 = this.maxCol[output_idx] | 0;
                                var prev_activation_idx = offset + (r0 + r2) * prev_Layer.imgCols + (c0 + c2);
                                this.deltaX.dt[prev_activation_idx] += delta;
                            }

                            output_idx++;
                        }
                    }
                }
            }

            Assert(output_idx == this.nextLayer.deltaX.dt.length);
            lap.Time();
        }
    }


    class DropoutLayer extends Layer {
        constructor(ratio) {
            super();
            this.ratio = ratio;
        }

        init(prev_layer) {
            super.init(prev_layer);
            this.unitSize = prev_layer.unitSize;
        }


        miniBatchSizeChanged(){
            super.miniBatchSizeChanged();

            this.activation = new ArrayView(miniBatchSize, this.unitSize);
            this.deltaX     = new ArrayView(miniBatchSize, this.unitSize);
            this.valid      = new Int8Array(miniBatchSize * this.unitSize);
        }

        forward() {
            var lap = new Lap(this.fwTime);
            for(var i = 0; i < this.activation.dt.length; i++){
                if(! isTest){

                    if(this.ratio <= Math.random()){

                        this.valid[i]   = 1;
                        this.activation.dt[i] = this.prevLayer.activation.dt[i];
                    }
                    else{

                        this.valid[i]   = 0;
                        this.activation.dt[i] = 0;
                    }
                }
                else{

                    this.activation.dt[i] = (1 - this.ratio) *  this.prevLayer.activation.dt[i];
                }
            }
            lap.Time();
        }

        Backpropagation() {
            var lap = new Lap(this.bwTime);
            for(var i = 0; i < this.activation.dt.length; i++){
                if(this.valid[i] == 1){

                    this.deltaX.dt[i] = this.nextLayer.deltaX.dt[i];
                }
                else{

                    this.deltaX.dt[i] = 0;
                }
            }
            lap.Time();
        }
    }

    class NeuralNetwork {
        constructor(gpgpu) {
            WebGL2 = gpgpu;
            this.trainingCost = [];
            this.testCost = [];
            this.trainingAccuracy = [];
            this.testAccuracy = [];
        }

        setLayers(layers){
            this.layers = layers;
            this.lastLayer = layers[layers.length - 1];

            var prev_layer = null;
            for(let layer of layers) {
                layer.init(prev_layer);
                prev_layer = layer;
            }
        }

        InputLayer(channel_size, rows, cols){
            return new InputLayer(channel_size, rows, cols);
        }

        FullyConnectedLayer(size, activation_function){
            return new FullyConnectedLayer(size, activation_function);
        }

        ConvolutionalLayer(filter_size, channel_size, activation_function){
            return new ConvolutionalLayer(filter_size, channel_size, activation_function);
        }

        MaxPoolingLayer(filter_size){
            return new MaxPoolingLayer(filter_size);
        }

        DropoutLayer(ratio){
            return new DropoutLayer(ratio);
        }

        Laminate(data, idx_list, idx_start, idx_cnt) {
            var element_size = data.shape.slice(1).reduce((x, y) => x * y);

            var shape = data.shape.slice();
            shape[0] = idx_cnt;
            var X = new ArrayView(shape);
            var dst = 0;
            for (var idx = idx_start; idx < idx_start + idx_cnt; idx++) {
                var src = idx_list[idx] * element_size;
                for (var i = 0; i < element_size; i++) {
                    X.dt[dst] = data.dt[src];
                    src++;
                    dst++;
                }
            }

            return X;
        }

        CorrectCount(Y){
            var result = this.lastLayer.activation;

            var ok_cnt = 0;
            // バッチ内のデータに対し
            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++) {
                var max_idx = argmax(result.Row(batch_idx));
                if(Y.dt[batch_idx * this.lastLayer.unitSize + max_idx] == 1){
                    ok_cnt++;
                }
            }

            return ok_cnt;
        }

        /*
        最小二乗誤差の微分
        */
        LeastSquaresDelta(cost_derivative, last_y, batch_Y) {
            for(var i = 0; i < cost_derivative.length; i++){
                cost_derivative[i] = last_y[i] - batch_Y[i];
            }
        }

        /*
        損失関数の微分
        */
        SoftMax(cost_derivative, last_y, batch_Y, exp_work, range_len, batch_idx) {
            var cost_sum = 0;

            var max_val = -10000;
            for (var i = 0; i < range_len; i++) {
                var k = batch_idx * range_len + i;

                if (max_val < last_y[k]) {
                    max_val = last_y[k];
                }
            }

            var sum = 0;
            for (var i = 0; i < range_len; i++) {
                var k = batch_idx * range_len + i;

                var d = Math.exp(last_y[k] - max_val);
                if(! isFinite (d) || d <= 0){
                    d = 0.0000001;
                }
                sum += d;
                exp_work[i] = d;
            }

            for (var i = 0; i < range_len; i++) {
                var k = batch_idx * range_len + i;

                var y = exp_work[i] / sum;
                cost_derivative[k] = y - batch_Y[k];

                var log_y = Math.log(y);
                if(! isFinite (log_y)){
                    continue;
                }

                cost_sum += (batch_Y[k] * log_y);

                if(! isFinite(cost_sum)){
                    Assert(false);
                }
            }

            return - cost_sum;
        }

        TestSoftMax(cost_derivative, last_y, batch_Y, exp_work, range_len){
            var costs = this.SoftMax(cost_derivative, last_y, batch_Y, exp_work, range_len);

            var cost_derivative_work = new Float32Array(cost_derivative.length);

            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++){
                for (var i = 0; i < range_len; i++) {
                    var k = batch_idx * range_len + i;

                    var y = last_y[k];
                    var eps = y * 0.01;

                    last_y[k] = y - eps;
                    var cost1 = this.SoftMax(cost_derivative_work, last_y, batch_Y, exp_work, range_len, batch_idx);

                    last_y[k] = y + eps;
                    var cost2 = this.SoftMax(cost_derivative_work, last_y, batch_Y, exp_work, range_len, batch_idx);

                    var diff = cost_derivative[k] * 2 * eps - (cost2 - cost1);
                    console.log("diff:%f dC:%f eps:%f cost1,2:%f,%f", diff, cost_derivative[k], eps, cost2, cost1);
                }

            }

        }

        forwardCost(batch_Y, exp_work, batch_idx, layer_idx, cost_derivative) {
            var last_layer = this.layers[this.layers.length - 1];

            for(; layer_idx < this.layers.length; layer_idx++){
                this.layers[layer_idx].forward();
            }

            return this.SoftMax(cost_derivative, last_layer.activation.dt, batch_Y, exp_work, last_layer.unitSize, batch_idx);
        }

        ParamGradientCheck(name, params, delta_params, batch_Y, exp_work, cost, batch_idx, layer_idx, cost_derivative_work){
            Assert(params.length == delta_params.length);
            // delta bias
            var idx_list = random.RandomSampling(params.length, 3);

            for(let i of idx_list){
                var b = params[i];
                var db = delta_params[i];
                var eps = b * 0.01;

                params[i] = b - eps;
                var cost1 = this.forwardCost(batch_Y, exp_work, batch_idx, layer_idx, cost_derivative_work);

                params[i] = b + eps;
                var cost2 = this.forwardCost(batch_Y, exp_work, batch_idx, layer_idx, cost_derivative_work);

                var diff = db * 2 * eps - (cost2 - cost1);
                console.log("delta-%s : %f dC:%f eps:%f cost:%f,%f,%f", name, diff, db, eps, cost1, cost - cost1, cost2 - cost);

                params[i] = b;
            }
        }

        netGradientCheck(batch_Y, exp_work, costs){
            inGradientCheck = true;

            var last_layer = this.layers[this.layers.length - 1];
            var last_cost_derivative = new Float32Array(last_layer.costDerivative.dt);

            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++){
                last_layer.costDerivative.dt = new Float32Array(last_cost_derivative.length);

                for(var i = 0; i < last_layer.unitSize; i++){
                    var k = batch_idx * last_layer.unitSize + i;
                    last_layer.costDerivative.dt[k] = last_cost_derivative[k];
                }

                for (var i = this.layers.length - 1; 1 <= i; i--) {
                    this.layers[i].Backpropagation();
                }

                for(var layer_idx = 0; layer_idx < this.layers.length; layer_idx++){

                    console.log("勾配確認 %s", this.layers[layer_idx].constructor.name);
                    this.layers[layer_idx].GradientCheck(this, batch_Y, exp_work, costs[batch_idx], batch_idx, layer_idx);
                }
            }

            inGradientCheck = false;
        }


        * SGD(training_data, test_data, epochs, mini_batch_size, learning_rate) {
            learningRate = learning_rate;
            var last_layer = this.layers[this.layers.length - 1];
            last_layer.costDerivative = new ArrayView(mini_batch_size, last_layer.unitSize);
            var exp_work = new Float32Array(last_layer.unitSize);

            for (var epoch_idx = 0; epoch_idx < epochs; epoch_idx++) {
                this.trainingAccuracy.push(0);
                this.testAccuracy.push(0);

                for(var mode = 0; mode < 2; mode++){
                    var data;
                    var ok_cnt = 0;

                    isTest = (mode == 1);
                    if(mode == 0){

                        data = training_data;
                        miniBatchSize = mini_batch_size;
                    }
                    else{

                        data = test_data;
                        miniBatchSize = 10 * mini_batch_size;
                    }
                    var costs = new Float32Array(miniBatchSize);

                    this.layers.forEach(x => x.miniBatchSizeChanged());

                    var idx_list = random.RandomSampling(data.X.shape[0]);

                    var show_time = new Date();

                    var data_cnt = data.X.shape[0];
                    TrainingDataCnt = data_cnt;
                    var mini_batch_cnt = Math.floor(data_cnt / miniBatchSize);
                    var mini_batch_time = [];
                    for (var idx = 0; idx < mini_batch_cnt; idx++) {
                        miniBatchIdx = idx;
                        var lap = new Lap(mini_batch_time);
                        var X = this.Laminate(data.X, idx_list, idx * miniBatchSize, miniBatchSize);
                        lap.Time();
                        var Y = this.Laminate(data.Y, idx_list, idx * miniBatchSize, miniBatchSize);
                        lap.Time();

                        this.layers[0].activation = X;
                        this.layers.forEach(x => x.forward());
                        lap.Time();

                        if(useSoftMax){

                            for (var batch_idx = 0; batch_idx < miniBatchSize; batch_idx++){

                                costs[batch_idx] = this.SoftMax(last_layer.costDerivative.dt, last_layer.activation.dt, Y.dt, exp_work, last_layer.unitSize, batch_idx);
                            }

                            var avg_costs = costs.reduce((x,y) => x + y) / miniBatchSize;
                            //if(mode == 0){

                            //    this.trainingCost.push( avg_costs );
                            //}
                            //else{

                            //    this.testCost.push( avg_costs );
                            //}
                        }
                        else{

                            this.LeastSquaresDelta(last_layer.costDerivative.dt, last_layer.activation.dt, Y.dt);
                        }

                        if(mode == 0){
                            lap.Time();

                            if(useGradientCheck){

                                this.netGradientCheck(Y.dt, exp_work, costs);
                            }
                            else{

                                for (var i = this.layers.length - 1; 1 <= i; i--) {
                                    this.layers[i].Backpropagation();
                                }
                            }

                            lap.Time();

                            this.layers.forEach(x => x.updateParameter());
                            lap.Time();
                        }

                        ok_cnt += this.CorrectCount(Y);

                        if (60 * 1000 < new Date() - show_time || idx == mini_batch_cnt - 1) {

                            var s = Stats(mini_batch_time, idx);
                            for(let layer of this.layers.slice(1)) {
                                s += " (" + Stats(layer.fwTime, idx) + " " + Stats(layer.bwTime, idx) + " " + Stats(layer.udTime, idx) + ")";
                            }
                            var accuracy = ok_cnt / (idx * miniBatchSize);
                            console.log("update mini batch: %.2f %d  %s", accuracy, idx * miniBatchSize, s);

                            if(mode == 0){

                                this.trainingAccuracy[epoch_idx] = accuracy;
                            }
                            else{

                                this.testAccuracy[epoch_idx] = accuracy;
                            }

                            show_time = new Date();

                            yield 1;
                        }
                    }

                    this.layers.forEach(x => x.clear());

                    if(mode == 1){

                        console.log("Epoch %d  %d / %d eta:%.02f", epoch_idx, ok_cnt, mini_batch_cnt * miniBatchSize, learningRate);
                    }
                }

                yield 2;
            }

            yield 0;
        }
    }

    function sigmoid_primeF(z) {
        var f = sigmoidF(z);
        return f * (1 - f);
    }

    //??
    function sigmoidF(z){
        return 1.0 / (1.0 + Math.exp(-z));
    }

    return new NeuralNetwork(gpgpu);
}